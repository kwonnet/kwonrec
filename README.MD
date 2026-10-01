# KwonRec

Kwonnet's online hybrid newsfeed recommender. Python 3.12, FastAPI, Redis (standalone or Cluster), and a PostgreSQL transactional outbox connected to the existing `kwonserver` Prisma schema.

This implementation updates user interests and post co-engagement statistics continuously from interactions. Its final ranking coefficients are fixed; it does not train or serve a neural-network model. It combines item-to-item collaborative filtering, token/topic content similarity, freshness, decayed engagement popularity, following boosts, and seeded exploration. New users receive recent/popular public posts; new posts can be retrieved immediately after indexing.

**Capacity is not a claim:** this repository provides bounded retrieval, independently scalable API/worker processes, and cluster-compatible state. Millions of registered users are an infrastructure sizing and workload-validation target, not a measured guarantee. The included single-node Compose setup is for development/staging. No production database has been migrated or deployment performed automatically.

## Run locally

For the single-VM deployment, see [Compute Engine deployment](deploy/COMPUTE_ENGINE.md).

```sh
cp .env.example .env
# Set a random KWONREC_API_KEY of at least 32 characters in .env.
# Example key generator: python3 -c 'import secrets; print(secrets.token_hex(32))'
docker compose up --build -d
curl http://localhost:8001/ready
```

All recommendation, mutation, NLP, and metrics endpoints require `Authorization: Bearer <KWONREC_API_KEY>`. `/health` and `/ready` are unauthenticated probes. Explicit `KWONREC_ALLOW_UNAUTHENTICATED=true` is available for local Python development only. The API fails startup if neither a key nor that opt-out is configured. Redis must be available at startup.

## Connect to kwonserver

1. Set `KWONREC_DATABASE_URL` to **kwonserver's PostgreSQL database**. Do not point it at ClickHouse. Within Docker Desktop, host services use `host.docker.internal`, not `localhost`.
2. Review and apply `sql/001_outbox.sql` to that database using a migration role. Test on staging first; trigger creation briefly locks source tables. It creates a separate `kwonrec` schema plus triggers on `Post` and interaction tables. It does not change Prisma models.

   ```sh
   psql "$KWONREC_DATABASE_URL" -v ON_ERROR_STOP=1 -f sql/001_outbox.sql
   ```

3. Bootstrap the recent catalog and enqueue up to seven days of existing interactions. Install triggers **before** bootstrap so concurrent writes are captured. The bootstrap uses bounded keyset pages and is safe to restart. Historical interaction source IDs are unique against live capture.

   ```sh
   docker compose --profile connected run --rm worker python -m src.runtime.worker --bootstrap
   docker compose --profile connected run --rm worker python -m src.runtime.worker --bootstrap-history
   docker compose --profile connected up -d worker
   ```

4. Set `KWONREC_API` and the matching `KWONREC_API_KEY` in kwonserver (see `integration/kwonserver.env.example`), then restart it. For same-network containers use `http://kwonrec:8001`.

The accompanying kwonserver changes replace hardcoded URLs with `src/services/kwonrec.ts`, use the authenticated user for personalization, remove the two-minute feed cache, retain recommendation ordering, and fall back to a bounded chronological query after a 1.5-second service timeout. The recommendations route now requires authentication. Empty feeds return an empty list.

Database inserts for likes, bookmarks, shares, clicks, views, impressions, tips, reports, disinterest, replies, quotes, and reposts append an outbox row **in the same source transaction**. Rollbacks roll back the event. Content/status/scope/deletion changes enqueue reindexing; counter-only updates do not. Workers poll every 250 ms by default. This is near-real-time, with actual latency depending on queue depth and infrastructure. Views use duration; impressions suppress repeats but do not train positive preference.

Triggers currently capture interaction **inserts**, not undo operations (unlike/unbookmark), and feedback expires/decays according to the retention policy. Removing a negative signal in the app does not immediately erase its retained recommendation suppression. Update-only dwell tracking must send `/v1/events` explicitly or be changed to append views. These semantics should be aligned with product requirements before broad rollout.

### Authoritative visibility

KwonRec is a **private backend service**, not a browser API. It returns IDs, not post bodies. kwonserver rechecks PostgreSQL at hydration time for publication, public scope, hidden/deleted posts, private/inactive authors, blocks in both directions, mutes, reports, and disinterest, including immediate parent/root visibility. Its existing rendering/authorization rules must continue to protect any deeper embedded post chains. KwonRec does not support restricted-audience candidate discovery in this version; only `ANYONE` posts enter the index.

Do not treat a cached recommendation as authorization. Account visibility changes can leave stale IDs/features in Redis until reindexing/expiry; the source hydration gate prevents those posts from being returned. Privacy/account-erasure workflows must additionally purge derived user/item state and source outbox records as required by the application's retention policy; TTL alone is not immediate erasure.

## API

`GET /recommend/{user_id}?limit=20` keeps the existing response contract:

```json
{"user_id":"user-1","recommendations":[{"id":"post-1","score":0.42}],"model_version":"online-itemcf-content-v2","exhausted":false}
```

Additional result fields include candidate count, retrieval sources, exploration flags, and generation time. Limit is 1–100. No endless refill loop: a short or empty result has `exhausted: true`.

`POST /v1/recommendations` also accepts onboarding interests, followed authors, blocked authors, explicit post exclusions, and a reproducible exploration seed:

```json
{"user_id":"user-1","limit":20,"interests":["technology"],"following_author_ids":["author-1"],"blocked_author_ids":[],"exclude_ids":[],"seed":0}
```

`PUT /v1/posts/post-1` supports alternative publishers without PostgreSQL triggers:

```json
{"id":"post-1","author_id":"author-1","content":"A new space telescope","topics":["science"],"created_at":"2026-09-29T10:00:00Z","version":1,"scope":"ANYONE","status":"PUBLISHED","kind":"ROOT"}
```

Use monotonically increasing versions per post, the same ID on edits, and `deleted: true` or the appropriate visibility/status on removal. Reusing a post ID after its tombstone expires is unsupported. Catalog lifetime is 30 days. API publishers must not mix their own version domain with the PostgreSQL worker's microsecond versions for the same post.

`POST /v1/events` applies up to 100 interactions synchronously, each with a stable event ID:

```json
{"events":[{"event_id":"like-123","user_id":"user-1","post_id":"post-1","type":"like","occurred_at":"2026-09-29T10:00:01Z","duration_seconds":0}]}
```

Use current timestamps in real requests. Events must be within seven days and no more than five minutes in the future. Results report `applied`, `missing_post` (retryable after indexing), or `rejected`. Retry network failures with the **same payload and event IDs**, including partially successful batches. A conflicting event ID is rejected. The caller must retain/retry direct API events durably; use the outbox when connected to kwonserver. Do not publish the same interaction through both paths with different IDs.

`POST /keywords` and `/classify` preserve the auxiliary response shapes with deterministic lexical processing (`method: lexical-v1`). This is **not semantic zero-shot classification**. No matching label yields score 0 and `matched: false`; kwonserver's jobs avoid assigning a topic in that case. Heavy transformer inference should run in a separately provisioned enrichment worker if needed. `/train` is intentionally removed: learning occurs per event, outside API startup jobs.

## Retrieval and learning

- Each post has at most 32 normalized content features: 28 token features plus four topics. No raw post body is stored in the serving document.
- A user has at most 128 signed interest features, a 50-item positive history, and 5,000-entry seen and negative lists. Interests decay over seven days, states expire after 30 idle days, and aged list members are filtered/pruned.
- New distinct positive user-item engagements increment co-engagement edges to up to 20 recent positive items. Both directions are updated; each item's strongest 200 neighbors are retained. Repeated same-user/post/type signals contribute at most once per 24 hours. Negative feedback removes that seed and suppresses later positive feedback while the negative entry is retained.
- User/item mutations have separate idempotency receipts, retained for eight days. Multi-shard delivery is a retryable sequence, not a cross-shard transaction. A failed request may have updated some stages; replay finishes the remaining stages without repeating completed increments. Replay beyond the accepted event window is rejected.
- Global recent/popularity indexes are striped across eight keys to reduce hot-key contention. Candidate gathering round-robins recent, six hourly popularity buckets, eight interest indexes, ten collaborative neighborhoods, and up to 20 followed-author indexes. At most 1,000 candidate documents are fetched/ranked by default, independent of total user count.
- Ranking blends 35% collaborative, 35% content, 15% freshness, 10% popularity, and 5% following. A maximum of three posts per author and 10% seeded exploration encourage diversity. Scores are heuristic relevance scores, not click probabilities. The item graph has TTL and bounded neighbor counts; old edges on continuously active items are not time-decayed individually.
- Concurrent requests atomically reserve distinct IDs in the user's seen list. A successful feed fetch counts as served, even if the client never renders it. Thus a lost response/retry can skip that page. There is no durable cursor or feed-request replay contract. After retention expiry or list-cap eviction, items can be eligible again.

All Lua script keys use one entity's Redis hash tag; cross-slot retrieval uses pipelined GETs. No full catalog scan or training occurs on a feed request. Receipts and indexes are bounded in lifetime, but total storage still grows with active users, posts, and event rate.

## Reliability and operations

Outbox consumers use `FOR UPDATE SKIP LOCKED` and one row per transaction. A per-post advisory lock serializes authoritative rereads during indexing. Workers commit `processed_at` only after Redis work succeeds; failures back off up to five minutes and enter a dead-letter state after ten attempts. Validation/age rejection and exhausted retries require operator attention. Source tables remain authoritative. The outbox provides durable capture as long as PostgreSQL is available; Redis AOF with `everysec` can lose recent acknowledged state during a hard failure. It is not an exactly-once durable log.

Example operational queries (run on the source DB):

```sql
-- Backlog and event age. Export these through your PostgreSQL monitoring agent.
SELECT count(*), extract(epoch FROM (now() - min(created_at))) AS oldest_seconds
FROM kwonrec.outbox WHERE processed_at IS NULL AND dead_at IS NULL;
SELECT id, kind, attempts, error FROM kwonrec.outbox WHERE dead_at IS NOT NULL ORDER BY id LIMIT 100;
-- Requeue a specific repaired dead letter, within the accepted event age window:
UPDATE kwonrec.outbox SET dead_at=NULL, attempts=0, available_at=now() WHERE id = 123;
-- Periodic bounded retention cleanup; never trim pending/dead-letter events implicitly:
DELETE FROM kwonrec.outbox WHERE id IN (
  SELECT id FROM kwonrec.outbox WHERE processed_at < now() - interval '8 days'
  ORDER BY processed_at LIMIT 10000
);
```

Rebuild a lost Redis index using a **new namespace** during a controlled cutover. Stop all live workers first, configure the new namespace, run `--bootstrap`, `--bootstrap-history`, then `python -m src.runtime.worker --replay-retained`. This replays retained outbox rows without changing their delivery status. Switch API and workers together and resume consumption; new source writes remain pending throughout the rebuild. Keep the old API in maintenance/fallback mode during cutover. Expired or already-pruned interactions cannot be recovered from the outbox. Do not run independent production namespaces against the same consumed outbox without separate replay/cursor coordination. Test recovery/failover with your actual persistence and replica settings.

`/metrics` exports request latency, request statuses, event outcomes, and candidate counts with bounded label cardinality. Protect it with the service key. API metrics are **per process**; the supplied container runs one Uvicorn process per pod for per-pod aggregation. Configure Prometheus multiprocess collection if increasing that count. Monitor queue age, dead letters, Redis memory/evictions, empty feeds, fallback rate in kwonserver, and source DB latency as well. API readiness probes Redis; it does not imply that bootstrap is complete or the outbox is caught up.

Use Redis `noeviction`, persistence, replicas, backups, memory headroom, private networking, TLS/ACLs, and a least-privileged PostgreSQL worker role (`SELECT` on source tables and `SELECT/INSERT/UPDATE` on the outbox, plus sequence usage for backfill). Source application roles need outbox INSERT, schema USAGE, and sequence privileges for the triggers. The migration role requires trigger/table creation privileges. Monitor write failures rather than evicting idempotency keys to relieve pressure. Put service authentication, request rate limits, connection limits, and TLS at the internal gateway; this service is not an untrusted public event collector.

`deploy/kubernetes.yaml` supplies an API Deployment, Service, HPA, disruption budget, and independently scaled worker Deployment. Replace the image with an immutable digest, provision the named Secret, and use external Redis Cluster (database 0). It is a deployment template, not a complete managed database/network/monitoring installation. Keep all settings, especially namespace, retention and catalog shard count, identical across API and workers. Changing the catalog shard count requires a rebuild.

## Validate and measure

```sh
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements-dev.txt
# Disposable test services; tests require a database named kwonrec_test.
docker run -d --name kwonrec-test-redis -p 127.0.0.1:16379:6379 redis:7.4-alpine
docker run -d --name kwonrec-test-postgres -p 127.0.0.1:15432:5432 \
  -e POSTGRES_PASSWORD=kwonrec_test -e POSTGRES_DB=kwonrec_test postgres:17-alpine
pytest -q
ruff check src/runtime src/main.py tests scripts
```

Integration tests use real Redis Lua and PostgreSQL triggers, including rollback, retry recovery after partial shard updates, duplicate delivery, concurrent consumers/feeds, negative feedback, visibility, timezones, cold starts, bounded inputs, auth, and outage behavior. They fail rather than silently skip when infrastructure is absent. The PostgreSQL test fixture uses a minimal representative source schema, not a complete production database restore.

For ranking evaluation, supply recent sanitized chronological training JSONL records (`kind: post|event`, `data: API payload`) and one later holdout per user (`user_id`, `relevant_ids`). Publish all candidate posts before their interactions. No future positive interaction may enter training. Run:

```sh
python -m scripts.evaluate train.jsonl holdout.jsonl --k 20
python -m scripts.loadtest --url http://localhost:8001 --requests 10000 --concurrency 50 --users 10000
```

Evaluation uses a disposable namespace with exploration off and reports Recall@K, NDCG@K, and unique recommended items. Ensure temporal/catalog eligibility and compare with chronological/popular baselines; no relevance lift is claimed without real holdout data. The load probe reports p50/p95/p99, RPS, errors, and empty responses separately. Seed representative posts, histories, content distribution, active users, and event writes before benchmarking; an empty-catalog test is not meaningful capacity evidence.

Before a millions-user launch, measure peak feed RPS and events/sec, per-user/item memory, source outbox WAL/lock cost, tail latency under hot authors, ingestion lag, shard failover, and recovery. Size replicas and Redis shards from those results and keep headroom. Registered user count alone does not determine capacity. PostgreSQL outbox throughput and retained per-event receipts can become the next bottlenecks; CDC/broker ingestion is an evolution path when measured limits require it.

## Active source layout

- `src/main.py`: HTTP API, authentication, health checks, and metrics.
- `src/runtime/engine.py`: indexing, online learning, retrieval, and ranking.
- `src/runtime/features.py`: text features and interaction weights.
- `src/runtime/scripts.py`: atomic Redis operations and duplicate handling.
- `src/runtime/worker.py`: outbox consumption, bootstrap, and recovery replay.
- `src/runtime/schemas.py`: validated post, event, and feed contracts.
- `src/runtime/settings.py`: active `KWONREC_` environment configuration.
- `src/stopwords.txt`: shared stopwords for content features.
- `sql/`: PostgreSQL outbox migration.
- `scripts/`, `tests/`, `integration/`, and `deploy/`: evaluation, verification, integration, and deployment support.

Recommendation state resides in versioned Redis keys and is reconstructed from retained source data using the documented recovery workflow.

### Additional integration checks

With kwonserver dependencies installed, run its adapter tests from the parent workspace:

```sh
node --test kwonrec/integration/kwonserver.test.cjs
```

A three-primary Redis Cluster smoke test is included (no replicas; test use only):

```sh
docker build -t kwonrec:local-test .
docker network create kwonrec-test-cluster
docker compose -f deploy/cluster-test.yaml up -d --wait redis-a redis-b redis-c
docker compose -f deploy/cluster-test.yaml run --rm init
docker compose -f deploy/cluster-test.yaml run --rm smoke
docker compose -f deploy/cluster-test.yaml down
docker network rm kwonrec-test-cluster
```

The source kwonserver TypeScript configuration currently includes `prisma/seed.ts` outside its `rootDir: src`; `npx tsc --noEmit --rootDir .` verifies the integration while accounting for that existing configuration issue.

Reference semantics: [Redis script atomic execution](https://redis.io/docs/latest/develop/programmability/eval-intro/), [Redis Cluster hash slots](https://redis.io/docs/latest/operate/oss_and_stack/reference/cluster-spec/), and [PostgreSQL queue locking](https://www.postgresql.org/docs/current/sql-select.html).

### Connect the local kwonserver newsfeed

Set `KWONREC_API=http://127.0.0.1:8001` in `kwonserver/.env`, and use the same random `KWONREC_API_KEY` in both projects. In `kwonrec/.env`, set `KWONREC_DATABASE_URL` to kwonserver's PostgreSQL database. For Neon, use its **direct (unpooled)** endpoint for this long-running worker; Prisma's `schema` URL option is supported. Never point this at ClickHouse.

From the kwonrec directory:

```sh
docker compose build
docker compose run --rm setup
docker compose --profile connected up -d kwonrec worker
```

The setup command installs the transactional outbox triggers, indexes posts from the last 30 days, and imports interactions from the last 7 days. It can be rerun after interruption. Run it before starting the worker. Compose uses its own persistent Redis for both the API and worker, regardless of the host Redis URL in `.env`; do not mix a host worker with a Compose API using another Redis.

Restart kwonserver after changing its environment. Its existing authenticated `GET /api/v1/posts/feed/:feedType?limit=21` endpoint calls kwonrec, hydrates ranked IDs from PostgreSQL, and rechecks visibility. `X-Feed-Source: kwonrec` identifies recommendation responses; `fallback` identifies the chronological fallback during service failures. The JSON response remains the existing post array.

From the repository parent, `node kwonrec/integration/live-kwonserver.cjs` checks the actual server adapter → HTTP API → Redis → PostgreSQL visibility path without modifying application rows. It uses a temporary recommendation profile that expires normally. An empty result is valid when there are no eligible recent posts. Use `docker compose logs --tail=50 worker` to diagnose delivery errors.


https://ssh.cloud.google.com/v2/ssh/projects/kwonnet-b3857/zones/europe-west1-b/instances/instance-20261001-091042?authuser=0&hl=en_US&projectNumber=615224090920&useAdminProxy=true

