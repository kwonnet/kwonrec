# Automated Compute Engine deployment

Push main or run **Deploy kwonrec to Compute Engine** in GitHub Actions. Tests gate
release. GitHub builds without secrets, publishes an immutable Artifact Registry
image, uploads via IAP, and runs Compose automatically. No manual Compose step.

## GitHub setup

Create the **production** environment in kwonnet/kwonrec → Settings → Environments.
Add these environment secrets (same convention as kwonserver):

- GCP_COMPUTE_ENGINE_PROJECT: project ID
- GCP_COMPUTE_ENGINE_NAME: target VM name
- GCP_COMPUTE_ENGINE_ZONE: VM zone
- GCP_WIF_PROVIDER: full projects/NUMBER/.../providers/PROVIDER resource name
- GCP_DEPLOY_SERVICE_ACCOUNT: deployment service account email
- GAR_LOCATION: region of the existing Docker repository named kwonnet
- KWONREC_ENV: complete multiline runtime contents, NOT a filename
- GCP_COMPUTE_ENGINE_EXTERNAL_IP: optional inventory value; IAP does not use it

Example KWONREC_ENV (replace placeholders; no surrounding quotes or export):

```dotenv
KWONREC_API_KEY=REPLACE_WITH_SHARED_RANDOM_KEY_OF_AT_LEAST_32_CHARACTERS
KWONREC_DATABASE_URL=postgresql://USER:PASSWORD@DATABASE_HOST/kwonnet?sslmode=require
KWONREC_REDIS_URL=redis://redis:6379/0
KWONREC_NAMESPACE=kwonrec:v2
KWONREC_BIND_IP=127.0.0.1
KWONREC_ALLOW_UNAUTHENTICATED=false
KWONREC_POLL_SECONDS=1
KWONREC_REDIS_MAX_CONNECTIONS=16
```

Other .env.example settings pass through unchanged. Dollar signs are preserved
using Compose raw env-file format. localhost URLs are rejected: those would refer
to each container. Bundled Redis uses redis://redis:6379/0. The database must be
kwonserver's PostgreSQL and the role needs trigger/schema setup permissions.

## Google permissions

The kwonnet Docker repository must exist in GAR_LOCATION. Deployment requires the
same Artifact Registry Writer, Compute SSH/sudo and IAP permissions as kwonserver.
The VM identity needs Logs Writer and logging.write/cloud-platform scope; a
preflight checks this before replacement. Allow IAP SSH from 35.235.240.0/20.

**WIF must trust kwonnet/kwonrec too.** If sharing kwonserver's provider, retain the
subject and repository mappings, and set its attribute condition to:

```text
attribute.repository in ['kwonnet/kwonserver', 'kwonnet/kwonrec']
```

On the deployment service account, grant Workload Identity User to:

```text
principalSet://iam.googleapis.com/projects/POOL_PROJECT_NUMBER/locations/global/workloadIdentityPools/POOL_ID/attribute.repository/kwonnet/kwonrec
```

Use the pool project NUMBER. Retain the kwonserver binding. Alternatively use a
separate provider restricted to kwonnet/kwonrec. No service-account key is needed.

## Deployment and connection

Supports Debian/Ubuntu x86-64, including Debian 13. Installs Docker and Compose if
missing. Compose 2.30+ is required. Budget at least 4 GiB RAM and free disk for images.

Uses /opt/kwonrec, project kwonrec-production and existing volume
kwonrec-production_redis_data. Confirm that this project/volume belongs to your
existing recommender before first deployment. No volume deletion or global pruning.
It leaves kwonserver, wallet Redis and Caddy alone. API joins kwonserver-production.

Stop workers on OTHER VMs/Cloud Run consuming this outbox into a different Redis
before moving the service. Back up Redis. Setup installs triggers each release;
first automation run bootstraps and replays retained history. Replay processes events
one by one and can be slow with a remote database. CI streams batch last_id/high_water
counters and a 30-second process heartbeat; a heartbeat alone does not prove progress. Later runs skip the
backfill when database/Redis/namespace settings are unchanged. A failed setup does
not advance its marker. After Redis loss, diagnose/restore it, remove
/opt/kwonrec/setup-target and rerun to rebuild retained data.

For kwonserver on the SAME VM, update KWONSERVER_ENV and redeploy it:

```dotenv
KWONREC_API=http://kwonrec:8001
KWONREC_API_KEY=THE_SAME_KEY_AS_KWONREC
```

For a DIFFERENT VM, use the recommender's private IP for KWONREC_BIND_IP and
http://PRIVATE_IP:8001 for KWONREC_API. Allow TCP 8001 only from the backend source.
The workflow creates no public firewall rules. Do not use a public or wildcard bind.

## Verification and operations

Deployment checks Redis/API readiness, runs a real database outbox batch, tests an
authenticated recommendation and checks worker liveness/restarts. This verifies
startup/connectivity, not sustained throughput or complete backlog drainage.

API/worker logs go to Google Cloud Logging (gce_instance; filter this VM and
container names containing kwonrec-production). Local logs:

```bash
sudo docker logs --tail 100 kwonrec-production-api-1
sudo docker logs --tail 100 kwonrec-production-worker-1
```

Successful configuration/digest lives root-only in /opt/kwonrec/current; prior
releases in /opt/kwonrec/history. Failed application deployment attempts restore the
last successful automated API/worker release when available. First-time adoption
cannot guarantee rollback to an unknown manual image/config. SQL/Redis writes are
not undone by application rollback. This is a single VM with a short replacement
window, not zero downtime. Registry credentials/staging files are cleaned up.

No production deployment occurs until you push main or dispatch the workflow.
