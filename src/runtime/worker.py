"""At-least-once PostgreSQL outbox delivery; Redis stages are individually idempotent.

Multiple worker processes use SKIP LOCKED. No Redis access occurs inside a source
application transaction: source writes only append small transactional outbox rows.
"""
import argparse
import logging
import json
import signal
import time
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row
from psycopg import sql

from .database import postgres_url, postgres_options
from .engine import Engine, InvalidEvent
from .schemas import Interaction, Post
from .settings import settings

log = logging.getLogger("kwonrec.worker")


def sync_post(connection, engine, post_id, fallback=None):
    # Serialize re-reading and materializing one post across bootstrap/live workers.
    # A current row snapshot, rather than stale event payload, prevents resurrection.
    connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", ("kwonrec:" + post_id,))
    row = connection.execute('SELECT * FROM "Post" WHERE id = %s', (post_id,)).fetchone()
    # Monotonic wall-clock microseconds generated under the same per-post lock.
    version = int(connection.execute("SELECT (extract(epoch FROM clock_timestamp()) * 1000000)::bigint AS v").fetchone()["v"])
    if row:
        created = row["createdAt"].replace(tzinfo=timezone.utc) if row["createdAt"].tzinfo is None else row["createdAt"]
        post = Post(id=row["id"], author_id=row["userId"], content=row.get("content") or "",
                    topics=[row["topic"]] if row.get("topic") else [], created_at=created,
                    version=version, status=row["status"], scope=row["scope"], kind=row["kind"],
                    hidden=row.get("isHidden", False), deleted=row.get("deletedAt") is not None)
    else:
        fallback = fallback or {}
        post = Post(id=post_id, author_id=fallback.get("author_id") or "deleted",
                    created_at=datetime.now(timezone.utc), version=version, deleted=True)
    engine.upsert_post(post)


def process_batch(connection, engine):
    processed = 0
    config = engine.config
    # One row per transaction avoids retaining a batch of source DB locks during I/O.
    for _ in range(config.worker_batch):
        with connection.transaction():
            row = connection.execute("""
                SELECT * FROM kwonrec.outbox
                WHERE processed_at IS NULL AND dead_at IS NULL AND available_at <= clock_timestamp()
                ORDER BY available_at, id FOR UPDATE SKIP LOCKED LIMIT 1
            """).fetchone()
            if not row:
                break
            try:
                # Isolate database errors in a savepoint so retry bookkeeping can commit.
                with connection.transaction():
                    if row["kind"] == "post" or not engine.get_posts([row["entity_id"]]):
                        sync_post(connection, engine, row["entity_id"], row["payload"])
                    if row["kind"] == "interaction":
                        payload = dict(row["payload"])
                        if isinstance(payload.get("occurred_at"), str):
                            occurred = datetime.fromisoformat(payload["occurred_at"].replace("Z", "+00:00"))
                            if occurred.tzinfo is None:
                                occurred = occurred.replace(tzinfo=timezone.utc)
                            payload["occurred_at"] = occurred
                        event = Interaction(event_id=f"outbox:{row['id']}", post_id=row["entity_id"], **payload)
                        engine.interact(event)
                connection.execute("UPDATE kwonrec.outbox SET processed_at=clock_timestamp(), error=NULL WHERE id=%s", (row["id"],))
                processed += 1
            except Exception as error:
                attempts = row["attempts"] + 1
                terminal = attempts >= config.max_attempts or isinstance(error, InvalidEvent)
                # Never put DB connection strings, post bodies or credentials into logs.
                log.warning("outbox id=%s failure=%s attempt=%s terminal=%s", row["id"], type(error).__name__, attempts, terminal)
                connection.execute("""UPDATE kwonrec.outbox SET attempts=%s, error=%s,
                    available_at=clock_timestamp() + (%s * interval '1 second'),
                    dead_at=CASE WHEN %s THEN clock_timestamp() ELSE NULL END WHERE id=%s""",
                    (attempts, type(error).__name__, min(300, 2 ** min(attempts, 9)), terminal, row["id"]))
    return processed


def bootstrap(connection, engine):
    """Bounded keyset catalog backfill; run after installing triggers."""
    last = ""
    while True:
        rows = connection.execute('''SELECT id FROM "Post" WHERE id > %s
            AND "createdAt" >= (CURRENT_TIMESTAMP - (%s * interval '1 day'))
            ORDER BY id LIMIT %s''', (last, engine.config.catalog_days, engine.config.worker_batch)).fetchall()
        if not rows:
            return
        for row in rows:
            with connection.transaction():
                sync_post(connection, engine, row["id"])
        last = rows[-1]["id"]
        log.info("catalog bootstrap batch=%s", len(rows))


SOURCES = {
    "LikedPost": "like", "Bookmark": "bookmark", "PostClick": "click", "PostView": "view",
    "PostImpression": "impression", "PostShare": "share", "PostTip": "tip",
    "PostDisinterest": "dislike", "PostReport": "report", "Post": None,
}


def bootstrap_history(connection, engine):
    """Enqueue a bounded recent history replay, deduplicated against live triggers.

    Call before starting live consumers for a new namespace to preserve chronological
    replay as closely as possible. Late data is age-decayed, never treated as new.
    """
    for table, kind in SOURCES.items():
        last = ""
        while True:
            # PostClick has a `source` column. Explicit .* selects the whole row
            # rather than letting PostgreSQL resolve the alias as that column.
            rows = connection.execute(sql.SQL('''SELECT to_jsonb(source.*) AS data FROM {} AS source
                WHERE id > %s AND "createdAt" >= (CURRENT_TIMESTAMP - (%s * interval '1 day'))
                ORDER BY id LIMIT %s''').format(sql.Identifier(table)),
                (last, engine.config.event_max_age_days, engine.config.worker_batch)).fetchall()
            if not rows:
                break
            for row in rows:
                data = row["data"]
                event_kind = kind
                post_id = data.get("postId")
                if table == "Post":
                    if data.get("kind") not in {"REPLY", "QUOTE", "REPOST"}:
                        continue
                    event_kind = data["kind"].lower()
                    post_id = data.get("parentId")
                uid = data.get("userId") or data.get("senderId")
                if not uid or not post_id:
                    continue
                connection.execute('''INSERT INTO kwonrec.outbox(kind, entity_id, source_key, payload)
                    VALUES ('interaction', %s, %s, %s::jsonb) ON CONFLICT (source_key) DO NOTHING''',
                    (post_id, table + ":" + data["id"], json.dumps({
                        "user_id": uid, "type": event_kind, "occurred_at": data["createdAt"],
                        "duration_seconds": data.get("duration", 0),
                    })))
            last = rows[-1]["data"]["id"]
        log.info("history source=%s enqueued", table)



def replay_retained(connection, engine):
    """Replay retained rows into a fresh namespace without changing delivery cursors.

    Stop all live consumers first, bootstrap catalog/history, replay, then switch API
    and workers together. New/uncommitted source writes remain pending for workers.
    """
    high_water = connection.execute("SELECT coalesce(max(id),0) AS id FROM kwonrec.outbox").fetchone()["id"]
    last = 0
    while last < high_water:
        rows = connection.execute('''SELECT * FROM kwonrec.outbox WHERE id > %s AND id <= %s
            AND created_at >= CURRENT_TIMESTAMP - (%s * interval '1 day')
            ORDER BY id LIMIT %s''', (last, high_water, engine.config.event_max_age_days,
                                     engine.config.worker_batch)).fetchall()
        if not rows:
            break
        for row in rows:
            with connection.transaction():
                sync_post(connection, engine, row["entity_id"], row["payload"])
                if row["kind"] == "interaction":
                    payload = dict(row["payload"])
                    if isinstance(payload.get("occurred_at"), str):
                        occurred = datetime.fromisoformat(payload["occurred_at"].replace("Z", "+00:00"))
                        payload["occurred_at"] = occurred.replace(tzinfo=timezone.utc) if occurred.tzinfo is None else occurred
                    event = Interaction(event_id=f"outbox:{row['id']}", post_id=row["entity_id"], **payload)
                    try:
                        engine.interact(event)
                    except InvalidEvent:
                        log.warning("replay skipped expired/rejected interaction id=%s", row["id"])
            last = row["id"]
        log.info("replay batch=%s last_id=%s high_water=%s", len(rows), last, high_water)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true", help="Backfill recent catalog then exit")
    parser.add_argument("--bootstrap-history", action="store_true", help="Enqueue recent interactions then exit")
    parser.add_argument("--replay-retained", action="store_true", help="Rebuild from retained outbox without moving its delivery cursor")
    parser.add_argument("--once", action="store_true", help="Process one bounded batch then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = settings()
    if not config.database_url:
        raise SystemExit("KWONREC_DATABASE_URL is required for the outbox worker")
    engine = Engine(config)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not stopping:
            try:
                with psycopg.connect(postgres_url(config.database_url), autocommit=True, row_factory=dict_row,
                                     connect_timeout=5, options=postgres_options(config.database_url)) as connection:
                    if args.replay_retained:
                        replay_retained(connection, engine)
                        return
                    if args.bootstrap_history:
                        bootstrap_history(connection, engine)
                        return
                    if args.bootstrap:
                        bootstrap(connection, engine)
                        return
                    while not stopping:
                        count = process_batch(connection, engine)
                        if args.once:
                            return
                        if not count:
                            time.sleep(config.poll_seconds)
            except psycopg.Error as error:
                if args.once or args.bootstrap or args.bootstrap_history or args.replay_retained:
                    raise
                log.error("database unavailable: %s", type(error).__name__)
                time.sleep(2)
    finally:
        engine.redis.close()


if __name__ == "__main__":
    main()
