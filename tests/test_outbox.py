import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import dict_row

from src.runtime.schemas import FeedRequest
from src.runtime.worker import bootstrap, process_batch

TABLES = ["LikedPost", "Bookmark", "PostClick", "PostView", "PostImpression", "PostShare", "PostTip", "PostDisinterest", "PostReport"]


@pytest.fixture
def database():
    # This fixture intentionally requires a disposable database named kwonrec_test.
    url = os.getenv("TEST_DATABASE_URL", "postgresql://postgres:kwonrec_test@127.0.0.1:15432/kwonrec_test")
    with psycopg.connect(url, autocommit=True, row_factory=dict_row) as connection:
        assert connection.info.dbname == "kwonrec_test", "Use a disposable kwonrec_test database"
        connection.execute('DROP SCHEMA IF EXISTS kwonrec CASCADE')
        connection.execute('DROP TABLE IF EXISTS "Post" CASCADE')
        connection.execute('''CREATE TABLE "Post" (
            id text PRIMARY KEY, "userId" text NOT NULL, content text, topic text,
            status text DEFAULT 'PUBLISHED', scope text DEFAULT 'ANYONE', kind text DEFAULT 'ROOT',
            "isHidden" boolean DEFAULT false, "deletedAt" timestamp, "parentId" text,
            "createdAt" timestamp DEFAULT CURRENT_TIMESTAMP, "updatedAt" timestamp DEFAULT CURRENT_TIMESTAMP,
            "totalLikes" bigint DEFAULT 0)''')
        for table in TABLES:
            connection.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE').format(sql.Identifier(table)))
            connection.execute(sql.SQL('''CREATE TABLE {} (
                id text PRIMARY KEY, "userId" text, "senderId" text, "postId" text,
                "createdAt" timestamp DEFAULT CURRENT_TIMESTAMP, duration integer DEFAULT 0)''').format(sql.Identifier(table)))
        migration = (Path(__file__).parents[1] / "sql" / "001_outbox.sql").read_text()
        connection.execute(migration)
        # The migration must be repeatable.
        connection.execute(migration)
        yield connection, url


def insert_post(connection, id="p"):
    connection.execute('INSERT INTO "Post" (id,"userId",content) VALUES (%s,%s,%s)',
                       (id, "author-" + id, "football stadium"))


def test_transactional_outbox_rollback_and_delivery(database, engine):
    connection, _ = database
    with pytest.raises(RuntimeError):
        with connection.transaction():
            insert_post(connection, "rolled-back")
            raise RuntimeError("rollback")
    assert connection.execute('SELECT count(*) AS n FROM kwonrec.outbox').fetchone()["n"] == 0
    insert_post(connection)
    connection.execute('INSERT INTO "LikedPost" (id,"userId","postId") VALUES (\'like1\',\'reader\',\'p\')')
    assert process_batch(connection, engine) == 2
    assert engine.redis.hgetall(engine.user_key("reader", "profile"))
    assert process_batch(connection, engine) == 0
    assert not engine.recommend(FeedRequest(user_id="reader"))["recommendations"]


def test_outbox_reads_current_state_not_stale_payload(database, engine):
    connection, _ = database
    insert_post(connection)
    connection.execute('UPDATE "Post" SET status=\'DELETED\' WHERE id=\'p\'')
    assert process_batch(connection, engine) == 2
    assert not engine.recommend(FeedRequest(user_id="reader"))["recommendations"]
    connection.execute('DELETE FROM "Post" WHERE id=\'p\'')
    assert process_batch(connection, engine) == 1
    assert engine.get_posts(["p"])["p"]["deleted"] is True


def test_counter_only_changes_do_not_enqueue_catalog(database, engine):
    connection, _ = database
    insert_post(connection)
    connection.execute('UPDATE "Post" SET "totalLikes"=2, "updatedAt"=CURRENT_TIMESTAMP WHERE id=\'p\'')
    assert connection.execute('SELECT count(*) AS n FROM kwonrec.outbox').fetchone()["n"] == 1


def test_source_timestamp_is_accepted_as_utc(database, engine):
    connection, _ = database
    insert_post(connection)
    connection.execute('INSERT INTO "PostTip" (id,"senderId","postId") VALUES (\'tip1\',\'reader\',\'p\')')
    process_batch(connection, engine)
    rows = connection.execute('SELECT * FROM kwonrec.outbox WHERE kind=\'interaction\'').fetchall()
    assert rows[0]["processed_at"] is not None


def test_poison_message_retries_and_dead_letter(database, engine):
    connection, _ = database
    insert_post(connection)
    connection.execute("INSERT INTO kwonrec.outbox(kind,entity_id,payload) VALUES ('interaction','p','{}')")
    engine.config.max_attempts = 1
    assert process_batch(connection, engine) == 1
    dead = connection.execute('SELECT * FROM kwonrec.outbox WHERE dead_at IS NOT NULL').fetchone()
    assert dead["attempts"] == 1 and dead["error"] == "ValidationError"


def test_bootstrap_and_parallel_consumers(database, engine):
    connection, url = database
    for index in range(20):
        insert_post(connection, f"post{index}")
    bootstrap(connection, engine)
    assert len(engine.get_posts([f"post{index}" for index in range(20)])) == 20

    def consume(_):
        with psycopg.connect(url, autocommit=True, row_factory=dict_row) as conn:
            return process_batch(conn, engine)
    with ThreadPoolExecutor(max_workers=3) as executor:
        assert sum(executor.map(consume, range(3))) == 20
    assert connection.execute('SELECT count(*) AS n FROM kwonrec.outbox WHERE processed_at IS NULL').fetchone()["n"] == 0


def test_history_backfill_deduplicates_live_capture(database, engine):
    from src.runtime.worker import bootstrap_history
    connection, _ = database
    insert_post(connection)
    connection.execute('INSERT INTO "LikedPost" (id,"userId","postId") VALUES (\'like1\',\'reader\',\'p\')')
    bootstrap_history(connection, engine)
    bootstrap_history(connection, engine)
    assert connection.execute("SELECT count(*) AS n FROM kwonrec.outbox WHERE kind='interaction'").fetchone()["n"] == 1
    assert process_batch(connection, engine) == 2


def test_replay_restores_consumed_rows_to_new_namespace(database, engine):
    from src.runtime.engine import Engine
    from src.runtime.worker import replay_retained
    connection, _ = database
    insert_post(connection)
    connection.execute('INSERT INTO "LikedPost" (id,"userId","postId") VALUES (\'like1\',\'reader\',\'p\')')
    assert process_batch(connection, engine) == 2
    fresh = Engine(engine.config.model_copy(update={"namespace": engine.config.namespace + ":rebuild"}), engine.redis)
    replay_retained(connection, fresh)
    assert fresh.redis.hgetall(fresh.user_key("reader", "profile"))
    assert connection.execute('SELECT count(*) AS n FROM kwonrec.outbox WHERE processed_at IS NULL').fetchone()["n"] == 0
