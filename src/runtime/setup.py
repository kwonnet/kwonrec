"""Install transactional capture and backfill the existing catalog/history.

Run once before starting the connected worker; safe to rerun after interruption.
No application rows are deleted or rewritten.
"""
import argparse
import logging
from pathlib import Path
import psycopg
from psycopg.rows import dict_row
from .database import postgres_url, postgres_options
from .engine import Engine
from .settings import settings
from .worker import bootstrap, bootstrap_history, process_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bootstrap", action="store_true", help="Install triggers without repeating an existing backfill")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = settings()
    if not config.database_url:
        raise SystemExit("KWONREC_DATABASE_URL must point to kwonserver PostgreSQL")
    engine = Engine(config)
    try:
        engine.redis.ping()
        with psycopg.connect(postgres_url(config.database_url), autocommit=True,
                             row_factory=dict_row, connect_timeout=10,
                             options=postgres_options(config.database_url)) as connection:
            migration = Path(__file__).resolve().parents[2] / "sql" / "001_outbox.sql"
            logging.info("Installing recommendation outbox triggers")
            connection.execute(migration.read_text())
            if not args.skip_bootstrap:
                logging.info("Indexing recent posts")
                bootstrap(connection, engine)
                logging.info("Importing recent interactions")
                bootstrap_history(connection, engine)
            # Bound setup work; the continuous worker drains any remaining backlog.
            for _ in range(100):
                if not process_batch(connection, engine):
                    break
            counts = connection.execute("SELECT count(*) FILTER (WHERE processed_at IS NULL AND dead_at IS NULL) AS pending, count(*) FILTER (WHERE dead_at IS NOT NULL) AS dead FROM kwonrec.outbox").fetchone()
            logging.info("Recommendation setup complete: pending=%s dead=%s", counts["pending"], counts["dead"])
    finally:
        engine.redis.close()


if __name__ == "__main__":
    main()
