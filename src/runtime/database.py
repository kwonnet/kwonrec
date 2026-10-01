"""Connection options shared by the worker and setup command."""
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def postgres_url(value: str) -> str:
    # Prisma adds client-side options that libpq does not recognize. The schema
    # is retained separately as search_path; TLS and authentication options stay.
    parts = urlsplit(value)
    prisma_options = {"schema", "connection_limit", "pool_timeout", "pgbouncer"}
    query = [(k, v) for k, v in parse_qsl(parts.query) if k not in prisma_options]
    return urlunsplit(parts._replace(query=urlencode(query)))


def postgres_options(value: str) -> str:
    import re
    schema = dict(parse_qsl(urlsplit(value).query)).get("schema", "public")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", schema):
        raise ValueError("Database schema must be a simple PostgreSQL identifier")
    return f"-c statement_timeout=10000 -c lock_timeout=5000 -c search_path={schema}"
