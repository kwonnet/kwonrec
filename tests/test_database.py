from urllib.parse import parse_qsl, urlsplit
import pytest
from src.runtime.database import postgres_url, postgres_options


def test_prisma_url_preserves_credentials_and_tls():
    url = 'postgresql://user:p%40ss@db/app?schema=public&sslmode=require&channel_binding=require&connection_limit=5&pgbouncer=true&pool_timeout=10'
    result = postgres_url(url)
    assert urlsplit(result).netloc == urlsplit(url).netloc
    assert dict(parse_qsl(urlsplit(result).query)) == {'sslmode': 'require', 'channel_binding': 'require'}
    assert 'search_path=public' in postgres_options(url)


def test_schema_is_preserved_and_options_cannot_be_injected():
    assert 'search_path=kwonnet' in postgres_options('postgresql://db/app?schema=kwonnet')
    with pytest.raises(ValueError):
        postgres_options('postgresql://db/app?schema=public%20-c%20role%3Dadmin')
