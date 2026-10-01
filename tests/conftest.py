import os
import uuid
from datetime import datetime, timezone

import pytest
from redis import Redis

from src.runtime.engine import Engine
from src.runtime.schemas import Post, Interaction
from src.runtime.settings import Settings


@pytest.fixture
def engine():
    url = os.getenv("TEST_REDIS_URL", "redis://127.0.0.1:16379/0")
    client = Redis.from_url(url, decode_responses=True)
    client.ping()  # Infrastructure failure must fail, not silently skip all tests.
    config = Settings(allow_unauthenticated=True, redis_url=url,
                      namespace="test:" + uuid.uuid4().hex, exploration_rate=0, catalog_shards=2)
    service = Engine(config, client)
    yield service
    for key in client.scan_iter(config.namespace + ":*"):
        client.delete(key)
    client.close()


@pytest.fixture
def post_factory(engine):
    def create(id, content="", author=None, **kwargs):
        post = Post(id=id, author_id=author or "author-" + id,
                    content=content, created_at=kwargs.pop("created_at", datetime.now(timezone.utc)),
                    version=kwargs.pop("version", 1), **kwargs)
        engine.upsert_post(post)
        return post
    return create


@pytest.fixture
def event_factory():
    def make(user, post, kind="like", **kwargs):
        return Interaction(event_id=kwargs.pop("event_id", uuid.uuid4().hex), user_id=user,
                           post_id=post, type=kind,
                           occurred_at=kwargs.pop("occurred_at", datetime.now(timezone.utc)), **kwargs)
    return make
