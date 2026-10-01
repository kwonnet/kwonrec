import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

from src.main import app
from src.runtime.settings import settings


@pytest.fixture
def client(engine, monkeypatch):
    monkeypatch.setenv("KWONREC_REDIS_URL", engine.config.redis_url)
    monkeypatch.setenv("KWONREC_NAMESPACE", engine.config.namespace)
    monkeypatch.setenv("KWONREC_API_KEY", "test-service-key-with-at-least-32-characters")
    settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    settings.cache_clear()


def auth():
    return {"Authorization": "Bearer test-service-key-with-at-least-32-characters"}


def test_auth_health_ready_and_bounds(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/recommend/u").status_code == 401
    assert client.get("/metrics").status_code == 401
    assert client.get("/recommend/u?limit=101", headers=auth()).status_code == 422
    assert client.get("/recommend/u?limit=1", headers=auth()).json()["recommendations"] == []
    assert client.get("/metrics", headers=auth()).status_code == 200


def test_posts_events_feed_and_conflict(client, post_factory, event_factory):
    post = post_factory("seed", "football league")
    assert client.put("/v1/posts/other", json=post.model_dump(mode="json"), headers=auth()).status_code == 422
    assert client.put("/v1/posts/seed", json=post.model_dump(mode="json"), headers=auth()).status_code == 200
    event = event_factory("reader", "seed").model_dump(mode="json")
    missing = {**event, "event_id": "missing", "post_id": "unknown"}
    result = client.post("/v1/events", json={"events": [event, missing]}, headers=auth()).json()
    assert [row["status"] for row in result["results"]] == ["applied", "missing_post"]
    result = client.get("/recommend/reader", headers=auth())
    assert result.headers["cache-control"] == "no-store"
    assert result.json()["recommendations"] == []


def test_body_limit_and_deterministic_nlp(client):
    response = client.post("/keywords", content=b"x" * (1048576 + 1), headers=auth())
    assert response.status_code == 413
    data = {"text": "football football stadium"}
    assert client.post("/keywords", json=data, headers=auth()).json() == client.post("/keywords", json=data, headers=auth()).json()
    result = client.post("/classify", json={"text": "astronomy", "labels": ["music", "sports"]}, headers=auth()).json()
    assert result["score"] == 0 and result["matched"] is False


def test_redis_outage_is_explicit(client, monkeypatch):
    def unavailable(*args, **kwargs):
        raise ConnectionError("secret connection details")
    monkeypatch.setattr(app.state.engine.redis, "ping", unavailable)
    response = client.get("/ready")
    assert response.status_code == 503
    assert "secret" not in response.text
