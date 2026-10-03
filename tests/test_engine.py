import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from redis.exceptions import ConnectionError
from redis.cluster import key_slot

from src.runtime.engine import InvalidEvent, MissingPost
from src.runtime.features import cosine, features
from src.runtime.schemas import FeedRequest, Post
from src.runtime.settings import Settings


def ids(response):
    return [item["id"] for item in response["recommendations"]]


def test_empty_catalog_terminates(engine):
    result = engine.recommend(FeedRequest(user_id="new"))
    assert result["exhausted"] and result["recommendations"] == []


def test_cold_start_visibility_freshness_and_author_diversity(engine, post_factory):
    for index in range(8):
        post_factory(f"public{index}", author="same")
    post_factory("draft", status="DRAFT")
    post_factory("private", scope="FOLLOWED")
    post_factory("hidden", hidden=True)
    post_factory("deleted", deleted=True)
    post_factory("reply", kind="REPLY")
    post_factory("old", created_at=datetime.now(timezone.utc) - timedelta(days=31))
    post_factory("own", author="new")
    result = ids(engine.recommend(FeedRequest(user_id="new")))
    assert len(result) == 3
    assert all(pid.startswith("public") for pid in result)


def test_content_personalization_is_immediate(engine, post_factory, event_factory):
    post_factory("seed", "football stadium tournament league")
    post_factory("football", "football stadium tournament league")
    post_factory("cooking", "recipe cooking soup kitchen")
    engine.interact(event_factory("reader", "seed"))
    result = engine.recommend(FeedRequest(user_id="reader"))
    assert ids(result)[0] == "football"
    assert "seed" not in ids(result)
    assert "content" in result["recommendations"][0]["sources"]


def test_collaborative_discovers_content_unrelated_item(engine, post_factory, event_factory):
    post_factory("seed", "football stadium")
    post_factory("neighbor", "galaxy telescope")
    for index in range(5):
        engine.interact(event_factory(f"peer{index}", "seed"))
        engine.interact(event_factory(f"peer{index}", "neighbor"))
    engine.interact(event_factory("reader", "seed"))
    result = engine.recommend(FeedRequest(user_id="reader"))
    assert "neighbor" in ids(result)
    assert "collaborative" in result["recommendations"][0]["sources"]
    assert engine.redis.zscore(engine.item_key("seed", "neighbors"), "neighbor") == 5


def test_negative_feedback_dominates_later_positive(engine, post_factory, event_factory):
    post_factory("p", "space telescope")
    engine.interact(event_factory("reader", "p", "report"))
    engine.interact(event_factory("reader", "p", "like"))
    assert ids(engine.recommend(FeedRequest(user_id="reader"))) == []
    assert engine.redis.zcard(engine.user_key("reader", "history")) == 0
    assert all(float(v) < 0 for v in engine.redis.hgetall(engine.user_key("reader", "profile")).values())


def test_impression_is_not_positive_label(engine, post_factory, event_factory):
    post_factory("p", "space telescope")
    engine.interact(event_factory("reader", "p", "impression"))
    assert not engine.redis.hgetall(engine.user_key("reader", "profile"))
    assert not engine.redis.zrange(engine.user_key("reader", "history"), 0, -1)
    assert not ids(engine.recommend(FeedRequest(user_id="reader")))


def test_retries_and_repeated_signal_do_not_inflate(engine, post_factory, event_factory):
    post_factory("p", "space telescope")
    event = event_factory("reader", "p")
    engine.interact(event)
    profile = engine.redis.hgetall(engine.user_key("reader", "profile"))
    for _ in range(3):
        engine.interact(event)
    engine.interact(event_factory("reader", "p"))
    assert profile == engine.redis.hgetall(engine.user_key("reader", "profile"))
    assert float(engine.redis.hget(engine.item_key("p", "stats"), "score")) == 3


def test_event_id_conflict_is_rejected(engine, post_factory, event_factory):
    post_factory("p")
    event = event_factory("reader", "p")
    engine.interact(event)
    with pytest.raises(InvalidEvent, match="different payload"):
        engine.interact(event.model_copy(update={"type": "report"}))


def test_retry_after_partial_shard_failure(engine, post_factory, event_factory, monkeypatch):
    post_factory("a", "astronomy")
    post_factory("b", "football")
    engine.interact(event_factory("reader", "a"))
    event = event_factory("reader", "b")
    original = engine.update_item

    def fail_reciprocal(post_id, *args):
        if post_id == "a":
            raise ConnectionError("simulated unavailable shard")
        return original(post_id, *args)

    monkeypatch.setattr(engine, "update_item", fail_reciprocal)
    with pytest.raises(ConnectionError):
        engine.interact(event)
    monkeypatch.setattr(engine, "update_item", original)
    engine.interact(event)
    assert engine.redis.zscore(engine.item_key("b", "neighbors"), "a") == 1
    assert engine.redis.zscore(engine.item_key("a", "neighbors"), "b") == 1
    assert float(engine.redis.hget(engine.item_key("b", "stats"), "score")) == 3


def test_concurrent_feed_requests_do_not_repeat(engine, post_factory):
    for index in range(50):
        post_factory(f"post{index}")
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: engine.recommend(FeedRequest(user_id="reader", limit=5)), range(8)))
    served = [pid for result in results for pid in ids(result)]
    assert len(served) == 40
    assert len(set(served)) == 40
    rest = ids(engine.recommend(FeedRequest(user_id="reader", limit=100)))
    assert not set(rest) & set(served)
    assert len(rest) == 10


def test_deleted_or_out_of_order_posts_cannot_resurrect(engine, post_factory):
    original = post_factory("p", "space telescope")
    engine.upsert_post(original.model_copy(update={"version": 3, "deleted": True}))
    assert not engine.upsert_post(original.model_copy(update={"version": 2}))
    assert not ids(engine.recommend(FeedRequest(user_id="reader")))


def test_filters_and_onboarding(engine, post_factory):
    post_factory("a", topics=["music"])
    post_factory("b", topics=["music"], author="blocked")
    post_factory("c", topics=["music"])
    post_factory("d", topics=["science"])
    result = ids(engine.recommend(FeedRequest(user_id="reader", interests=["music"],
                                             blocked_author_ids=["blocked"], exclude_ids=["c"])))
    assert result == ["a", "d"]


def test_missing_and_invalid_events(engine, post_factory, event_factory):
    with pytest.raises(MissingPost):
        engine.interact(event_factory("r", "missing"))
    post_factory("p")
    for delta in [timedelta(days=-8), timedelta(hours=1)]:
        with pytest.raises(InvalidEvent):
            engine.interact(event_factory("r", "p", occurred_at=datetime.now(timezone.utc) + delta))


def test_api_contract_constraints():
    for limit in [0, 101]:
        with pytest.raises(ValidationError):
            FeedRequest(user_id="reader", limit=limit)
    with pytest.raises(ValidationError):
        FeedRequest(user_id="user{injected-slot}")
    with pytest.raises(ValidationError):
        Post(id="p", author_id="a", created_at=datetime.now(), version=1)
    with pytest.raises(ValidationError):
        Settings(api_key="short")


def test_feature_extraction_deterministic_and_bounded():
    value = features("<p>Football football &amp; stadium</p>", ["Sports"])
    assert value == features("<p>Football football &amp; stadium</p>", ["Sports"])
    assert abs(cosine(value, value) - 1) < 0.00001
    assert len(features(" ".join(f"word{i}" for i in range(500)), [])) <= 28


def test_hash_slots_are_entity_local(engine):
    assert key_slot(engine.user_key("u", "seen").encode()) == key_slot(engine.user_key("u", "profile").encode())
    assert key_slot(engine.item_key("p", "stats").encode()) == key_slot(engine.item_key("p", "neighbors").encode())
    assert engine.user_key("u", "seen") != engine.user_key("v", "seen")


def test_catalog_does_not_store_raw_text(engine, post_factory):
    post_factory("p", "original raw content")
    assert "content" not in json.loads(engine.redis.get(engine.item_key("p", "post")))


def test_offline_ranking_metrics():
    from scripts.evaluate import ranking_metrics
    recall, ndcg = ranking_metrics(["a", "x", "b"], ["a", "b"], 3)
    assert recall == 1 and 0 < ndcg < 1
    assert ranking_metrics(["a", "b"], ["a", "b"], 2) == (1, 1)
    assert ranking_metrics([], ["a"], 10) == (0, 0)


def test_equal_version_conflicting_post_is_rejected(engine, post_factory):
    from src.runtime.engine import PostConflict
    post = post_factory("p", "original content")
    assert engine.upsert_post(post)
    with pytest.raises(PostConflict):
        engine.upsert_post(post.model_copy(update={"deleted": True}))
    assert engine.get_posts(["p"])["p"]["deleted"] is False


def test_expired_seen_entries_do_not_block_next_request(engine, post_factory):
    import time
    post_factory("p")
    engine.redis.zadd(engine.user_key("reader", "seen"), {"p": time.time() - engine.ttl - 1})
    assert ids(engine.recommend(FeedRequest(user_id="reader"))) == ["p"]


def test_redis_connection_pool_is_bounded_and_configurable(monkeypatch):
    from src.runtime.engine import Engine
    from unittest.mock import MagicMock
    client = MagicMock()
    factory = MagicMock(return_value=client)
    monkeypatch.setattr('src.runtime.engine.Redis.from_url', factory)
    Engine(Settings(allow_unauthenticated=True, redis_max_connections=4))
    assert factory.call_args.kwargs['max_connections'] == 4
    with pytest.raises(ValidationError):
        Settings(allow_unauthenticated=True, redis_max_connections=0)
