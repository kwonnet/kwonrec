"""Run inside the test Redis Cluster network; validates actual cross-shard delivery."""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from src.runtime.engine import Engine
from src.runtime.schemas import Post, Interaction, FeedRequest
from src.runtime.settings import Settings


def main():
    config = Settings(allow_unauthenticated=True, redis_cluster=True,
                      redis_url=os.environ["TEST_REDIS_CLUSTER_URL"],
                      namespace="kwonrec:cluster-test:" + uuid.uuid4().hex, exploration_rate=0)
    engine = Engine(config)
    try:
        for index in range(30):
            engine.upsert_post(Post(id=f"p{index}", author_id=f"a{index}", content="football stadium",
                                    created_at=datetime.now(timezone.utc), version=1))
        for index in range(2):
            event = Interaction(event_id=f"e{index}", user_id="reader", post_id=f"p{index}", type="like",
                                occurred_at=datetime.now(timezone.utc))
            engine.interact(event)
            engine.interact(event)
        assert engine.redis.zscore(engine.item_key("p0", "neighbors"), "p1") == 1
        assert engine.redis.zscore(engine.item_key("p1", "neighbors"), "p0") == 1
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(lambda _: engine.recommend(FeedRequest(user_id="reader", limit=5)), range(4)))
        ids = [item["id"] for response in responses for item in response["recommendations"]]
        assert len(ids) == len(set(ids)) == 20
        assert not {"p0", "p1"} & set(ids)
        print("Redis Cluster smoke passed: indexing, cross-shard retry deduplication, collaborative edges, concurrent feeds")
    finally:
        for key in engine.redis.scan_iter(config.namespace + ":*"):
            engine.redis.delete(key)
        engine.redis.close()


if __name__ == "__main__":
    main()
