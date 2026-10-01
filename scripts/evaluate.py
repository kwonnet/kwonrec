"""Evaluate a temporally split JSONL replay in a disposable namespace.

Train lines: {"kind":"post"|"event", "data": <API payload>}
Holdout lines: {"user_id":"...", "relevant_ids":["..."]}
Use one holdout per user, whose positives occur after all their training events.
This evaluator accepts recent (<=7 days) data; timestamps must not be fabricated.
"""
import argparse
import json
import math
import uuid

from src.runtime.engine import Engine
from src.runtime.schemas import Post, Interaction, FeedRequest
from src.runtime.settings import settings


def lines(path):
    with open(path) as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def ranking_metrics(predicted, relevant, k):
    relevant = set(relevant)
    hits = [int(item in relevant) for item in predicted[:k]]
    recall = sum(hits) / max(1, len(relevant))
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
    return recall, dcg / ideal if ideal else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train")
    parser.add_argument("holdout")
    parser.add_argument("--k", type=int, default=20, choices=range(1, 101))
    args = parser.parse_args()
    config = settings().model_copy(update={"namespace": "kwonrec:evaluation:" + uuid.uuid4().hex, "exploration_rate": 0})
    engine = Engine(config)
    # TTLs remove evaluation data automatically. Never FLUSHDB a shared instance.
    users, recall, ndcg, recommended = set(), 0, 0, set()
    for row in lines(args.train):
        if row["kind"] == "post":
            engine.upsert_post(Post.model_validate(row["data"]))
        elif row["kind"] == "event":
            engine.interact(Interaction.model_validate(row["data"]))
        else:
            raise ValueError("Unknown training record kind")
    for row in lines(args.holdout):
        if row["user_id"] in users or not row["relevant_ids"]:
            raise ValueError("Use one nonempty holdout per user")
        users.add(row["user_id"])
        result = engine.recommend(FeedRequest(user_id=row["user_id"], limit=args.k))
        ids = [item["id"] for item in result["recommendations"]]
        r, n = ranking_metrics(ids, row["relevant_ids"], args.k)
        recall += r
        ndcg += n
        recommended.update(ids)
    print(json.dumps({"users": len(users), f"recall@{args.k}": recall / max(1, len(users)),
                      f"ndcg@{args.k}": ndcg / max(1, len(users)), "unique_recommended_items": len(recommended),
                      "namespace": config.namespace}, indent=2))
    engine.redis.close()


if __name__ == "__main__":
    main()
