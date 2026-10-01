import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

from redis import Redis
from redis.cluster import RedisCluster

from . import scripts
from .features import cosine, digest, event_weight, features
from .schemas import FeedRequest, Interaction, Post
from .settings import Settings


class PostConflict(Exception):
    pass


class MissingPost(Exception):
    pass


class InvalidEvent(Exception):
    pass


class Engine:
    version = "online-itemcf-content-v2"

    def __init__(self, config: Settings, client=None):
        self.config = config
        client_type = RedisCluster if config.redis_cluster else Redis
        self.redis = client or client_type.from_url(
            config.redis_url, decode_responses=True, socket_timeout=2,
            socket_connect_timeout=2, max_connections=100,
            health_check_interval=30,
        )
        self.ttl = config.history_days * 86400
        self.catalog_ttl = config.catalog_days * 86400
        self.receipt_ttl = (config.event_max_age_days + 1) * 86400
        self._post = self.redis.register_script(scripts.POST)
        self._index = self.redis.register_script(scripts.INDEX)
        self._user_event = self.redis.register_script(scripts.USER_EVENT)
        self._item_event = self.redis.register_script(scripts.ITEM_EVENT)
        self._reserve = self.redis.register_script(scripts.RESERVE)

    def key(self, entity: str, identifier: str, part: str) -> str:
        return f"{self.config.namespace}:{{{entity}:{digest(identifier)}}}:{part}"

    def user_key(self, user: str, part: str):
        return self.key("user", user, part)

    def item_key(self, post: str, part: str):
        return self.key("item", post, part)

    def index_key(self, kind: str, identifier: str):
        return self.key("index", kind + ":" + identifier, "posts")

    def index(self, kind: str, identifier: str, post: str, score: float, *, temporal=True):
        if kind in {"recent", "popular"}:
            identifier += ":" + str(int(digest(post), 16) % self.config.catalog_shards)
        self._index(keys=[self.index_key(kind, identifier)], args=[
            score, post, time.time() - self.catalog_ttl if temporal else "-inf",
            self.config.index_limit, self.catalog_ttl if temporal else 172800,
        ])

    def upsert_post(self, post: Post):
        now = time.time()
        document = post.model_dump(mode="json")
        document["features"] = features(post.content, post.topics)
        document["eligible"] = post.eligible
        document["timestamp"] = post.created_at.timestamp()
        # Retain only derived features, not original post text, in the serving store.
        document.pop("content")
        applied = self._post(keys=[self.item_key(post.id, "post")], args=[
            post.version, json.dumps(document), self.catalog_ttl + self.receipt_ttl,
        ])
        if applied == -1:
            raise PostConflict("A different post payload already uses this version")
        if applied and post.eligible and document["timestamp"] >= now - self.catalog_ttl:
            for kind, identifier in [("recent", "all"), ("author", post.author_id)]:
                self.index(kind, identifier, post.id, document["timestamp"])
            for term in document["features"]:
                self.index("content", term, post.id, document["timestamp"])
        return bool(applied)

    def get_posts(self, ids):
        if not ids:
            return {}
        # GET pipelines work across cluster slots; MGET does not.
        pipe = self.redis.pipeline(transaction=False)
        for post in ids:
            pipe.get(self.item_key(post, "post"))
        return {pid: json.loads(raw) for pid, raw in zip(ids, pipe.execute()) if raw}

    def interact(self, event: Interaction):
        now = time.time()
        ts = event.occurred_at.timestamp()
        if ts > now + 300 or ts < now - self.config.event_max_age_days * 86400:
            raise InvalidEvent("Event must be within the last seven days and no more than five minutes ahead")
        post = self.get_posts([event.post_id]).get(event.post_id)
        if post is None:
            raise MissingPost(event.post_id)
        def u(part):
            return self.user_key(event.user_id, part)
        receipt_id = digest(event.event_id)
        raw = self._user_event(keys=[
            u("profile"), u("history"), u("seen"), u("hidden"),
            u("receipt:" + receipt_id), u("signal:" + digest(event.post_id + ":" + event.type)), u("updated"),
        ], args=[now, ts, event_weight(event.type, event.duration_seconds), event.post_id,
                 self.ttl, json.dumps(post["features"]), self.receipt_ttl, digest(event.model_dump_json())])
        result = json.loads(raw)
        if result["fingerprint"] != digest(event.model_dump_json()):
            raise InvalidEvent("Event ID was already used with a different payload")
        history = [p for p in result["history"] if p != event.post_id] if result["edge"] else []
        weight = result["weight"]
        # Each stage has its own receipt. A crash between shards is safe to retry.
        if weight != 0:
            score = self.update_item(event.post_id, event, history, weight, now)
            if weight > 0:
                self.index("popular", str(int(ts // 3600)), event.post_id, score, temporal=False)
            for neighbor in history:
                self.update_item(neighbor, event, [event.post_id], 0, now)
        return {"event_id": event.event_id, "status": "applied"}

    def update_item(self, post_id, event, neighbors, weight, now):
        return float(self._item_event(keys=[
            self.item_key(post_id, "stats"), self.item_key(post_id, "neighbors"),
            self.item_key(post_id, "receipt:" + digest(event.user_id + ":" + event.event_id)),
        ], args=[now, weight, json.dumps(neighbors), self.catalog_ttl, self.receipt_ttl]))

    def recommend(self, request: FeedRequest):
        now = time.time()
        user = request.user_id
        pipe = self.redis.pipeline(transaction=False)
        pipe.hgetall(self.user_key(user, "profile"))
        pipe.zrevrange(self.user_key(user, "history"), 0, 9)
        pipe.zrangebyscore(self.user_key(user, "seen"), now - self.ttl, "+inf")
        pipe.zrangebyscore(self.user_key(user, "hidden"), now - self.ttl, "+inf")
        raw_profile, history, seen, hidden = pipe.execute()
        profile = {k: float(v) for k, v in raw_profile.items()}
        for k, v in features("", request.interests).items():
            profile[k] = profile.get(k, 0) + v
        terms = sorted((k for k in profile if profile[k] > 0), key=lambda k: profile[k], reverse=True)[:8]
        queries = [(self.index_key("recent", f"all:{shard}"), "fresh", 50)
                   for shard in range(self.config.catalog_shards)]
        queries += [(self.index_key("popular", f"{int(now // 3600) - hour}:{shard}"), "popular", 20)
                    for hour in range(6) for shard in range(self.config.catalog_shards)]
        queries += [(self.index_key("content", term), "content", 60) for term in terms]
        queries += [(self.item_key(post, "neighbors"), "collaborative", 50) for post in history]
        queries += [(self.index_key("author", author), "following", 20) for author in request.following_author_ids[:20]]
        pipe = self.redis.pipeline(transaction=False)
        for key, _, size in queries:
            pipe.zrevrange(key, 0, size - 1, withscores=True)
        results = pipe.execute()
        sources = defaultdict(set)
        collaborative = defaultdict(float)
        # Round-robin retrieval keeps large sources from starving personalized candidates.
        ids = []
        excluded = set(seen) | set(hidden) | set(request.exclude_ids)
        for offset in range(max(q[2] for q in queries)):
            for (_, source, _), rows in zip(queries, results):
                if offset >= len(rows):
                    continue
                post, score = rows[offset]
                if post in excluded:
                    continue
                if post not in sources:
                    if len(ids) >= self.config.candidate_limit:
                        continue
                    ids.append(post)
                sources[post].add(source)
                if source == "collaborative":
                    collaborative[post] += math.log1p(max(0, score))
        posts = self.get_posts(ids)
        blocked = set(request.blocked_author_ids)
        following = set(request.following_author_ids)
        pipe = self.redis.pipeline(transaction=False)
        for post in ids:
            pipe.hgetall(self.item_key(post, "stats"))
        stats = dict(zip(ids, pipe.execute()))
        ranked = []
        for pid in ids:
            post = posts.get(pid)
            if not post or not post["eligible"] or post["author_id"] in blocked or post["author_id"] == user:
                continue
            age = max(0, now - post["timestamp"])
            if age > self.catalog_ttl:
                continue
            cf = min(1.0, collaborative[pid] / 5)
            content = cosine(profile, post["features"])
            fresh = math.exp(-age / 172800)
            values = stats.get(pid) or {}
            pop = float(values.get("score", 0)) * math.exp(-max(0, now-float(values.get("updated", now))) / 86400)
            popularity = min(1, math.log1p(pop) / 8)
            social = float(post["author_id"] in following)
            score = .35*cf + .35*content + .15*fresh + .1*popularity + .05*social
            ranked.append({"id": pid, "score": round(score, 6), "sources": sorted(sources[pid]),
                           "author": post["author_id"]})
        ranked.sort(key=lambda item: (-item["score"], item["id"]))
        # Exploration uses a request seed and is reproducible in offline evaluation.
        rng = random.Random(digest(user + ":" + str(request.seed)))
        ordered, counts = [], Counter()
        pool = list(ranked)
        while pool:
            explore = rng.random() < self.config.exploration_rate
            index = rng.randrange(len(pool)) if explore else 0
            item = pool.pop(index)
            if counts[item["author"]] >= self.config.author_cap:
                continue
            counts[item["author"]] += 1
            item["exploration"] = explore
            ordered.append(item)
        selected = self._reserve(keys=[self.user_key(user, "seen"), self.user_key(user, "hidden")],
                                 args=[json.dumps([item["id"] for item in ordered]), now, request.limit, self.ttl])
        chosen = set(selected)
        recommendations = [{k: v for k, v in item.items() if k != "author"}
                           for item in ordered if item["id"] in chosen]
        return {"user_id": user, "recommendations": recommendations, "model_version": self.version,
                "candidate_count": len(posts), "exhausted": len(recommendations) < request.limit,
                "generated_at": datetime.now(timezone.utc).isoformat()}
