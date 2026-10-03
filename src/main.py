"""KwonRec API. Run workers separately: python -m src.runtime.worker."""
import hmac
import logging
import time
from contextlib import asynccontextmanager
from collections import Counter
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Path, Query, Request
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter as MetricCounter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from redis.exceptions import RedisError

from .runtime.engine import Engine, InvalidEvent, MissingPost, PostConflict
from .runtime.features import tokens
from .runtime.schemas import ClassificationRequest, EventBatch, FeedRequest, Identifier, Post, TextRequest
from .runtime.settings import settings

logger = logging.getLogger("kwonrec")
REQUESTS = MetricCounter("kwonrec_requests_total", "Requests", ["route", "status"])
LATENCY = Histogram("kwonrec_request_seconds", "Request duration", ["route"])
CANDIDATES = Histogram("kwonrec_candidates", "Candidates fetched", buckets=(0, 50, 100, 250, 500, 1000, 2000))
EVENTS = MetricCounter("kwonrec_events_total", "Interaction attempts", ["result"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = Engine(settings())
    engine.redis.ping()
    app.state.engine = engine
    yield
    engine.redis.close()


app = FastAPI(title="KwonRec", version="2.0.0", lifespan=lifespan)


class BodyLimit:
    """Bound streamed and Content-Length bodies before JSON parsing."""
    def __init__(self, app, max_bytes=1048576):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > self.max_bytes:
                return await JSONResponse({"detail": "Request body exceeds 1 MiB"}, status_code=413)(scope, receive, send)
            chunks.append(message)
            if not message.get("more_body", False):
                break
        async def replay():
            if chunks:
                return chunks.pop(0)
            return await receive()
        await self.app(scope, replay, send)


app.add_middleware(BodyLimit)


@app.middleware("http")
async def instrument(request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    route = request.scope.get("route")
    label = route.path if route else "unmatched"
    REQUESTS.labels(label, str(response.status_code)).inc()
    LATENCY.labels(label).observe(time.monotonic() - start)
    response.headers["Cache-Control"] = "no-store"
    return response


def engine(request: Request) -> Engine:
    return request.app.state.engine


def authorize(authorization: Annotated[str | None, Header()] = None):
    config = settings()
    if config.allow_unauthenticated and not config.api_key:
        return
    expected = "Bearer " + config.api_key
    if not authorization or not hmac.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(401, "Invalid service credentials", headers={"WWW-Authenticate": "Bearer"})


protected = [Depends(authorize)]


@app.exception_handler(PostConflict)
async def post_conflict(request, error):
    return JSONResponse({"detail": str(error)}, status_code=409)


@app.exception_handler(RedisError)
async def unavailable(request, error):
    logger.error("Redis unavailable (%s)", type(error).__name__)
    return JSONResponse({"detail": "Recommendation store unavailable"}, status_code=503,
                        headers={"Retry-After": "1"})


@app.get("/health")
def health():
    return {"status": "healthy", "service": "kwonrec", "version": "2.0.0"}


@app.get("/ready")
def ready(service: Engine = Depends(engine)):
    service.redis.ping()
    return {"status": "ready"}


@app.get("/metrics", dependencies=protected)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.put("/v1/posts/{post_id}", dependencies=protected)
def upsert(post_id: Identifier, post: Post, service: Engine = Depends(engine)):
    if post_id != post.id:
        raise HTTPException(422, "Path and body post IDs must match")
    return {"applied": service.upsert_post(post)}


@app.post("/v1/events", dependencies=protected)
def events(batch: EventBatch, service: Engine = Depends(engine)):
    results = []
    for event in batch.events:
        try:
            results.append(service.interact(event))
            EVENTS.labels("applied").inc()
        except MissingPost:
            results.append({"event_id": event.event_id, "status": "missing_post", "retryable": True})
            EVENTS.labels("missing_post").inc()
        except InvalidEvent as error:
            results.append({"event_id": event.event_id, "status": "rejected", "reason": str(error), "retryable": False})
            EVENTS.labels("rejected").inc()
    # Partial success is explicit; replaying a batch uses the same event IDs.
    return {"results": results}


@app.post("/v1/recommendations", dependencies=protected)
def feed(request: FeedRequest, service: Engine = Depends(engine)):
    result = service.recommend(request)
    CANDIDATES.observe(result["candidate_count"])
    return result


@app.get("/recommend/{user_id}", dependencies=protected)
def recommend_feed(user_id: Annotated[str, Path(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")],
                limit: Annotated[int, Query(ge=1, le=100)] = 20,
                service: Engine = Depends(engine)):
    return feed(FeedRequest(user_id=user_id, limit=limit), service)


@app.post("/keywords", dependencies=protected)
def keywords(request: TextRequest):
    counts = Counter(tokens(request.text))
    maximum = max(counts.values(), default=1)
    return {"keywords": [{"phrase": term, "score": round(count / maximum, 6)}
                         for term, count in counts.most_common(20)], "method": "lexical-v1"}


@app.post("/classify", dependencies=protected)
def classify(request: ClassificationRequest):
    # Deterministic compatibility baseline, explicitly not zero-shot semantic inference.
    words = set(tokens(request.text))
    scores = [(len(words & set(tokens(label))) / max(1, len(set(tokens(label)))), label)
              for label in request.labels]
    score, label = max(scores, key=lambda item: item[0])
    return {"label": label, "score": score, "method": "lexical-v1", "matched": score > 0}