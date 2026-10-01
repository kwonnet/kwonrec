"""Deterministic multilingual token features; no model downloads on the request path."""
import hashlib
import math
import re
from collections import Counter
from html import unescape
from pathlib import Path

STOPWORDS = set((Path(__file__).parents[1] / "stopwords.txt").read_text().lower().split())
WEIGHTS = {"impression": 0, "view": 0.4, "click": 1, "like": 3, "bookmark": 4,
           "share": 4, "reply": 3, "quote": 4, "repost": 4, "tip": 5,
           "hide": -5, "dislike": -5, "report": -8}


def tokens(text: str) -> list[str]:
    clean = unescape(re.sub(r"<[^>]*>", " ", text)).casefold()
    return [w for w in re.findall(r"[\w#]{2,64}", clean) if w not in STOPWORDS and not w.isdigit()]


def features(content: str, topics: list[str]) -> dict[str, float]:
    counts = Counter(tokens(content))
    result = {"w:" + term: 1 + math.log(count) for term, count in counts.most_common(28)}
    for topic in topics[:4]:
        result["t:" + topic.casefold().strip()[:100]] = 3.0
    norm = math.sqrt(sum(v * v for v in result.values())) or 1
    return {k: round(v / norm, 6) for k, v in result.items()}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:32]


def event_weight(kind: str, duration: float) -> float:
    if kind == "view":
        return min(duration / 30, 2) if duration >= 3 else 0
    return WEIGHTS[kind]


def cosine(profile: dict[str, float], item: dict[str, float]) -> float:
    norm = math.sqrt(sum(v * v for v in profile.values())) or 1
    return max(-1, min(1, sum(profile.get(k, 0) * v for k, v in item.items()) / norm))
