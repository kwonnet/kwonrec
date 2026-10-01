from datetime import datetime, timezone, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:@-]+$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Post(StrictModel):
    id: Identifier
    author_id: Identifier
    content: str = Field(default="", max_length=50000)
    topics: list[Annotated[str, Field(max_length=100)]] = Field(default_factory=list, max_length=20)
    created_at: datetime
    version: int = Field(ge=0, le=9007199254740991)
    status: Literal["PUBLISHED", "DRAFT", "SCHEDULED", "REPORTED", "DELETED"] = "PUBLISHED"
    scope: Literal["ANYONE", "VERIFIED", "FOLLOWED", "MENTIONS", "COUNTRY", "CONTINENT"] = "ANYONE"
    kind: Literal["ROOT", "REPOST", "QUOTE", "REPLY", "THREAD"] = "ROOT"
    hidden: bool = False
    deleted: bool = False

    @field_validator("created_at")
    @classmethod
    def aware_date(cls, value):
        if value.tzinfo is None:
            raise ValueError("created_at must include a timezone")
        if value > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ValueError("created_at is in the future")
        return value

    @property
    def eligible(self):
        return (self.status == "PUBLISHED" and self.scope == "ANYONE"
                and self.kind in {"ROOT", "REPOST", "QUOTE"} and not self.hidden and not self.deleted)


EventType = Literal["impression", "view", "click", "like", "bookmark", "share", "reply", "quote", "repost", "tip", "hide", "dislike", "report"]


class Interaction(StrictModel):
    event_id: Identifier
    user_id: Identifier
    post_id: Identifier
    type: EventType
    occurred_at: datetime
    duration_seconds: float = Field(default=0, ge=0, le=3600)

    @field_validator("occurred_at")
    @classmethod
    def aware_date(cls, value):
        if value.tzinfo is None:
            raise ValueError("occurred_at must include a timezone")
        return value


class EventBatch(StrictModel):
    events: list[Interaction] = Field(min_length=1, max_length=100)


class FeedRequest(StrictModel):
    user_id: Identifier
    limit: int = Field(default=20, ge=1, le=100)
    exclude_ids: list[Identifier] = Field(default_factory=list, max_length=500)
    # The caller is a trusted backend; these restrictions are additive.
    blocked_author_ids: list[Identifier] = Field(default_factory=list, max_length=500)
    following_author_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    interests: list[Annotated[str, Field(max_length=100)]] = Field(default_factory=list, max_length=20)
    seed: int = Field(default=0, ge=0, le=2147483647)


class TextRequest(StrictModel):
    text: str = Field(min_length=1, max_length=50000)


class ClassificationRequest(TextRequest):
    labels: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(min_length=1, max_length=100)
