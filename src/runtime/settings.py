from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KWONREC_", env_file=".env", extra="ignore")
    redis_url: str = "redis://localhost:6379/0"
    redis_cluster: bool = False
    api_key: str = ""
    allow_unauthenticated: bool = False
    database_url: str = ""
    namespace: str = Field(default="kwonrec:v2", pattern=r"^[a-zA-Z0-9:_-]+$")
    catalog_days: int = Field(default=30, ge=1, le=90)
    history_days: int = Field(default=30, ge=1, le=90)
    event_max_age_days: int = Field(default=7, ge=1, le=7)
    candidate_limit: int = Field(default=1000, ge=100, le=2000)
    catalog_shards: int = Field(default=8, ge=1, le=32)
    index_limit: int = Field(default=5000, ge=100, le=20000)
    author_cap: int = Field(default=3, ge=1, le=20)
    exploration_rate: float = Field(default=0.1, ge=0, le=0.3)
    worker_batch: int = Field(default=100, ge=1, le=1000)
    poll_seconds: float = Field(default=0.25, ge=0.05, le=60)
    max_attempts: int = Field(default=10, ge=1, le=100)

    @model_validator(mode="after")
    def check_auth(self):
        if not self.api_key and not self.allow_unauthenticated:
            raise ValueError("Set KWONREC_API_KEY (or explicitly allow unauthenticated local development)")
        if self.api_key and len(self.api_key) < 32:
            raise ValueError("KWONREC_API_KEY must contain at least 32 characters")
        return self


@lru_cache
def settings():
    return Settings()
