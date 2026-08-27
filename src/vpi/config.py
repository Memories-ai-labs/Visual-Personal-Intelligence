"""Settings, resolved from the environment and `.env`."""

from __future__ import annotations

from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DATALAKE_BASE_URL = "https://api.memories.ai/datalake/v1"


class Settings(BaseSettings):
    """Everything the agent needs to run. Nothing here is a secret at rest."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- DataLake ---
    memories_api_key: str = Field("", alias="MEMORIES_API_KEY")
    datalake_base_url: str = Field(DATALAKE_BASE_URL, alias="VPI_DATALAKE_BASE_URL")
    collection_id: str = Field("", alias="VPI_COLLECTION_ID")

    # --- LLM ---
    llm_model: str = Field("claude-opus-5", alias="VPI_LLM_MODEL")
    llm_effort: str = Field("high", alias="VPI_LLM_EFFORT")
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    llm_base_url: str = Field("", alias="VPI_LLM_BASE_URL")
    llm_api_key: str = Field("", alias="VPI_LLM_API_KEY")

    # --- Agent ---
    timezone: str = Field("UTC", alias="VPI_TIMEZONE")
    max_steps: int = Field(12, alias="VPI_MAX_STEPS")
    demo: bool = Field(False, alias="VPI_DEMO")

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:  # pragma: no cover - env dependent
            raise ValueError(
                f"VPI_TIMEZONE={v!r} is not an IANA timezone (e.g. Asia/Shanghai)"
            ) from exc
        return v

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def uses_openai_compatible(self) -> bool:
        """A base URL means the user pointed us at an OpenAI-compatible endpoint."""
        return bool(self.llm_base_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
