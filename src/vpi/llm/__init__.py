"""LLM backends. Claude by default; any OpenAI-compatible endpoint as fallback."""

from __future__ import annotations

import os
from pathlib import Path

from vpi.config import Settings, get_settings
from vpi.llm.base import (
    LLMBackend,
    LLMResponse,
    ToolCall,
    ToolResult,
    ToolSpec,
    Turn,
    Usage,
)


class MissingLLMCredentials(RuntimeError):
    """No way to reach a model. Said plainly, because it is most people's first error."""


# The SDK also resolves an `ant auth login` profile from disk, so a missing env
# var is not on its own proof that there are no credentials.
_ANTHROPIC_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
_ANTHROPIC_PROFILE_DIR = Path.home() / ".config" / "anthropic"


def _has_anthropic_credentials(settings: Settings) -> bool:
    if settings.anthropic_api_key:
        return True
    if any(os.environ.get(name) for name in _ANTHROPIC_ENV):
        return True
    return _ANTHROPIC_PROFILE_DIR.exists()


def build_backend(settings: Settings | None = None) -> LLMBackend:
    """Pick a backend from config: a base URL means OpenAI-compatible."""
    settings = settings or get_settings()
    if settings.uses_openai_compatible:
        from vpi.llm.openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(
            settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

    if not _has_anthropic_credentials(settings):
        raise MissingLLMCredentials(
            "No model credentials found. Either:\n"
            "  · set ANTHROPIC_API_KEY (from console.anthropic.com) in .env, or\n"
            "  · point VPI_LLM_BASE_URL at any OpenAI-compatible endpoint "
            "(OpenRouter, vLLM, LM Studio) and set VPI_LLM_API_KEY.\n"
            "The datalake key is separate — it reads your videos, not the model."
        )

    from vpi.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(
        settings.llm_model,
        api_key=settings.anthropic_api_key,
        effort=settings.llm_effort,
    )


__all__ = [
    "LLMBackend",
    "MissingLLMCredentials",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
    "build_backend",
]
