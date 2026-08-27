"""LLM backends. Claude by default; any OpenAI-compatible endpoint as fallback."""

from __future__ import annotations

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

    from vpi.llm.anthropic_backend import AnthropicBackend

    return AnthropicBackend(
        settings.llm_model,
        api_key=settings.anthropic_api_key,
        effort=settings.llm_effort,
    )


__all__ = [
    "LLMBackend",
    "LLMResponse",
    "ToolCall",
    "ToolResult",
    "ToolSpec",
    "Turn",
    "Usage",
    "build_backend",
]
