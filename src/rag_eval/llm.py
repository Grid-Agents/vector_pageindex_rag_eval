from __future__ import annotations

import os
from typing import Any

from .types import LLMResponse, Usage


class AnthropicLLM:
    def __init__(self, cfg: dict[str, Any]):
        self.model = cfg.get("model", "claude-sonnet-4-6")
        self.max_tokens = int(cfg.get("max_tokens", 700))
        self.temperature = float(cfg.get("temperature", 0.0))
        self.pricing = cfg.get("pricing", {})
        api_key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing Claude API key. Set {api_key_env} or change llm.api_key_env."
            )

        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError("Install dependencies with `pip install -e .`.") from exc

        self.client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature if temperature is None else temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text_parts = []
        for block in message.content:
            if getattr(block, "type", "") == "text":
                text_parts.append(getattr(block, "text", ""))
        usage = self._usage_from_response(getattr(message, "usage", None))
        return LLMResponse(text="".join(text_parts).strip(), usage=usage)

    def _usage_from_response(self, raw_usage: Any) -> Usage:
        input_tokens = int(getattr(raw_usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(raw_usage, "output_tokens", 0) or 0)
        cache_read = int(getattr(raw_usage, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(
            getattr(raw_usage, "cache_creation_input_tokens", 0) or 0
        )
        cost = (
            input_tokens * float(self.pricing.get("input_per_mtok_usd", 0.0))
            + output_tokens * float(self.pricing.get("output_per_mtok_usd", 0.0))
            + cache_read * float(self.pricing.get("cache_read_per_mtok_usd", 0.0))
            + cache_creation
            * float(self.pricing.get("cache_write_5m_per_mtok_usd", 0.0))
        ) / 1_000_000
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            estimated_cost_usd=cost,
        )


class FakeLLM:
    """Small test double used by unit tests and examples."""

    def __init__(self, text: str = "{}"):
        self.text = text

    def complete(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        return LLMResponse(text=self.text, usage=Usage())
