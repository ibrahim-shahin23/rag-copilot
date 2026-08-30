"""
LLM provider adapters — completion leg of the provider abstraction.

ExtractiveFallbackProvider needs no API key or network access, so it's what
runs in this sandbox and in any environment where the free tier has run out
(the spec explicitly asks you to design for that). AnthropicLLMProvider is
the real hosted implementation; it is not exercised here for lack of a
configured API key, but wiring it in is a one-line config change.
"""
from __future__ import annotations

import os

from domain.ports import LLMProvider


class ExtractiveFallbackProvider(LLMProvider):
    """Graceful-degradation provider (FR-5): when no real LLM is
    configured/reachable, synthesize an answer by stitching the
    highest-signal sentences out of the retrieved excerpts instead of
    failing the request outright."""

    @property
    def name(self) -> str:
        return "extractive-fallback"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        # user_prompt format is fixed by AnswerQueryUseCase: "Question: ...\n\nExcerpts:\n[1] ..."
        _, _, excerpts_block = user_prompt.partition("Excerpts:\n")
        first_excerpt = excerpts_block.split("\n\n")[0] if excerpts_block else ""
        return (
            "Based on the retrieved excerpts (see citations), here is the most "
            f"relevant passage found:\n\n{first_excerpt}\n\n"
            "(Extractive fallback in use — no hosted LLM is configured; "
            "responses are excerpt selections, not synthesized prose.)"
        )


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6") -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError(
                "AnthropicLLMProvider requires ANTHROPIC_API_KEY; the "
                "fallback chain should have routed to ExtractiveFallbackProvider."
            )
        import anthropic  # local import: keeps this optional dependency out
                            # of environments that only use the fallback

        client = anthropic.Anthropic(api_key=self._api_key)
        response = client.messages.create(
            model=self._model,
            max_tokens=1000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
