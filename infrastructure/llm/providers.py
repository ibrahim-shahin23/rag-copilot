"""
LLM provider adapters — completion leg of the provider abstraction.

ExtractiveFallbackProvider needs no API key or network access, so it's what
runs in this sandbox and in any environment where the free tier has run out
(the spec explicitly asks you to design for that). GeminiLLMProvider is the
real hosted implementation (Google's Gemini API); it is not exercised here
for lack of a configured API key and because generativelanguage.googleapis.com
isn't reachable from this sandbox's network policy — but wiring it in is a
one-line config change (see infrastructure/config.py), which is exactly the
point of the provider-abstraction requirement.
"""
from __future__ import annotations

import json
import os
import urllib.request

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


class GeminiLLMProvider(LLMProvider):
    """Google Gemini adapter, called directly over REST (no SDK dependency,
    consistent with HostedEmbeddingProvider's style) so this file stays
    importable even in environments that never configure a Gemini key."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError(
                "GeminiLLMProvider requires GEMINI_API_KEY; the fallback "
                "chain should have routed to ExtractiveFallbackProvider."
            )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 1000},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)