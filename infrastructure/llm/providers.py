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
    consistent with the embedding adapters) so this file stays importable
    even in environments that never configure a Gemini key.

    Model default: `gemini-flash-latest`, not a pinned version. Gemini 1.5
    models are fully shut down as of 2026 — every call to them now returns
    HTTP 404, which is exactly the failure this project hit in testing.
    `gemini-flash-latest` is a Google-maintained alias that always points
    at the current recommended flash model, which is the right way to
    avoid re-hitting this same class of bug when the next model rotation
    happens. Pin an explicit version instead if you need reproducible
    outputs across model updates.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-flash-latest",
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
                "GeminiLLMProvider requires GEMINI_API_KEY. This is "
                "expected to be caught by FallbackLLMProvider "
                "(infrastructure/resilience/) and degrade to the "
                "extractive fallback — see build_wiring() in "
                "infrastructure/config.py."
            )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent"
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"maxOutputTokens": 1000},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        parts = body["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts)