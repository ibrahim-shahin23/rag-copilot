"""
Gemini embedding provider — the hosted leg of the embedding provider
abstraction, replacing OpenAI's paid endpoint (hosted_provider.py) as the
default primary. `gemini-embedding-001` is on a genuine free tier (no
card required, generous daily quota as of 2026), so this consolidates
embeddings and completion onto the same free GEMINI_API_KEY instead of
needing a second, paid provider.

hosted_provider.py (OpenAI) is left in place, not deleted — it's a valid
alternative for anyone who already has OpenAI credits and wants to use
them instead — but it's no longer what infrastructure/config.py wires by
default.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Sequence

from domain.ports import EmbeddingProvider


class GeminiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None = None, model: str = "gemini-embedding-001") -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model

    @property
    def name(self) -> str:
        return f"gemini-embedding:{self._model}"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._api_key:
            raise RuntimeError(
                "GeminiEmbeddingProvider requires GEMINI_API_KEY. This is "
                "expected to be caught by FallbackEmbeddingProvider "
                "(infrastructure/resilience/) and degrade to the local "
                "TF-IDF provider — see build_wiring() in infrastructure/config.py."
            )
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:batchEmbedContents"
        )
        payload = {
            "requests": [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": text}]},
                }
                for text in texts
            ]
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
        return [item["values"] for item in body["embeddings"]]