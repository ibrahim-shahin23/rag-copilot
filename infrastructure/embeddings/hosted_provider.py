"""
Hosted-API embedding provider — the second leg of the provider abstraction.

This adapter is written against OpenAI's embeddings endpoint as a concrete
example of "a hosted API" per the spec's provider-abstraction requirement.
It requires OPENAI_API_KEY and network access to api.openai.com — neither
is available in this sandbox, so it is not exercised by the test suite here.
The documented fallback chain (infrastructure/config.py) falls back to
TfidfEmbeddingProvider when this adapter's health check fails, which is the
"free tier running out" scenario the spec explicitly asks you to design for.
"""
from __future__ import annotations

import os
from typing import Sequence

import urllib.request
import json

from domain.ports import EmbeddingProvider


class HostedEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small") -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model

    @property
    def name(self) -> str:
        return f"hosted-openai:{self._model}"

    def is_configured(self) -> bool:
        return bool(self._api_key)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._api_key:
            raise RuntimeError(
                "HostedEmbeddingProvider requires OPENAI_API_KEY; "
                "the fallback chain should have routed to the local provider "
                "instead of reaching this call."
            )
        req = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps({"model": self._model, "input": list(texts)}).encode(),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        return [item["embedding"] for item in body["data"]]
