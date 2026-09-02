"""
Runtime fallback wrappers — the actual mechanism behind the "documented
fallback chain" the provider-abstraction requirement asks for.

Prior versions of infrastructure/config.py only checked *is an API key
present* at startup (`is_configured()`) and picked a provider once for the
whole process. That missed the far more common real failure mode: the key
is present and valid, but the call itself fails at request time — quota
exhausted (HTTP 429), rate limited, a transient network error. That
crashed the whole CLI with an unhandled HTTPError instead of degrading,
which is exactly the "design for a free tier running out" scenario the
spec calls out by name.

These wrappers catch that class of failure at the call site and degrade to
the secondary provider instead, printing a one-line notice to stderr so
the degradation is visible, never silent.

Caught exception types are deliberately scoped to plausible provider
failures — OSError (covers urllib's URLError/HTTPError and general network
faults), RuntimeError (the "not configured" case raised by the hosted
adapters themselves), ValueError (covers json.JSONDecodeError on a
malformed response body), and KeyError (an unexpected response shape). A
bug elsewhere in this code should still surface as itself, not get
silently absorbed as "provider failure."

Known limitation: falling back to a *different* provider mid-session means
chunks already indexed with the primary provider's embedding space won't
dimensionally match a query embedded by the secondary provider.
NumpyVectorStore.query already handles that safely — it skips mismatched
vectors rather than crashing — so the result is reduced recall, not
another crash, but it's a real limitation, not a solved problem. See
PLAN.md's roadmap for embedding-provider versioning as the actual fix.
"""
from __future__ import annotations

import sys
from typing import Sequence

from domain.ports import EmbeddingProvider, LLMProvider

_CAUGHT = (OSError, RuntimeError, ValueError, KeyError)


class FallbackEmbeddingProvider(EmbeddingProvider):
    def __init__(self, primary: EmbeddingProvider, secondary: EmbeddingProvider) -> None:
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        return f"fallback({self._primary.name}->{self._secondary.name})"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            return self._primary.embed(texts)
        except _CAUGHT as e:
            print(
                f"[fallback] embedder '{self._primary.name}' failed ({e!r}); "
                f"degrading to '{self._secondary.name}' for this call",
                file=sys.stderr,
            )
            return self._secondary.embed(texts)


class FallbackLLMProvider(LLMProvider):
    def __init__(self, primary: LLMProvider, secondary: LLMProvider) -> None:
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        return f"fallback({self._primary.name}->{self._secondary.name})"

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            return self._primary.complete(system_prompt, user_prompt)
        except _CAUGHT as e:
            print(
                f"[fallback] LLM '{self._primary.name}' failed ({e!r}); "
                f"degrading to '{self._secondary.name}' for this call",
                file=sys.stderr,
            )
            return self._secondary.complete(system_prompt, user_prompt)
