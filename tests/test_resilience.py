import urllib.error

from domain.ports import EmbeddingProvider, LLMProvider
from infrastructure.resilience.fallback_providers import (
    FallbackEmbeddingProvider,
    FallbackLLMProvider,
)


class _AlwaysRateLimited(EmbeddingProvider):
    """Simulates the exact bug report: a configured, valid API key that
    still fails per-call with HTTP 429 (quota/rate limit exhausted)."""
    name = "always-429"

    def embed(self, texts):
        raise urllib.error.HTTPError(
            "https://api.openai.com/v1/embeddings", 429, "Too Many Requests", {}, None
        )


class _WorkingLocalEmbedder(EmbeddingProvider):
    name = "local-stub"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _AlwaysFailingLLM(LLMProvider):
    name = "always-fails-llm"

    def complete(self, system_prompt, user_prompt):
        raise urllib.error.HTTPError(
            "https://generativelanguage.googleapis.com/x", 429, "Too Many Requests", {}, None
        )


class _WorkingLLM(LLMProvider):
    name = "working-llm"

    def complete(self, system_prompt, user_prompt):
        return "fallback answer"


def test_embedding_429_degrades_instead_of_crashing():
    provider = FallbackEmbeddingProvider(
        primary=_AlwaysRateLimited(), secondary=_WorkingLocalEmbedder()
    )
    # This must NOT raise — the whole point of the fix.
    result = provider.embed(["some text"])
    assert result == [[1.0, 0.0]]


def test_llm_429_degrades_instead_of_crashing():
    provider = FallbackLLMProvider(primary=_AlwaysFailingLLM(), secondary=_WorkingLLM())
    result = provider.complete("system", "user")
    assert result == "fallback answer"


def test_not_configured_runtime_error_also_degrades():
    """The 'no API key at all' case must degrade the same way as a 429 —
    both are RuntimeError/OSError-family failures caught by the same path."""

    class _Unconfigured(EmbeddingProvider):
        name = "unconfigured"

        def embed(self, texts):
            raise RuntimeError("requires OPENAI_API_KEY")

    provider = FallbackEmbeddingProvider(
        primary=_Unconfigured(), secondary=_WorkingLocalEmbedder()
    )
    assert provider.embed(["x"]) == [[1.0, 0.0]]
