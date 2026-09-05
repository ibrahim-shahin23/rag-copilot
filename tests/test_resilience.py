import urllib.error

from domain.ports import EmbeddingProvider, LLMProvider, StreamingLLMProvider
from infrastructure.resilience.fallback_providers import (
    FallbackEmbeddingProvider,
    FallbackLLMProvider,
    FallbackStreamingLLMProvider,
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


# --- FR-6: streaming fallback ---

class _FailsImmediatelyStreaming(StreamingLLMProvider):
    name = "fails-immediately-streaming"

    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("not configured")

    def stream_complete(self, system_prompt, user_prompt):
        raise RuntimeError("not configured")
        yield  # pragma: no cover - unreachable, makes this a generator function


class _FailsMidStreamStreaming(StreamingLLMProvider):
    name = "fails-mid-stream"

    def complete(self, system_prompt, user_prompt):
        raise RuntimeError("n/a")

    def stream_complete(self, system_prompt, user_prompt):
        yield "partial "
        yield "output "
        raise urllib.error.HTTPError("https://example.com", 429, "Too Many Requests", {}, None)


class _WorkingStreamingLLM(StreamingLLMProvider):
    name = "working-streaming-llm"

    def complete(self, system_prompt, user_prompt):
        return "fallback answer"

    def stream_complete(self, system_prompt, user_prompt):
        yield "fallback "
        yield "answer"


def test_streaming_llm_degrades_immediately_on_first_chunk_failure():
    provider = FallbackStreamingLLMProvider(
        primary=_FailsImmediatelyStreaming(), secondary=_WorkingStreamingLLM()
    )
    chunks = list(provider.stream_complete("system", "user"))
    assert "".join(chunks) == "fallback answer"


def test_streaming_llm_degrades_mid_stream_by_restarting_with_secondary():
    """Documented limitation: a failure partway through primary's stream
    means the client sees primary's partial output, THEN secondary's full
    output from scratch — not a seamless resume. This test pins down that
    exact (stated, not hidden) behavior."""
    provider = FallbackStreamingLLMProvider(
        primary=_FailsMidStreamStreaming(), secondary=_WorkingStreamingLLM()
    )
    chunks = list(provider.stream_complete("system", "user"))
    assert "".join(chunks) == "partial output fallback answer"


def test_streaming_llm_no_fallback_needed_when_primary_works():
    provider = FallbackStreamingLLMProvider(
        primary=_WorkingStreamingLLM(), secondary=_FailsImmediatelyStreaming()
    )
    chunks = list(provider.stream_complete("system", "user"))
    assert "".join(chunks) == "fallback answer"