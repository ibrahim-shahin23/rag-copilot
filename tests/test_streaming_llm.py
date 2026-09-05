import json
from unittest.mock import patch, MagicMock

import pytest

from infrastructure.llm.providers import ExtractiveFallbackProvider, GeminiLLMProvider


# --- ExtractiveFallbackProvider.stream_complete ---

def test_extractive_stream_complete_reconstructs_exact_text():
    provider = ExtractiveFallbackProvider()
    system = "sys"
    user = "Question: What is FR-2?\n\nExcerpts:\n[1] some excerpt text here"

    full_text = provider.complete(system, user)
    streamed_chunks = list(provider.stream_complete(system, user))

    assert "".join(streamed_chunks) == full_text  # concatenation must be lossless
    assert len(streamed_chunks) > 1  # actually chunked, not one giant blob


def test_extractive_stream_complete_yields_multiple_chunks_for_long_text():
    provider = ExtractiveFallbackProvider()
    user = "Question: Q\n\nExcerpts:\n[1] " + " ".join(f"word{i}" for i in range(20))
    chunks = list(provider.stream_complete("sys", user))
    assert len(chunks) > 10


# --- GeminiLLMProvider.stream_complete ---

def test_gemini_stream_complete_raises_without_api_key():
    provider = GeminiLLMProvider(api_key=None)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        list(provider.stream_complete("sys", "user"))


def test_gemini_stream_complete_parses_sse_response():
    provider = GeminiLLMProvider(api_key="fake-key")

    sse_body = (
        b'data: {"candidates": [{"content": {"parts": [{"text": "Hello "}]}}]}\n\n'
        b'data: {"candidates": [{"content": {"parts": [{"text": "world"}]}}]}\n\n'
    )
    # Simulate iterating a urllib response object line-by-line.
    lines = sse_body.splitlines(keepends=True)

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    mock_resp.__iter__.return_value = iter(lines)

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        chunks = list(provider.stream_complete("sys", "user prompt"))

    assert chunks == ["Hello ", "world"]
    sent_request = mock_urlopen.call_args[0][0]
    assert "streamGenerateContent" in sent_request.full_url
    assert "alt=sse" in sent_request.full_url
    assert sent_request.headers.get("X-goog-api-key") == "fake-key"


def test_gemini_stream_complete_skips_malformed_sse_lines_without_crashing():
    provider = GeminiLLMProvider(api_key="fake-key")
    sse_body = (
        b"data: not valid json at all\n\n"
        b'data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}\n\n'
        b"\n"  # blank keep-alive line, no "data:" prefix
    )
    lines = sse_body.splitlines(keepends=True)

    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = False
    mock_resp.__iter__.return_value = iter(lines)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        chunks = list(provider.stream_complete("sys", "user"))

    assert chunks == ["ok"]  # malformed line skipped, not raised