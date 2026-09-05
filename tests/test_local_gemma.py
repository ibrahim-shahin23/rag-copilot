import json
from io import BytesIO
from unittest.mock import MagicMock, patch
import urllib.error

import pytest
from infrastructure.llm.providers import GemmaLocalLLMProvider


def _mock_response(status=200, json_data=None):
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(json_data or {}).encode("utf-8")
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = None
    return resp


def test_gemma_local_init_defaults(monkeypatch):
    provider = GemmaLocalLLMProvider()
    assert provider.base_url == "http://127.0.0.1:1234"
    assert provider.model == "gemma-4-e4b"
    assert provider.name == "gemma-local:gemma-4-e4b"
    assert provider.is_configured() is True
    assert provider._timeout == 120.0

    monkeypatch.setenv("GEMMA_TIMEOUT", "180")
    provider_env = GemmaLocalLLMProvider()
    assert provider_env._timeout == 180.0


def test_gemma_local_complete_chat_endpoint():
    provider = GemmaLocalLLMProvider(base_url="http://127.0.0.1:1234", model="gemma-4-e4b")
    fake_response = {
        "output": [
            {
                "type": "message",
                "content": "Hello from Gemma local!",
            }
        ]
    }

    with patch.object(provider, "list_models", return_value={"models": [{"key": "google/gemma-4-e4b"}]}), \
         patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        result = provider.complete("System instruction", "User query")
        assert result == "Hello from Gemma local!"

        # Verify request parameters
        assert mock_urlopen.call_count == 1
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/chat"
        assert req.method == "POST"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "google/gemma-4-e4b"
        assert payload["input"] == "System: System instruction\n\nUser: User query"


def test_gemma_local_list_models():
    provider = GemmaLocalLLMProvider()
    fake_response = {
        "object": "list",
        "data": [{"id": "gemma-4-e4b", "object": "model"}],
    }
    with patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        models = provider.list_models()
        assert models == fake_response
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/models"
        assert req.method == "GET"


def test_gemma_local_load_model():
    provider = GemmaLocalLLMProvider()
    fake_response = {"status": "ok", "message": "Model loaded"}
    with patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        res = provider.load_model()
        assert res == fake_response
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/models/load"
        assert req.method == "POST"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "gemma-4-e4b"


def test_gemma_local_unload_model():
    provider = GemmaLocalLLMProvider()
    fake_response = {"status": "ok", "message": "Model unloaded"}
    with patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        res = provider.unload_model("gemma-4-e4b")
        assert res == fake_response
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/models/unload"
        assert req.method == "POST"


def test_gemma_local_download_model():
    provider = GemmaLocalLLMProvider()
    fake_response = {"status": "downloading", "task_id": "dl-123"}
    with patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        res = provider.download_model()
        assert res == fake_response
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/models/download"
        assert req.method == "POST"


def test_gemma_local_download_status():
    provider = GemmaLocalLLMProvider()
    fake_response = {"status": "completed", "progress": 100}
    with patch("urllib.request.urlopen", return_value=_mock_response(200, fake_response)) as mock_urlopen:
        res = provider.get_download_status(task_id="dl-123")
        assert res == fake_response
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:1234/api/v1/models/download/status?task_id=dl-123"
        assert req.method == "GET"


def test_gemma_local_complete_failure_raises_oserror():
    provider = GemmaLocalLLMProvider()
    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
        with pytest.raises(OSError):
            provider.complete("System", "User")
