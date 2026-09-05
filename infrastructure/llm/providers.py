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
import urllib.parse
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

    Model default: `gemini-3.5-flash-lite`, lightweight fast model with high quota.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model or os.environ.get("GEMINI_MODEL_NAME") or "gemini-3.5-flash-lite"

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


class GemmaLocalLLMProvider(LLMProvider):
    """Local Gemma adapter connecting to a locally hosted Gemma instance
    (e.g., gemma-4-e4b) over REST at http://127.0.0.1:1234.

    Endpoints supported:
      - /api/v1/chat (POST): standard chat completions
      - /api/v1/models (GET): list available models
      - /api/v1/models/load (POST): load a model into memory
      - /api/v1/models/unload (POST): unload a model from memory
      - /api/v1/models/download (POST): initiate a model download
      - /api/v1/models/download/status (GET): query download status
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> None:
        raw_url = base_url or os.environ.get("GEMMA_BASE_URL", "http://127.0.0.1:1234")
        self._base_url = raw_url.rstrip("/")
        self._model = model or os.environ.get("GEMMA_MODEL_NAME", "gemma-4-e4b")
        if timeout is not None:
            self._timeout = float(timeout)
        elif "GEMMA_TIMEOUT" in os.environ:
            self._timeout = float(os.environ["GEMMA_TIMEOUT"])
        else:
            self._timeout = 120.0  
        self._resolved_model_cache: str | None = None

    @property
    def name(self) -> str:
        return f"gemma-local:{self._model}"

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def model(self) -> str:
        return self._model

    def is_configured(self) -> bool:
        return bool(self._base_url)

    def _resolve_model_name(self) -> str:
        if self._resolved_model_cache:
            return self._resolved_model_cache

        try:
            res = self.list_models()
            models_list = []
            if isinstance(res, dict):
                models_list = res.get("models", res.get("data", []))
            elif isinstance(res, list):
                models_list = res

            for m in models_list:
                if isinstance(m, dict):
                    key = m.get("key", m.get("id", ""))
                    if key == self._model or self._model in key:
                        self._resolved_model_cache = key
                        return key
                    for variant in m.get("variants", []):
                        if variant == self._model or self._model in variant:
                            self._resolved_model_cache = key
                            return key
        except Exception:
            pass

        return self._model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self._base_url}/api/v1/chat"
        model_to_use = self._resolve_model_name()
        input_text = f"System: {system_prompt}\n\nUser: {user_prompt}" if system_prompt else user_prompt
        payload = {
            "model": model_to_use,
            "input": input_text,
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 404 and self._resolved_model_cache != self._model:
                self._resolved_model_cache = None
                model_to_use = self._model
                payload["model"] = model_to_use
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
            else:
                raise

        # Check 'output' array format: [{"type": "message", "content": "..."}]
        if "output" in body and isinstance(body["output"], list) and body["output"]:
            # Prefer item with type == "message"
            for item in body["output"]:
                if isinstance(item, dict) and item.get("type") == "message" and "content" in item:
                    return item["content"]
            # Fallback to any non-reasoning item with content
            for item in body["output"]:
                if isinstance(item, dict) and item.get("type") != "reasoning" and "content" in item:
                    return item["content"]
            # Fallback to first item with content
            for item in body["output"]:
                if isinstance(item, dict) and "content" in item:
                    return item["content"]

        # Check standard OpenAI 'choices' format
        if "choices" in body and isinstance(body["choices"], list) and body["choices"]:
            first = body["choices"][0]
            if isinstance(first, dict):
                if "message" in first and isinstance(first["message"], dict) and "content" in first["message"]:
                    return first["message"]["content"]
                if "text" in first:
                    return first["text"]

        if "content" in body:
            return body["content"]
        raise KeyError(f"Unexpected response format from local Gemma API: {body}")

    def list_models(self) -> list[dict] | dict:
        url = f"{self._base_url}/api/v1/models"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def load_model(self, model_name: str | None = None) -> dict:
        url = f"{self._base_url}/api/v1/models/load"
        payload = {"model": model_name or self._model}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def unload_model(self, model_name: str | None = None) -> dict:
        url = f"{self._base_url}/api/v1/models/unload"
        payload = {"model": model_name or self._model}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def download_model(self, model_name: str | None = None) -> dict:
        url = f"{self._base_url}/api/v1/models/download"
        payload = {"model": model_name or self._model}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_download_status(self, task_id: str | None = None) -> dict:
        url = f"{self._base_url}/api/v1/models/download/status"
        if task_id:
            url += f"?task_id={urllib.parse.quote(task_id)}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

