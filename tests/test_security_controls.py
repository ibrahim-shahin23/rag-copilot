import tempfile
import shutil
import pytest
from fastapi.testclient import TestClient

from interface.http_api import app, sanitize_secrets
from domain.auth_entities import Role, User

CONTRIBUTOR_KEY = "contributor-demo-key"
REVIEWER_KEY = "reviewer-demo-key"


@pytest.fixture()
def client():
    return TestClient(app)


def test_security_headers_present(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("x-xss-protection") == "1; mode=block"
    assert resp.headers.get("content-security-policy") == "default-src 'self'"


def test_cors_headers_present(client):
    resp = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_secret_sanitization_function():
    raw_log = "Error connecting with api_key=AIzaSySecretKey1234567890123456789012 and token=secretTokenVal"
    sanitized = sanitize_secrets(raw_log)
    assert "AIzaSySecretKey" not in sanitized
    assert "secretTokenVal" not in sanitized
    assert "***REDACTED***" in sanitized


def test_rate_limiting_trigger(client, monkeypatch):
    monkeypatch.setattr("interface.http_api.RATE_LIMIT_PER_MINUTE", 3)
    # Reset store for clean test environment
    from interface.http_api import _RATE_LIMIT_STORE
    _RATE_LIMIT_STORE.clear()

    headers = {"X-API-Key": CONTRIBUTOR_KEY}
    
    # 3 requests allowed
    r1 = client.get("/sessions", headers=headers)
    r2 = client.get("/sessions", headers=headers)
    r3 = client.get("/sessions", headers=headers)
    
    # 4th request must be rate limited
    r4 = client.get("/sessions", headers=headers)
    assert r4.status_code == 429
    assert "Rate limit exceeded" in r4.json()["detail"]
    assert r4.headers.get("retry-after") == "60"

    # Reset store after test
    _RATE_LIMIT_STORE.clear()


def test_input_validation_oversized_query(client):
    headers = {"X-API-Key": CONTRIBUTOR_KEY}
    oversized_query = "A" * 10001
    resp = client.post("/ask", json={"query": oversized_query}, headers=headers)
    assert resp.status_code == 422  # Unprocessable Entity from FastAPI/Pydantic schema validation
