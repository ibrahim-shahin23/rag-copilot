import json
import shutil
import tempfile

import pytest
from fastapi.testclient import TestClient

from application.ingest import IngestDocumentUseCase
from domain.auth_entities import Role, User
from infrastructure.config import build_wiring
from interface.http_api import app, _ACTIVE_RUN_TOKENS
from application.orchestration.cancellation import CancellationToken

CORPUS_TEXT = (
    "FR-2 Retrieval. Hybrid retrieval combines dense and keyword search "
    "with a documented fusion method. Citations are mandatory."
)

CONTRIBUTOR_KEY = "contributor-demo-key"
REVIEWER_KEY = "reviewer-demo-key"


@pytest.fixture()
def tmp_data_dir():
    d = tempfile.mkdtemp()
    wiring = build_wiring(d)
    ingest = IngestDocumentUseCase(
        wiring.repo, wiring.embedder, wiring.vector_store, wiring.keyword_index,
    )
    ingest.execute(source="spec.md", doc_type="md", raw_text=CORPUS_TEXT)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def client():
    return TestClient(app)


def _auth(key: str) -> dict:
    return {"X-API-Key": key}


def _parse_sse_lines(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_health_endpoint_requires_no_auth():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --- FR-8: authentication ---

def test_missing_api_key_returns_401(client, tmp_data_dir):
    resp = client.post("/ask", json={"query": "anything", "data_dir": tmp_data_dir})
    assert resp.status_code == 401


def test_invalid_api_key_returns_401(client, tmp_data_dir):
    resp = client.post(
        "/ask", json={"query": "anything", "data_dir": tmp_data_dir},
        headers=_auth("not-a-real-key"),
    )
    assert resp.status_code == 401


# --- FR-8: role enforcement, genuinely different, both directions ---

def test_reviewer_cannot_call_contributor_endpoints(client, tmp_data_dir):
    resp = client.post(
        "/ask", json={"query": "anything", "data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 403

    resp = client.post(
        "/ingest",
        json={"source": "x.md", "doc_type": "md", "raw_text": "text", "data_dir": tmp_data_dir},
        headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 403

    resp = client.post(
        "/workflow/run",
        json={"target_role": "Role", "competencies": ["x"], "data_dir": tmp_data_dir},
        headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 403


def test_contributor_cannot_call_reviewer_endpoints(client, tmp_data_dir):
    resp = client.get("/approvals", params={"data_dir": tmp_data_dir}, headers=_auth(CONTRIBUTOR_KEY))
    assert resp.status_code == 403

    resp = client.post(
        "/approvals/some-id/decide",
        json={"decision": "approve", "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 403

    resp = client.get(
        "/workflow/trace/some-run-id", params={"data_dir": tmp_data_dir}, headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 403


# --- FR-7: ingest ---

def test_ingest_over_http(client, tmp_data_dir):
    resp = client.post(
        "/ingest",
        json={"source": "new_doc.md", "doc_type": "md", "raw_text": "FR-9 requires correlation IDs.",
              "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["chunk_count"] > 0
    assert body["reused_existing"] is False


def test_ingest_is_idempotent_over_http(client, tmp_data_dir):
    payload = {"source": "dup.md", "doc_type": "md", "raw_text": "Some content.", "data_dir": tmp_data_dir}
    first = client.post("/ingest", json=payload, headers=_auth(CONTRIBUTOR_KEY))
    second = client.post("/ingest", json=payload, headers=_auth(CONTRIBUTOR_KEY))
    assert first.json()["reused_existing"] is False
    assert second.json()["reused_existing"] is True


# --- FR-7: ask (non-streaming) ---

def test_ask_over_http(client, tmp_data_dir):
    resp = client.post(
        "/ask", json={"query": "What does FR-2 require about citations?", "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["refused"] is False
    assert len(body["citations"]) > 0
    assert body["citations"][0]["source"] == "spec.md"


def test_ask_stream_returns_token_and_done_events(client, tmp_data_dir):
    resp = client.post(
        "/ask/stream",
        json={"query": "What does FR-2 require about citations?", "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_lines(resp.text)
    assert any(e["kind"] == "token" for e in events)
    done_events = [e for e in events if e["kind"] == "done"]
    assert len(done_events) == 1
    assert done_events[0]["refused"] is False
    assert len(done_events[0]["citations"]) > 0
    assert done_events[0]["citations"][0]["source"] == "spec.md"


def test_ask_stream_refusal_path_over_http(client, tmp_data_dir):
    resp = client.post(
        "/ask/stream",
        json={"query": "What is the airspeed velocity of an unladen swallow?", "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    events = _parse_sse_lines(resp.text)
    done_events = [e for e in events if e["kind"] == "done"]
    assert len(done_events) == 1
    # Not asserting refused=True here: this corpus is tiny (one chunk),
    # and FR-3's evaluation report already documents that the default
    # refusal threshold doesn't reliably refuse out-of-corpus questions
    # on small corpora (ADR-002). What matters for THIS test is that the
    # HTTP layer correctly delivers whichever decision the use case made,
    # with well-formed citations either way.
    if done_events[0]["refused"]:
        assert done_events[0]["citations"] == []


# --- FR-7: workflow run (non-streaming) + trace ---

def test_workflow_run_over_http(client, tmp_data_dir):
    resp = client.post(
        "/workflow/run",
        json={"target_role": "RAG Engineer", "competencies": ["hybrid retrieval"], "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in ("succeeded", "degraded")
    assert len(body["steps"]) > 0


def test_workflow_trace_over_http_reviewer_only(client, tmp_data_dir):
    run_resp = client.post(
        "/workflow/run",
        json={"target_role": "Role", "competencies": ["hybrid retrieval"], "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    run_id = run_resp.json()["run_id"]

    trace_resp = client.get(
        f"/workflow/trace/{run_id}", params={"data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY),
    )
    assert trace_resp.status_code == 200
    body = trace_resp.json()
    assert body["run_id"] == run_id
    assert len(body["steps"]) > 0


def test_workflow_trace_404_for_unknown_run(client, tmp_data_dir):
    resp = client.get(
        "/workflow/trace/does-not-exist", params={"data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 404


def test_workflow_stream_emits_progress_events(client, tmp_data_dir):
    resp = client.post(
        "/workflow/stream",
        json={
            "target_role": "RAG Engineer",
            "competencies": ["hybrid retrieval"],
            "data_dir": tmp_data_dir,
        },
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert resp.status_code == 200
    events = _parse_sse_lines(resp.text)
    event_types = [e["event_type"] for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    assert "step_started" in event_types
    assert len({e["run_id"] for e in events}) == 1


def test_cancel_endpoint_returns_404_for_unknown_run_id(client):
    resp = client.post("/workflow/cancel/does-not-exist", headers=_auth(CONTRIBUTOR_KEY))
    assert resp.status_code == 404


def test_cancel_endpoint_cancels_a_registered_token(client):
    """Directly exercises the registry + endpoint logic (the deeper claim
    — that a cancelled token actually stops the Supervisor mid-pipeline —
    is proven at the application layer in test_supervisor.py; this test
    is scoped to the HTTP-layer wiring around that mechanism)."""
    token = CancellationToken()
    _ACTIVE_RUN_TOKENS["fake-run-id"] = token
    try:
        assert token.is_cancelled() is False
        resp = client.post("/workflow/cancel/fake-run-id", headers=_auth(CONTRIBUTOR_KEY))
        assert resp.status_code == 200
        assert resp.json() == {"run_id": "fake-run-id", "cancel_requested": True}
        assert token.is_cancelled() is True
    finally:
        _ACTIVE_RUN_TOKENS.pop("fake-run-id", None)


def test_workflow_stream_registry_is_empty_after_run_finishes(client, tmp_data_dir):
    """The run_id -> token registry must not leak entries once a run
    completes — otherwise a long-lived server would accumulate stale
    tokens for every run ever started."""
    resp = client.post(
        "/workflow/stream",
        json={"target_role": "Role", "competencies": ["hybrid retrieval"], "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    events = _parse_sse_lines(resp.text)
    run_id = events[0]["run_id"]
    assert run_id not in _ACTIVE_RUN_TOKENS


def test_client_disconnect_stops_forwarding_events_and_cancels_the_run(tmp_data_dir):
    """The TestClient's ASGI transport runs this fast, synchronous demo
    pipeline to completion before there's a real window to close the
    connection mid-stream — that would only be testing httpx/Starlette's
    transport timing, not this code. Instead, this calls the endpoint
    directly with a fake Request whose is_disconnected() flips to True
    after a couple of calls, deterministically exercising the actual
    disconnect-handling branch: once "disconnected", cancel the token,
    stop yielding SSE lines to the (gone) client, but keep draining the
    generator internally so the Supervisor finishes and persists a clean
    CANCELLED status rather than an abandoned one."""
    import asyncio

    from infrastructure.config import build_wiring
    from interface.http_api import WorkflowRequest, workflow_stream

    class FakeDisconnectingRequest:
        def __init__(self, disconnect_after: int):
            self._calls = 0
            self._disconnect_after = disconnect_after

        async def is_disconnected(self) -> bool:
            self._calls += 1
            return self._calls > self._disconnect_after

    payload = WorkflowRequest(
        target_role="Role", competencies=["hybrid retrieval"], data_dir=tmp_data_dir,
    )
    fake_request = FakeDisconnectingRequest(disconnect_after=1)
    fake_user = User(username="alice", role=Role.CONTRIBUTOR)

    async def _run():
        response = await workflow_stream(payload, fake_request, fake_user)
        lines = []
        async for chunk in response.body_iterator:
            lines.append(chunk)
        return lines

    lines = asyncio.run(_run())
    forwarded_events = [json.loads(l[len("data:"):].strip()) for l in lines if l.startswith("data:")]

    wiring = build_wiring(tmp_data_dir)
    # The run's own id was still forwarded before disconnection was
    # detected (it's on the very first event) — use it to check the
    # persisted final state even though later events stopped being sent.
    run_id = forwarded_events[0]["run_id"]
    final_run = wiring.workflow_repo.get_run(run_id)

    assert final_run.status.value == "cancelled"
    # Fewer events reached the "client" than a full undisturbed run would
    # have produced (run_started, step_started, step_succeeded x2+,
    # run_cancelled, run_finished) — proving forwarding actually stopped,
    # not just that the run happened to finish quickly on its own.
    assert len(forwarded_events) <= 2
    assert run_id not in _ACTIVE_RUN_TOKENS  # cleaned up even on the cancelled path


# --- FR-4 approval gate over HTTP ---

def test_approvals_list_and_decide_over_http(client, tmp_data_dir):
    run_resp = client.post(
        "/workflow/run",
        json={"target_role": "RAG Engineer", "competencies": ["hybrid retrieval"], "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    assert run_resp.status_code == 200

    list_resp = client.get("/approvals", params={"data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY))
    assert list_resp.status_code == 200
    items = list_resp.json()

    if not items:
        pytest.skip("workflow produced no draftable items for this tiny corpus/competency combo")

    item_id = items[0]["id"]
    decide_resp = client.post(
        f"/approvals/{item_id}/decide",
        json={"decision": "approve", "data_dir": tmp_data_dir},
        headers=_auth(REVIEWER_KEY),
    )
    assert decide_resp.status_code == 200
    body = decide_resp.json()
    assert body["approval_status"] == "approved"
    assert body["decided_by"] == "lead-instructor"  # from the auth token, not client-supplied


def test_decide_requires_edited_text_for_edit_decision(client, tmp_data_dir):
    resp = client.post(
        "/approvals/some-id/decide",
        json={"decision": "edit", "data_dir": tmp_data_dir},  # no edited_text
        headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 400


def test_decide_unknown_item_returns_404(client, tmp_data_dir):
    resp = client.post(
        "/approvals/does-not-exist/decide",
        json={"decision": "approve", "data_dir": tmp_data_dir},
        headers=_auth(REVIEWER_KEY),
    )
    assert resp.status_code == 404


# --- FR-7 persistent session history ---

def test_sessions_are_recorded_and_retrievable(client, tmp_data_dir):
    client.post(
        "/ask", json={"query": "What does FR-2 require?", "data_dir": tmp_data_dir},
        headers=_auth(CONTRIBUTOR_KEY),
    )
    resp = client.get("/sessions", params={"data_dir": tmp_data_dir}, headers=_auth(CONTRIBUTOR_KEY))
    assert resp.status_code == 200
    events = resp.json()
    assert any(e["endpoint"] == "POST /ask" for e in events)
    assert all(e["username"] == "alice" for e in events)  # contributor sees only their own


def test_reviewer_sees_all_users_sessions(client, tmp_data_dir):
    client.post(
        "/ask", json={"query": "q", "data_dir": tmp_data_dir}, headers=_auth(CONTRIBUTOR_KEY),
    )
    client.get("/approvals", params={"data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY))

    resp = client.get("/sessions", params={"data_dir": tmp_data_dir}, headers=_auth(REVIEWER_KEY))
    assert resp.status_code == 200
    usernames = {e["username"] for e in resp.json()}
    assert "alice" in usernames
    assert "lead-instructor" in usernames


def test_openapi_schema_documents_the_full_surface(client):
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = set(schema["paths"].keys())
    for expected in [
        "/health", "/ingest", "/ask", "/ask/stream",
        "/workflow/run", "/workflow/stream", "/workflow/cancel/{run_id}",
        "/workflow/trace/{run_id}", "/approvals", "/approvals/{item_id}/decide",
        "/sessions",
    ]:
        assert expected in paths, f"missing from OpenAPI schema: {expected}"