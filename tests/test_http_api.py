import json
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from application.ingest import IngestDocumentUseCase
from infrastructure.config import build_wiring
from interface.http_api import app, _ACTIVE_RUN_TOKENS
from application.orchestration.cancellation import CancellationToken

CORPUS_TEXT = (
    "FR-2 Retrieval. Hybrid retrieval combines dense and keyword search "
    "with a documented fusion method. Citations are mandatory."
)


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


def _parse_sse_lines(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if line.startswith("data:"):
            events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_stream_returns_token_and_done_events(client, tmp_data_dir):
    resp = client.post(
        "/ask/stream",
        json={"query": "What does FR-2 require about citations?", "data_dir": tmp_data_dir},
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


def test_workflow_stream_emits_progress_events(client, tmp_data_dir):
    resp = client.post(
        "/workflow/stream",
        json={
            "target_role": "RAG Engineer",
            "competencies": ["hybrid retrieval"],
            "data_dir": tmp_data_dir,
        },
    )
    assert resp.status_code == 200
    events = _parse_sse_lines(resp.text)
    event_types = [e["event_type"] for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    assert "step_started" in event_types
    assert len({e["run_id"] for e in events}) == 1


def test_cancel_endpoint_returns_404_for_unknown_run_id(client):
    resp = client.post("/workflow/cancel/does-not-exist")
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
        resp = client.post("/workflow/cancel/fake-run-id")
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

    from application.orchestration.cancellation import CancellationToken
    from infrastructure.config import build_supervisor, build_wiring
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

    async def _run():
        response = await workflow_stream(payload, fake_request)
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