"""
Minimal HTTP transport for FR-6 (real-time). This is deliberately NOT the
full FR-7 surface (no OpenAPI-documented CRUD for every operation, no
auth, no persistent session history) — just the two SSE endpoints that
streaming and cancellation actually require, since neither can be
demonstrated meaningfully over a CLI. FR-7 will build the rest of the API
around this.

Two independent cancellation mechanisms, both real and tested:
  1. Connection-close detection: `await request.is_disconnected()` is
     polled between events; once true, the CancellationToken is set and
     no further steps/chunks are processed. This is how a browser's
     `EventSource.close()` or an aborted `fetch()` naturally cancels a
     stream.
  2. An explicit `POST /workflow/cancel/{run_id}` endpoint, for a client
     that wants to cancel from a different connection than the one
     streaming events (e.g. a "Cancel" button in a UI, which is exactly
     the kind of client action FR-6 has in mind) — EventSource in
     particular gives browser code no hook to signal a server any other
     way, so this is the more realistic mechanism for a real UI, not just
     a fallback.

Both ultimately call the same CancellationToken.cancel() that
Supervisor.run_streaming() already checks at every step boundary — see
application/orchestration/cancellation.py and ADR-005.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import AsyncIterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from application.orchestration.cancellation import CancellationToken
from application.retrieve import AnswerQueryUseCase
from infrastructure.config import build_supervisor, build_wiring

app = FastAPI(title="RAG Copilot API (FR-6 streaming slice)")

# In-memory registry of active workflow runs' cancellation tokens, keyed
# by run_id. Populated once a run's id is known (the first event out of
# run_streaming carries it) and removed once the run finishes. Module-level
# and unsynchronized deliberately: this is a single-process demo transport,
# not a production job registry — a real deployment would need this
# durable and shared across workers, tracked in PLAN.md's roadmap.
_ACTIVE_RUN_TOKENS: dict[str, CancellationToken] = {}


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


class AskRequest(BaseModel):
    query: str
    data_dir: str = "data"


class WorkflowRequest(BaseModel):
    target_role: str
    competencies: list[str]
    data_dir: str = "data"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask/stream")
async def ask_stream(payload: AskRequest, request: Request) -> StreamingResponse:
    wiring = build_wiring(payload.data_dir)
    use_case = AnswerQueryUseCase(
        embedder=wiring.embedder, vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index, llm=wiring.llm,
    )

    async def event_generator() -> AsyncIterator[str]:
        for event in use_case.execute_streaming(payload.query):
            if await request.is_disconnected():
                # Stated limitation (see ADR-005): if a hosted LLM call is
                # already in flight, we can't preempt it mid-request —
                # this stops FORWARDING further chunks to a client that's
                # gone, which is the meaningful half of "stop server-side
                # work" achievable without process-level isolation.
                break
            if event.kind == "token":
                yield _sse_event({"kind": "token", "text": event.text})
            else:
                answer = event.answer
                yield _sse_event({
                    "kind": "done",
                    "refused": answer.refused,
                    "text": answer.text,
                    "citations": [
                        {
                            "chunk_id": c.chunk_id, "source": c.source,
                            "section": c.section, "position": c.position, "score": c.score,
                        }
                        for c in answer.citations
                    ],
                })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/workflow/stream")
async def workflow_stream(payload: WorkflowRequest, request: Request) -> StreamingResponse:
    wiring = build_wiring(payload.data_dir)
    supervisor = build_supervisor(wiring)
    token = CancellationToken()

    async def event_generator() -> AsyncIterator[str]:
        run_id: str | None = None
        client_gone = False
        try:
            for event in supervisor.run_streaming(
                payload.target_role, payload.competencies, cancellation_token=token,
            ):
                if run_id is None:
                    run_id = event.run_id
                    _ACTIVE_RUN_TOKENS[run_id] = token

                if not client_gone and await request.is_disconnected():
                    client_gone = True
                    token.cancel()

                if client_gone:
                    # Keep draining the generator (still calling next() via
                    # this for-loop) so the Supervisor observes the
                    # cancellation and persists a clean CANCELLED run
                    # state — but stop sending anything to a connection
                    # nobody is reading from anymore.
                    continue

                yield _sse_event({
                    "run_id": event.run_id, "event_type": event.event_type,
                    "step_index": event.step_index, "agent_name": event.agent_name,
                    "message": event.message,
                })
        finally:
            if run_id is not None:
                _ACTIVE_RUN_TOKENS.pop(run_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/workflow/cancel/{run_id}")
async def cancel_workflow(run_id: str) -> dict:
    token = _ACTIVE_RUN_TOKENS.get(run_id)
    if token is None:
        raise HTTPException(status_code=404, detail=f"No active run with id {run_id!r} to cancel")
    token.cancel()
    return {"run_id": run_id, "cancel_requested": True}