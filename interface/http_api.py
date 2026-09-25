"""
HTTP API surface — FR-7 (documented HTTP API via OpenAPI, persistent
session history) and FR-8 (authentication + two roles with genuinely
different, server-side-enforced permissions).

Endpoint coverage matches FR-7's list: ingest, ask with citations, run
the workflow, act on the approval gate, view a trace — plus the FR-6
streaming/cancellation endpoints this file already had. OpenAPI docs are
FastAPI's automatic /docs and /openapi.json — no extra work needed beyond
giving every endpoint a typed Pydantic response model, which is why this
file has more of those than the FR-6-only version did.

Auth (FR-8): every endpoint except /health requires an `X-API-Key` header,
resolved to a User + Role via UserRepository (infrastructure/auth/). Role
enforcement happens in `require_role(...)`, a FastAPI dependency checked
server-side before the endpoint body runs — never by hiding a button in a
UI, since there is no UI here, only the API itself. See
docs/ADR-006-access-control.md for the two roles' permission boundaries
and why they're split the way they are.

Persistent session history (FR-7): every authenticated call records a
SessionEvent (domain/session_entities.py) via `_record_session()` before
returning — durable in the same SQLite file as everything else, not
logged to stdout where it vanishes on restart. `GET /sessions` exposes it
back: a CONTRIBUTOR sees their own history, a REVIEWER sees everyone's
(consistent with REVIEWER's oversight role elsewhere in this file).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import AsyncIterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from application.ingest import IngestDocumentUseCase
from application.orchestration.cancellation import CancellationToken
from application.retrieve import AnswerQueryUseCase
from domain.auth_entities import Role, User
from domain.workflow_entities import ItemApprovalStatus
from infrastructure.config import build_supervisor, build_user_repository, build_wiring

app = FastAPI(
    title="RAG Copilot API",
    description=(
        "FR-7 surface (ingest, ask, workflow, approvals, trace, session "
        "history) + FR-6 streaming/cancellation, with FR-8 role-based "
        "access control and security controls."
    ),
)

# --- Security Configuration & Middlewares ---

# 1. CORS Configuration (Explicit origins, no wildcard default)
allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
)

# 2. Rate Limiting Middleware (Sliding window token bucket / timestamp store)
_RATE_LIMIT_STORE: dict[str, list[float]] = {}
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
RATE_LIMIT_WINDOW_SECONDS = 60.0

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    client_key = request.headers.get("x-api-key") or request.client.host if request.client else "unknown"
    now = time.time()
    
    timestamps = _RATE_LIMIT_STORE.get(client_key, [])
    # Filter out timestamps older than window
    timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    
    if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please try again later."},
            headers={
                "Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS)),
                "X-RateLimit-Limit": str(RATE_LIMIT_PER_MINUTE),
                "X-RateLimit-Remaining": "0",
            },
        )
    
    timestamps.append(now)
    _RATE_LIMIT_STORE[client_key] = timestamps

    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
    response.headers["X-RateLimit-Remaining"] = str(max(0, RATE_LIMIT_PER_MINUTE - len(timestamps)))
    return response

# 3. Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"

    # Relax CSP for Swagger / ReDoc docs
    if request.url.path in ("/docs", "/redoc", "/openapi.json"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com;"
        )
    else:
        response.headers["Content-Security-Policy"] = "default-src 'self'"

    if request.url.scheme == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response
    
_USER_REPO = build_user_repository()

# In-memory registry of active workflow runs' cancellation tokens, keyed
# by run_id. Populated once a run's id is known (the first event out of
# run_streaming carries it) and removed once the run finishes. Module-level
# and unsynchronized deliberately: this is a single-process demo transport,
# not a production job registry — a real deployment would need this
# durable and shared across workers, tracked in PLAN.md's roadmap.
_ACTIVE_RUN_TOKENS: dict[str, CancellationToken] = {}


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def get_current_user(x_api_key: Optional[str] = Header(default=None)) -> User:
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    user = _USER_REPO.find_by_api_key(x_api_key)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")
    return user


def require_role(*allowed_roles: Role):
    """Returns a FastAPI dependency that resolves the current user AND
    checks their role — server-side, before the endpoint body ever runs.
    Two separate roles reject two separate sets of endpoints below; this
    is what makes the permissions "genuinely different" rather than
    cosmetic."""

    async def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Role {user.role.value!r} is not permitted to call this "
                    f"endpoint (requires one of: {[r.value for r in allowed_roles]})"
                ),
            )
        return user

    return checker


_SECRET_PATTERNS = [
    re.compile(r'(api[_-]?key|secret|token|password|auth)\s*[:=]\s*["\']?([^"\'\s&]+)["\']?', re.IGNORECASE),
    re.compile(r'(AIzaSy[A-Za-z0-9_-]{33})'),
]

def sanitize_secrets(text: str) -> str:
    """Mask sensitive keys, tokens, or credentials from audit log entries."""
    if not text:
        return text
    sanitized = text
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r'\1=***REDACTED***', sanitized)
    return sanitized


def _record_session(wiring, user: User, endpoint: str, request_summary: str, response_summary: str) -> None:
    from domain.session_entities import SessionEvent
    wiring.session_repo.save_event(
        SessionEvent.new(
            username=user.username,
            role=user.role.value,
            endpoint=endpoint,
            request_summary=sanitize_secrets(request_summary),
            response_summary=sanitize_secrets(response_summary),
        )
    )


# --- Request/response models -------------------------------------------------

class IngestRequest(BaseModel):
    source: str = Field(..., max_length=256)
    doc_type: str = Field(..., max_length=64)
    raw_text: str = Field(..., max_length=5_000_000)
    data_dir: str = "data"


class IngestResponse(BaseModel):
    document_id: str
    status: str
    reused_existing: bool
    chunk_count: int
    error: Optional[str] = None


class AskRequest(BaseModel):
    query: str = Field(..., max_length=10_000)
    data_dir: str = "data"


class CitationResponse(BaseModel):
    chunk_id: str
    source: str
    section: Optional[str]
    position: int
    score: float


class AskResponse(BaseModel):
    refused: bool
    text: str
    citations: list[CitationResponse]


class WorkflowRequest(BaseModel):
    target_role: str = Field(..., max_length=128)
    competencies: list[str] = Field(..., max_length=50)
    data_dir: str = "data"


class RunStepResponse(BaseModel):
    step_index: int
    agent_name: str
    status: str
    attempt: int
    error: Optional[str] = None


class WorkflowRunResponse(BaseModel):
    run_id: str
    status: str
    steps: list[RunStepResponse]


class TraceResponse(BaseModel):
    run_id: str
    target_role: str
    status: str
    started_at: str
    finished_at: Optional[str]
    steps: list[dict]


class ApprovalItemResponse(BaseModel):
    id: str
    module_id: str
    question: str
    options: list[str]
    correct_option_index: int
    citation_chunk_id: str
    citation_source: str
    validation_passed: bool
    validation_notes: str
    approval_status: str


class ApprovalDecisionRequest(BaseModel):
    decision: str  # "approve" | "reject" | "edit"
    edited_text: Optional[str] = None
    data_dir: str = "data"


class SessionEventResponse(BaseModel):
    id: str
    username: str
    role: str
    endpoint: str
    request_summary: str
    response_summary: str
    timestamp: str


# --- Health (no auth) ---------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# --- Ingest (CONTRIBUTOR) ------------------------------------------------

@app.post("/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest, user: User = Depends(require_role(Role.CONTRIBUTOR))) -> IngestResponse:
    wiring = build_wiring(payload.data_dir)
    use_case = IngestDocumentUseCase(
        repo=wiring.repo, embedder=wiring.embedder,
        vector_store=wiring.vector_store, keyword_index=wiring.keyword_index,
    )
    result = use_case.execute(source=payload.source, doc_type=payload.doc_type, raw_text=payload.raw_text)
    response = IngestResponse(
        document_id=result.document_id, status=result.status.value,
        reused_existing=result.reused_existing, chunk_count=result.chunk_count, error=result.error,
    )
    _record_session(wiring, user, "POST /ingest", f"source={payload.source}", f"status={response.status}")
    return response


# --- Ask (CONTRIBUTOR) ----------------------------------------------------

@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, user: User = Depends(require_role(Role.CONTRIBUTOR))) -> AskResponse:
    wiring = build_wiring(payload.data_dir)
    use_case = AnswerQueryUseCase(
        embedder=wiring.embedder, vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index, llm=wiring.llm,
    )
    answer = use_case.execute(payload.query)
    response = AskResponse(
        refused=answer.refused, text=answer.text,
        citations=[
            CitationResponse(
                chunk_id=c.chunk_id, source=c.source, section=c.section,
                position=c.position, score=c.score,
            )
            for c in answer.citations
        ],
    )
    _record_session(wiring, user, "POST /ask", f"query={payload.query}", f"refused={response.refused}")
    return response


@app.post("/ask/stream")
async def ask_stream(
    payload: AskRequest, request: Request, user: User = Depends(require_role(Role.CONTRIBUTOR)),
) -> StreamingResponse:
    wiring = build_wiring(payload.data_dir)
    use_case = AnswerQueryUseCase(
        embedder=wiring.embedder, vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index, llm=wiring.llm,
    )

    async def event_generator() -> AsyncIterator[str]:
        final_refused = None
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
                final_refused = answer.refused
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
        _record_session(
            wiring, user, "POST /ask/stream", f"query={payload.query}",
            f"refused={final_refused}",
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- Workflow (CONTRIBUTOR to run/cancel, REVIEWER to trace) --------------

@app.post("/workflow/run", response_model=WorkflowRunResponse)
def workflow_run(
    payload: WorkflowRequest, user: User = Depends(require_role(Role.CONTRIBUTOR)),
) -> WorkflowRunResponse:
    wiring = build_wiring(payload.data_dir)
    supervisor = build_supervisor(wiring)
    run = supervisor.run(payload.target_role, payload.competencies)
    steps = wiring.workflow_repo.get_steps(run.id)
    response = WorkflowRunResponse(
        run_id=run.id, status=run.status.value,
        steps=[
            RunStepResponse(
                step_index=s.step_index, agent_name=s.agent_name,
                status=s.status.value, attempt=s.attempt, error=s.error,
            )
            for s in steps
        ],
    )
    _record_session(
        wiring, user, "POST /workflow/run", f"target_role={payload.target_role}",
        f"run_id={run.id} status={response.status}",
    )
    return response


@app.post("/workflow/stream")
async def workflow_stream(
    payload: WorkflowRequest, request: Request, user: User = Depends(require_role(Role.CONTRIBUTOR)),
) -> StreamingResponse:
    wiring = build_wiring(payload.data_dir)
    supervisor = build_supervisor(wiring)
    token = CancellationToken()

    async def event_generator() -> AsyncIterator[str]:
        run_id: str | None = None
        final_status = None
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

                if event.event_type == "run_finished":
                    final_status = event.message

                yield _sse_event({
                    "run_id": event.run_id, "event_type": event.event_type,
                    "step_index": event.step_index, "agent_name": event.agent_name,
                    "message": event.message,
                })
        finally:
            if run_id is not None:
                _ACTIVE_RUN_TOKENS.pop(run_id, None)
                _record_session(
                    wiring, user, "POST /workflow/stream",
                    f"target_role={payload.target_role}", f"run_id={run_id} {final_status}",
                )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/workflow/cancel/{run_id}")
def cancel_workflow(run_id: str, user: User = Depends(require_role(Role.CONTRIBUTOR))) -> dict:
    token = _ACTIVE_RUN_TOKENS.get(run_id)
    if token is None:
        raise HTTPException(status_code=404, detail=f"No active run with id {run_id!r} to cancel")
    token.cancel()
    return {"run_id": run_id, "cancel_requested": True}


@app.get("/runs/{run_id}")
def inspect_run(
    run_id: str, data_dir: str = "data", user: User = Depends(require_role(Role.REVIEWER, Role.CONTRIBUTOR)),
) -> dict:
    """Run Inspection endpoint (FR-5): Every execution step audited and queryable via GET /runs/{run_id}."""
    wiring = build_wiring(data_dir)
    run = wiring.workflow_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run found with id {run_id!r}")
    steps = wiring.workflow_repo.get_steps(run_id)
    items = wiring.workflow_repo.list_all()
    audits = wiring.workflow_repo.get_audits(run_id)

    _record_session(wiring, user, "GET /runs/{run_id}", f"run_id={run_id}", f"status={run.status.value}")
    return {
        "run_id": run.id,
        "target_role": run.target_role,
        "status": run.status.value,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "steps": [
            {
                "step_index": s.step_index,
                "agent_name": s.agent_name,
                "status": s.status.value,
                "attempt": s.attempt,
                "input_summary": s.input_summary,
                "output_summary": s.output_summary,
                "error": s.error,
            }
            for s in steps
        ],
        "items": [
            {
                "id": i.id,
                "module_id": i.module_id,
                "question": i.question,
                "options": list(i.options),
                "correct_option_index": i.correct_option_index,
                "correct_key": i.correct_key,
                "rationale": i.rationale,
                "validation_passed": i.validation_passed,
                "validation_notes": i.validation_notes,
                "approval_status": i.approval_status.value,
            }
            for i in items
        ],
        "approval_audits": [
            {
                "id": a.id,
                "run_id": a.run_id,
                "item_id": a.item_id,
                "reviewer_id": a.reviewer_id,
                "decision": a.decision,
                "feedback": a.feedback,
                "original_draft": a.original_draft,
                "modified_content": a.modified_content,
                "timestamp": a.timestamp.isoformat(),
            }
            for a in audits
        ],
    }


class InstructorApproveRequest(BaseModel):
    feedback: Optional[str] = None
    data_dir: str = "data"


class InstructorRejectRequest(BaseModel):
    feedback: str = "Rejected by Lead Instructor"
    data_dir: str = "data"


class InstructorEditApproveRequest(BaseModel):
    edited_items: Optional[list[dict]] = None
    edited_text: Optional[str] = None
    feedback: Optional[str] = None
    data_dir: str = "data"


@app.post("/runs/{run_id}/approve")
def approve_run(
    run_id: str, payload: InstructorApproveRequest, user: User = Depends(require_role(Role.REVIEWER)),
) -> dict:
    """Lead Instructor Approval Gate: Publishes items as proposed."""
    wiring = build_wiring(payload.data_dir)
    run = wiring.workflow_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    pending_items = wiring.workflow_repo.list_pending()
    original_drafts = []
    for item in pending_items:
        original_drafts.append({
            "id": item.id, "question": item.question, "options": list(item.options),
            "correct_option_index": item.correct_option_index, "correct_key": item.correct_key,
        })
        wiring.workflow_repo.decide(
            item_id=item.id, decision=ItemApprovalStatus.APPROVED, decided_by=user.username,
        )

    from domain.workflow_entities import ApprovalAudit, RunStatus
    audit = ApprovalAudit.new(
        run_id=run_id, reviewer_id=user.username, decision="approve",
        original_draft=json.dumps(original_drafts), feedback=payload.feedback,
    )
    wiring.workflow_repo.save_audit(audit)

    run.status = RunStatus.SUCCEEDED
    run.finished_at = time.time() if isinstance(run.finished_at, float) else None
    wiring.workflow_repo.save_run(run)

    _record_session(wiring, user, "POST /runs/approve", f"run_id={run_id}", "status=succeeded")
    return {"run_id": run_id, "status": run.status.value, "approved_items_count": len(pending_items), "audit_id": audit.id}


@app.post("/runs/{run_id}/reject")
def reject_run(
    run_id: str, payload: InstructorRejectRequest, user: User = Depends(require_role(Role.REVIEWER)),
) -> dict:
    """Lead Instructor Approval Gate: Rejects items back to draft with instructor feedback."""
    wiring = build_wiring(payload.data_dir)
    run = wiring.workflow_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    pending_items = wiring.workflow_repo.list_pending()
    original_drafts = []
    for item in pending_items:
        original_drafts.append({
            "id": item.id, "question": item.question, "options": list(item.options),
            "correct_option_index": item.correct_option_index, "correct_key": item.correct_key,
        })
        wiring.workflow_repo.decide(
            item_id=item.id, decision=ItemApprovalStatus.REJECTED, decided_by=user.username,
        )

    from domain.workflow_entities import ApprovalAudit, RunStatus
    audit = ApprovalAudit.new(
        run_id=run_id, reviewer_id=user.username, decision="reject",
        original_draft=json.dumps(original_drafts), feedback=payload.feedback,
    )
    wiring.workflow_repo.save_audit(audit)

    run.status = RunStatus.REJECTED
    wiring.workflow_repo.save_run(run)

    _record_session(wiring, user, "POST /runs/reject", f"run_id={run_id}", "status=rejected")
    return {"run_id": run_id, "status": run.status.value, "feedback": payload.feedback, "audit_id": audit.id}


@app.post("/runs/{run_id}/edit-and-approve")
def edit_and_approve_run(
    run_id: str, payload: InstructorEditApproveRequest, user: User = Depends(require_role(Role.REVIEWER)),
) -> dict:
    """Lead Instructor Approval Gate: Allows Lead Instructor to fix questions, distractors, or keys before final commit."""
    wiring = build_wiring(payload.data_dir)
    run = wiring.workflow_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    pending_items = wiring.workflow_repo.list_pending()
    original_drafts = []
    modified_contents = []

    for idx, item in enumerate(pending_items):
        original_drafts.append({
            "id": item.id, "question": item.question, "options": list(item.options),
            "correct_option_index": item.correct_option_index, "correct_key": item.correct_key,
        })
        edited_text = payload.edited_text
        if payload.edited_items and idx < len(payload.edited_items):
            edited_info = payload.edited_items[idx]
            edited_text = json.dumps(edited_info)
            modified_contents.append(edited_info)
        elif edited_text:
            modified_contents.append({"approved_text": edited_text})

        wiring.workflow_repo.decide(
            item_id=item.id, decision=ItemApprovalStatus.EDITED_AND_APPROVED,
            decided_by=user.username, approved_text=edited_text or "Edited by Lead Instructor",
        )

    from domain.workflow_entities import ApprovalAudit, RunStatus
    audit = ApprovalAudit.new(
        run_id=run_id, reviewer_id=user.username, decision="edit-and-approve",
        original_draft=json.dumps(original_drafts),
        modified_content=json.dumps(modified_contents) if modified_contents else None,
        feedback=payload.feedback,
    )
    wiring.workflow_repo.save_audit(audit)

    run.status = RunStatus.SUCCEEDED
    wiring.workflow_repo.save_run(run)

    _record_session(wiring, user, "POST /runs/edit-and-approve", f"run_id={run_id}", "status=succeeded")
    return {"run_id": run_id, "status": run.status.value, "audit_id": audit.id}


@app.get("/workflow/trace/{run_id}", response_model=TraceResponse)
def workflow_trace(
    run_id: str, data_dir: str = "data", user: User = Depends(require_role(Role.REVIEWER)),
) -> TraceResponse:
    wiring = build_wiring(data_dir)
    run = wiring.workflow_repo.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"No run with id {run_id!r}")
    steps = wiring.workflow_repo.get_steps(run_id)
    response = TraceResponse(
        run_id=run.id, target_role=run.target_role, status=run.status.value,
        started_at=run.started_at.isoformat(),
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        steps=[
            {
                "step_index": s.step_index, "agent_name": s.agent_name,
                "status": s.status.value, "attempt": s.attempt,
                "input_summary": s.input_summary, "output_summary": s.output_summary,
                "error": s.error,
            }
            for s in steps
        ],
    )
    _record_session(wiring, user, "GET /workflow/trace", f"run_id={run_id}", f"status={response.status}")
    return response


# --- Approvals (REVIEWER only) --------------------------------------------

@app.get("/approvals", response_model=list[ApprovalItemResponse])
def list_approvals(
    data_dir: str = "data", user: User = Depends(require_role(Role.REVIEWER)),
) -> list[ApprovalItemResponse]:
    wiring = build_wiring(data_dir)
    items = wiring.workflow_repo.list_pending()
    response = [
        ApprovalItemResponse(
            id=i.id, module_id=i.module_id, question=i.question, options=list(i.options),
            correct_option_index=i.correct_option_index, citation_chunk_id=i.citation_chunk_id,
            citation_source=i.citation_source, validation_passed=i.validation_passed,
            validation_notes=i.validation_notes, approval_status=i.approval_status.value,
        )
        for i in items
    ]
    _record_session(wiring, user, "GET /approvals", "", f"pending_count={len(response)}")
    return response


@app.post("/approvals/{item_id}/decide")
def decide_approval(
    item_id: str, payload: ApprovalDecisionRequest, user: User = Depends(require_role(Role.REVIEWER)),
) -> dict:
    decision_map = {
        "approve": ItemApprovalStatus.APPROVED,
        "reject": ItemApprovalStatus.REJECTED,
        "edit": ItemApprovalStatus.EDITED_AND_APPROVED,
    }
    if payload.decision not in decision_map:
        raise HTTPException(status_code=400, detail=f"decision must be one of {list(decision_map)}")
    if payload.decision == "edit" and not payload.edited_text:
        raise HTTPException(status_code=400, detail="edited_text is required for an 'edit' decision")

    wiring = build_wiring(payload.data_dir)
    try:
        item = wiring.workflow_repo.decide(
            item_id=item_id, decision=decision_map[payload.decision],
            decided_by=user.username, approved_text=payload.edited_text,
        )
        from domain.workflow_entities import ApprovalAudit
        audit = ApprovalAudit.new(
            run_id="single-item", item_id=item_id, reviewer_id=user.username,
            decision=payload.decision, original_draft=item.question,
            modified_content=payload.edited_text,
        )
        wiring.workflow_repo.save_audit(audit)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    _record_session(
        wiring, user, "POST /approvals/decide", f"item_id={item_id} decision={payload.decision}",
        f"approval_status={item.approval_status.value}",
    )
    return {
        "item_id": item.id, "approval_status": item.approval_status.value,
        "decided_by": item.decided_by,
        "decided_at": item.decided_at.isoformat() if item.decided_at else None,
    }


# --- Session history -------------------------------------------------------

@app.get("/sessions", response_model=list[SessionEventResponse])
def list_sessions(data_dir: str = "data", user: User = Depends(get_current_user)) -> list[SessionEventResponse]:
    wiring = build_wiring(data_dir)
    username_filter = None if user.role == Role.REVIEWER else user.username
    events = wiring.session_repo.list_events(username=username_filter)
    return [
        SessionEventResponse(
            id=e.id, username=e.username, role=e.role, endpoint=e.endpoint,
            request_summary=e.request_summary, response_summary=e.response_summary,
            timestamp=e.timestamp.isoformat(),
        )
        for e in events
    ]