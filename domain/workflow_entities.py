"""
Domain entities for the multi-agent workflow (FR-4/FR-5). Same rule as
domain/entities.py: zero dependency on any LLM SDK, vector-store SDK,
or web framework.
These are the typed Pydantic contracts FR-4 requires agents to communicate through
— strictly no free-form text handoffs between agents.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid
from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Input Contract: Target Role ---

class TargetRoleContract(BaseModel):
    role_name: str
    target_seniority: str = "Mid"
    existing_prerequisites: list[str] = Field(default_factory=list)


# --- Standards Mapper output: Competency Gaps Contract -----------------------

class CompetencyGap(BaseModel):
    name: str
    description: str
    citation_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    matched: bool = True
    framework_standard: Optional[str] = None


class CompetencyGapsContract(BaseModel):
    target_role: str
    mapped_standards: list[dict] = Field(default_factory=list)
    gaps: tuple[CompetencyGap, ...] = Field(default_factory=tuple)
    gap_areas: list[str] = Field(default_factory=list)

    @field_validator("target_role", mode="before")
    @classmethod
    def _coerce_target_role(cls, v):
        if hasattr(v, "role_name"):
            return v.role_name
        return str(v)


# Backwards compatibility alias
CompetencyGapReport = CompetencyGapsContract


# --- Curriculum Designer output: Module Outline Contract --------------------

class Module(BaseModel):
    id: str
    title: str
    gap_names: tuple[str, ...] = Field(default_factory=tuple)
    order: int = 0
    units: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)


class ModuleOutlineContract(BaseModel):
    target_role: str
    modules: tuple[Module, ...] = Field(default_factory=tuple)
    needs_human_input: bool = False
    reason: Optional[str] = None

    @field_validator("target_role", mode="before")
    @classmethod
    def _coerce_target_role(cls, v):
        if hasattr(v, "role_name"):
            return v.role_name
        return str(v)



# Backwards compatibility alias
ModuleOutline = ModuleOutlineContract


# --- Item Generator output: Assessment Draft Contract ----------------------

class ItemApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED_AND_APPROVED = "edited_and_approved"


class AssessmentDraftItemContract(BaseModel):
    id: str
    module_id: str
    question: str
    options: tuple[str, ...]
    correct_option_index: int
    correct_key: str = ""
    rationale: str = ""
    citation_chunk_id: str
    citation_source: str
    validation_passed: bool = False
    validation_notes: str = "not yet validated"
    approval_status: ItemApprovalStatus = ItemApprovalStatus.PENDING
    approved_text: Optional[str] = None  # populated only on edit-and-approve
    decided_by: Optional[str] = None
    decided_at: Optional[datetime] = None

    @staticmethod
    def new(
        module_id: str,
        question: str,
        options: list[str],
        correct_option_index: int,
        citation_chunk_id: str,
        citation_source: str,
        rationale: str = "",
    ) -> AssessmentDraftItemContract:
        c_key = options[correct_option_index] if 0 <= correct_option_index < len(options) else ""
        return AssessmentDraftItemContract(
            id=str(uuid.uuid4()),
            module_id=module_id,
            question=question,
            options=tuple(options),
            correct_option_index=correct_option_index,
            correct_key=c_key,
            rationale=rationale or f"Derived from citation source {citation_source}",
            citation_chunk_id=citation_chunk_id,
            citation_source=citation_source,
            validation_passed=False,
            validation_notes="not yet validated",
        )


# Backwards compatibility alias
AssessmentItem = AssessmentDraftItemContract


class AssessmentDraftContract(BaseModel):
    module_id: str
    items: list[AssessmentDraftItemContract] = Field(default_factory=list)


# --- Approval Audit Record --------------------------------------------------

class ApprovalAudit(BaseModel):
    id: str
    run_id: str
    item_id: Optional[str] = None
    reviewer_id: str
    decision: str  # approve | reject | edit-and-approve
    feedback: Optional[str] = None
    original_draft: str
    modified_content: Optional[str] = None
    timestamp: datetime = Field(default_factory=_now)

    @staticmethod
    def new(
        run_id: str,
        reviewer_id: str,
        decision: str,
        original_draft: str,
        item_id: Optional[str] = None,
        feedback: Optional[str] = None,
        modified_content: Optional[str] = None,
    ) -> ApprovalAudit:
        return ApprovalAudit(
            id=str(uuid.uuid4()),
            run_id=run_id,
            item_id=item_id,
            reviewer_id=reviewer_id,
            decision=decision,
            feedback=feedback,
            original_draft=original_draft,
            modified_content=modified_content,
            timestamp=_now(),
        )


# --- Run / step tracking -----------------------------------------------------

class StepStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"  # graceful degradation to plain RAG was invoked


class RunStep(BaseModel):
    id: str
    run_id: str
    agent_name: str
    step_index: int
    status: StepStatus
    input_summary: str
    output_summary: str
    attempt: int = 1
    started_at: datetime = Field(default_factory=_now)
    error: Optional[str] = None

    @staticmethod
    def new(
        run_id: str,
        agent_name: str,
        step_index: int,
        status: StepStatus,
        input_summary: str,
        output_summary: str,
        attempt: int = 1,
        error: Optional[str] = None,
    ) -> RunStep:
        return RunStep(
            id=str(uuid.uuid4()),
            run_id=run_id,
            agent_name=agent_name,
            step_index=step_index,
            status=status,
            input_summary=input_summary[:500],
            output_summary=output_summary[:500],
            attempt=attempt,
            error=error,
        )


class RunStatus(str, Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    FAILED = "failed"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"


class Run(BaseModel):
    id: str
    target_role: str
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = Field(default_factory=_now)
    finished_at: Optional[datetime] = None

    @staticmethod
    def new(target_role: str) -> Run:
        return Run(id=str(uuid.uuid4()), target_role=target_role)


# --- Progress events ---

class ProgressEvent(BaseModel):
    run_id: str
    event_type: str  # run_started | step_started | step_failed | step_retrying | step_succeeded | waiting_approval | run_cancelled | degraded | run_finished
    step_index: Optional[int]
    agent_name: Optional[str]
    message: str
    timestamp: datetime = Field(default_factory=_now)