"""
Domain entities for the multi-agent workflow (FR-4/FR-5). Same rule as
domain/entities.py: plain dataclasses, zero dependency on any LLM SDK,
vector-store SDK, web framework, or the agent/orchestration code itself.
These are the typed contracts FR-4 requires agents to communicate through
— never free-form text between agents.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --- Standards Mapper output -------------------------------------------------

@dataclass(frozen=True)
class CompetencyGap:
    name: str
    description: str
    citation_chunk_ids: tuple[str, ...]  # empty tuple if unmapped
    matched: bool  # False = explicitly flagged unmapped, never silently dropped


@dataclass(frozen=True)
class CompetencyGapReport:
    target_role: str
    gaps: tuple[CompetencyGap, ...]


# --- Curriculum Designer output ----------------------------------------------

@dataclass(frozen=True)
class Module:
    id: str
    title: str
    gap_names: tuple[str, ...]
    order: int


@dataclass(frozen=True)
class ModuleOutline:
    target_role: str
    modules: tuple[Module, ...]
    needs_human_input: bool = False
    reason: Optional[str] = None


# --- Item Generator output ---------------------------------------------------

class ItemApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED_AND_APPROVED = "edited_and_approved"


@dataclass
class AssessmentItem:
    id: str
    module_id: str
    question: str
    options: tuple[str, ...]
    correct_option_index: int
    citation_chunk_id: str
    citation_source: str
    validation_passed: bool
    validation_notes: str
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
    ) -> "AssessmentItem":
        return AssessmentItem(
            id=str(uuid.uuid4()),
            module_id=module_id,
            question=question,
            options=tuple(options),
            correct_option_index=correct_option_index,
            citation_chunk_id=citation_chunk_id,
            citation_source=citation_source,
            validation_passed=False,
            validation_notes="not yet validated",
        )


# --- Run / step tracking (FR-9-adjacent: "every run inspectable step-by-step
# by run ID", per FR-5) -------------------------------------------------------

class StepStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"  # graceful degradation to plain RAG was invoked


@dataclass
class RunStep:
    id: str
    run_id: str
    agent_name: str
    step_index: int
    status: StepStatus
    input_summary: str
    output_summary: str
    attempt: int = 1
    started_at: datetime = field(default_factory=_now)
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
    ) -> "RunStep":
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
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass
class Run:
    id: str
    target_role: str
    status: RunStatus = RunStatus.RUNNING
    started_at: datetime = field(default_factory=_now)
    finished_at: Optional[datetime] = None

    @staticmethod
    def new(target_role: str) -> "Run":
        return Run(id=str(uuid.uuid4()), target_role=target_role)