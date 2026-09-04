"""
Ports for the multi-agent workflow's persistence needs: run/step tracking
(FR-5's "every run inspectable step-by-step by run ID") and the approval
gate (FR-4's one write/side-effecting tool lands here, and nothing reads
an item as usable until a human decision is recorded against it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from domain.workflow_entities import AssessmentItem, ItemApprovalStatus, Run, RunStep


class RunRepository(ABC):
    @abstractmethod
    def save_run(self, run: Run) -> None:
        ...

    @abstractmethod
    def get_run(self, run_id: str) -> Optional[Run]:
        ...

    @abstractmethod
    def save_step(self, step: RunStep) -> None:
        ...

    @abstractmethod
    def get_steps(self, run_id: str) -> list[RunStep]:
        ...


class ApprovalGateRepository(ABC):
    """The target of the one write/side-effecting tool (submit_for_approval).
    An item written here is PENDING and inert — no downstream effect exists
    for a pending item; only decide() changes that, and every decision is
    recorded (decided_by, decided_at), which is the audit trail FR-5 asks
    the approval gate to keep."""

    @abstractmethod
    def submit(self, item: AssessmentItem) -> None:
        ...

    @abstractmethod
    def get(self, item_id: str) -> Optional[AssessmentItem]:
        ...

    @abstractmethod
    def list_pending(self) -> list[AssessmentItem]:
        ...

    @abstractmethod
    def list_all(self) -> list[AssessmentItem]:
        ...

    @abstractmethod
    def decide(
        self,
        item_id: str,
        decision: ItemApprovalStatus,
        decided_by: str,
        approved_text: Optional[str] = None,
    ) -> AssessmentItem:
        ...