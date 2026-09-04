"""
SQLite-backed adapters for RunRepository and ApprovalGateRepository
(domain/workflow_ports.py). Uses the same SQLite file as
SqliteDocumentRepository (one relational store per FR-1's requirement,
different tables) rather than a second database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

from domain.workflow_entities import (
    AssessmentItem,
    ItemApprovalStatus,
    Run,
    RunStatus,
    RunStep,
    StepStatus,
)
from domain.workflow_ports import ApprovalGateRepository, RunRepository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    target_role TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS run_steps (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(id),
    agent_name TEXT NOT NULL,
    step_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_summary TEXT NOT NULL,
    output_summary TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_steps_run_id ON run_steps(run_id);

CREATE TABLE IF NOT EXISTS assessment_items (
    id TEXT PRIMARY KEY,
    module_id TEXT NOT NULL,
    question TEXT NOT NULL,
    options TEXT NOT NULL,
    correct_option_index INTEGER NOT NULL,
    citation_chunk_id TEXT NOT NULL,
    citation_source TEXT NOT NULL,
    validation_passed INTEGER NOT NULL,
    validation_notes TEXT NOT NULL,
    approval_status TEXT NOT NULL,
    approved_text TEXT,
    decided_by TEXT,
    decided_at TEXT
);
"""


class SqliteWorkflowRepository(RunRepository, ApprovalGateRepository):
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # --- RunRepository ---

    def save_run(self, run: Run) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (id, target_role, status, started_at, finished_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET status=excluded.status, finished_at=excluded.finished_at
            """,
            (
                run.id, run.target_role, run.status.value,
                run.started_at.isoformat(),
                run.finished_at.isoformat() if run.finished_at else None,
            ),
        )
        self._conn.commit()

    def get_run(self, run_id: str) -> Optional[Run]:
        row = self._conn.execute(
            "SELECT id, target_role, status, started_at, finished_at FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        from datetime import datetime
        return Run(
            id=row[0], target_role=row[1], status=RunStatus(row[2]),
            started_at=datetime.fromisoformat(row[3]),
            finished_at=datetime.fromisoformat(row[4]) if row[4] else None,
        )

    def save_step(self, step: RunStep) -> None:
        self._conn.execute(
            """
            INSERT INTO run_steps
            (id, run_id, agent_name, step_index, status, input_summary,
             output_summary, attempt, started_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                step.id, step.run_id, step.agent_name, step.step_index,
                step.status.value, step.input_summary, step.output_summary,
                step.attempt, step.started_at.isoformat(), step.error,
            ),
        )
        self._conn.commit()

    def get_steps(self, run_id: str) -> list[RunStep]:
        rows = self._conn.execute(
            "SELECT id, run_id, agent_name, step_index, status, input_summary, "
            "output_summary, attempt, started_at, error FROM run_steps "
            "WHERE run_id = ? ORDER BY step_index, attempt",
            (run_id,),
        ).fetchall()
        from datetime import datetime
        return [
            RunStep(
                id=r[0], run_id=r[1], agent_name=r[2], step_index=r[3],
                status=StepStatus(r[4]), input_summary=r[5], output_summary=r[6],
                attempt=r[7], started_at=datetime.fromisoformat(r[8]), error=r[9],
            )
            for r in rows
        ]

    # --- ApprovalGateRepository ---

    def submit(self, item: AssessmentItem) -> None:
        import json
        self._conn.execute(
            """
            INSERT INTO assessment_items
            (id, module_id, question, options, correct_option_index,
             citation_chunk_id, citation_source, validation_passed,
             validation_notes, approval_status, approved_text, decided_by, decided_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                validation_passed=excluded.validation_passed,
                validation_notes=excluded.validation_notes
            """,
            (
                item.id, item.module_id, item.question, json.dumps(list(item.options)),
                item.correct_option_index, item.citation_chunk_id, item.citation_source,
                int(item.validation_passed), item.validation_notes,
                item.approval_status.value, item.approved_text,
                item.decided_by, item.decided_at.isoformat() if item.decided_at else None,
            ),
        )
        self._conn.commit()

    def _row_to_item(self, row) -> AssessmentItem:
        import json
        from datetime import datetime
        return AssessmentItem(
            id=row[0], module_id=row[1], question=row[2],
            options=tuple(json.loads(row[3])), correct_option_index=row[4],
            citation_chunk_id=row[5], citation_source=row[6],
            validation_passed=bool(row[7]), validation_notes=row[8],
            approval_status=ItemApprovalStatus(row[9]), approved_text=row[10],
            decided_by=row[11], decided_at=datetime.fromisoformat(row[12]) if row[12] else None,
        )

    def get(self, item_id: str) -> Optional[AssessmentItem]:
        row = self._conn.execute(
            "SELECT id, module_id, question, options, correct_option_index, "
            "citation_chunk_id, citation_source, validation_passed, validation_notes, "
            "approval_status, approved_text, decided_by, decided_at "
            "FROM assessment_items WHERE id = ?",
            (item_id,),
        ).fetchone()
        return self._row_to_item(row) if row else None

    def list_pending(self) -> list[AssessmentItem]:
        rows = self._conn.execute(
            "SELECT id, module_id, question, options, correct_option_index, "
            "citation_chunk_id, citation_source, validation_passed, validation_notes, "
            "approval_status, approved_text, decided_by, decided_at "
            "FROM assessment_items WHERE approval_status = ?",
            (ItemApprovalStatus.PENDING.value,),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def list_all(self) -> list[AssessmentItem]:
        rows = self._conn.execute(
            "SELECT id, module_id, question, options, correct_option_index, "
            "citation_chunk_id, citation_source, validation_passed, validation_notes, "
            "approval_status, approved_text, decided_by, decided_at "
            "FROM assessment_items"
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def decide(
        self,
        item_id: str,
        decision: ItemApprovalStatus,
        decided_by: str,
        approved_text: Optional[str] = None,
    ) -> AssessmentItem:
        from datetime import datetime, timezone
        item = self.get(item_id)
        if item is None:
            raise ValueError(f"No assessment item with id {item_id!r}")
        item.approval_status = decision
        item.decided_by = decided_by
        item.decided_at = datetime.now(timezone.utc)
        if decision == ItemApprovalStatus.EDITED_AND_APPROVED:
            item.approved_text = approved_text
        self._conn.execute(
            """
            UPDATE assessment_items
            SET approval_status = ?, approved_text = ?, decided_by = ?, decided_at = ?
            WHERE id = ?
            """,
            (item.approval_status.value, item.approved_text, item.decided_by,
             item.decided_at.isoformat(), item.id),
        )
        self._conn.commit()
        return item