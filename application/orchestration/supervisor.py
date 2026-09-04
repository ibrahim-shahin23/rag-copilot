"""
Supervisor orchestrator (FR-5).

Pattern: supervisor, named and justified in docs/ADR-004-orchestration-pattern.md.
Short version: the pipeline here (map -> design -> generate -> validate ->
submit) is a fixed, known sequence, not a plan the system needs to
discover per-request — a supervisor's simpler control flow fits that
better than a general planner-executor.

Mandatory controls, all real mechanisms (not just recorded metadata):
  - max-iteration breaker: MaxIterationsExceeded raised once step_index
    would exceed config.max_iterations, before the step runs.
  - per-step timeout: enforced with a real thread-pool future.result(timeout=...)
    and executor.shutdown(wait=False) — the caller stops waiting and moves
    on immediately once the timeout elapses, rather than merely measuring
    elapsed time after the fact. Python cannot forcibly kill a running
    thread, so a timed-out call keeps executing in the background until it
    naturally returns; "enforced" means this orchestrator never blocks on
    it, not that the call is preemptively terminated. True hard preemption
    would need process-level isolation — out of scope for this slice, and
    worth knowing before relying on this for a genuinely hung/malicious
    tool call rather than a merely slow one.
  - retry with backoff: failed steps retry up to config.max_retries times
    with exponential backoff (time.sleep(base * 2**attempt)).
  - graceful degradation to plain RAG: if the pipeline fails even after
    retries, the orchestrator falls back to a direct AnswerQueryUseCase
    call against the target role rather than returning nothing.
  - every run inspectable step-by-step by run ID: every attempt of every
    step is persisted via RunRepository, success or failure, before the
    orchestrator moves on.
  - approval gate: every generated item (validated or not) is submitted
    via SubmitForApprovalTool — the one write/side-effecting tool — with
    its validation result attached, so a human reviewer sees exactly what
    the automated check found rather than only ever seeing pre-filtered
    "good" items.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from domain.ports import DocumentRepository
from domain.workflow_entities import Run, RunStatus, RunStep, StepStatus
from domain.workflow_ports import RunRepository
from application.agents.curriculum_designer import CurriculumDesignerAgent
from application.agents.item_generator import ItemGeneratorAgent
from application.agents.standards_mapper import StandardsMapperAgent
from application.retrieve import AnswerQueryUseCase
from application.tools import SubmitForApprovalTool
from application.validation import validate_item


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MaxIterationsExceeded(Exception):
    pass


class StepTimeoutError(Exception):
    pass


@dataclass(frozen=True)
class SupervisorConfig:
    max_iterations: int = 20
    step_timeout_seconds: float = 10.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.05


class Supervisor:
    def __init__(
        self,
        standards_mapper: StandardsMapperAgent,
        curriculum_designer: CurriculumDesignerAgent,
        item_generator: ItemGeneratorAgent,
        submit_for_approval: SubmitForApprovalTool,
        run_repo: RunRepository,
        document_repo: DocumentRepository,
        fallback_answer_uc: AnswerQueryUseCase,
        config: SupervisorConfig | None = None,
    ) -> None:
        self._standards_mapper = standards_mapper
        self._curriculum_designer = curriculum_designer
        self._item_generator = item_generator
        self._submit = submit_for_approval
        self._run_repo = run_repo
        self._document_repo = document_repo
        self._fallback_answer_uc = fallback_answer_uc
        self._cfg = config or SupervisorConfig()

    def _run_step(self, run_id: str, agent_name: str, step_index: int, fn, *args, **kwargs):
        if step_index >= self._cfg.max_iterations:
            raise MaxIterationsExceeded(
                f"step {step_index} would exceed max_iterations={self._cfg.max_iterations}"
            )

        input_summary = f"args={args!r} kwargs={kwargs!r}"
        last_error: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 2):  # first try + retries
            executor = cf.ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(fn, *args, **kwargs)
                result = future.result(timeout=self._cfg.step_timeout_seconds)
                self._run_repo.save_step(
                    RunStep.new(
                        run_id=run_id, agent_name=agent_name, step_index=step_index,
                        status=StepStatus.SUCCEEDED, input_summary=input_summary,
                        output_summary=repr(result), attempt=attempt,
                    )
                )
                return result
            except cf.TimeoutError:
                last_error = StepTimeoutError(
                    f"{agent_name} exceeded {self._cfg.step_timeout_seconds}s (attempt {attempt})"
                )
            except Exception as e:  # noqa: BLE001 - genuinely any agent/tool failure
                last_error = e
            finally:
                # shutdown(wait=False) deliberately: Python cannot forcibly
                # kill a running thread, so a timed-out call keeps executing
                # in the background until it naturally returns. What
                # "enforced" means here is that THIS caller stops waiting on
                # it and moves on immediately — using the default wait=True
                # (e.g. via `with ThreadPoolExecutor() as executor:`) defeats
                # the timeout entirely, because __exit__ blocks until the
                # slow call finishes anyway. True hard preemption would need
                # process-level isolation, out of scope for this slice.
                executor.shutdown(wait=False)

            self._run_repo.save_step(
                RunStep.new(
                    run_id=run_id, agent_name=agent_name, step_index=step_index,
                    status=StepStatus.FAILED, input_summary=input_summary,
                    output_summary="", attempt=attempt, error=str(last_error),
                )
            )
            if attempt <= self._cfg.max_retries:
                time.sleep(self._cfg.backoff_base_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        raise last_error

    def _degrade(self, run: Run, run_id: str, step_index: int, reason: Exception) -> None:
        """Graceful degradation to plain RAG: answer a direct question
        about the target role against the corpus instead of returning
        nothing, and record the degradation as its own inspectable step."""
        fallback_query = (
            f"What competencies, standards, or requirements are relevant "
            f"to the role: {run.target_role}?"
        )
        try:
            answer = self._fallback_answer_uc.execute(fallback_query)
            output = answer.text
        except Exception as e:  # noqa: BLE001 - even the fallback can fail; don't crash the run record
            output = f"(fallback also failed: {e})"
        self._run_repo.save_step(
            RunStep.new(
                run_id=run_id, agent_name="supervisor.degrade", step_index=step_index,
                status=StepStatus.DEGRADED, input_summary=f"pipeline failure: {reason}",
                output_summary=output,
            )
        )
        run.status = RunStatus.DEGRADED

    def run(self, target_role: str, competencies: list[str]) -> Run:
        run = Run.new(target_role)
        self._run_repo.save_run(run)
        step_index = 0

        try:
            gap_report = self._run_step(
                run.id, "standards_mapper", step_index,
                self._standards_mapper.execute, target_role, competencies,
            )
            step_index += 1

            outline = self._run_step(
                run.id, "curriculum_designer", step_index,
                self._curriculum_designer.execute, gap_report,
            )
            step_index += 1

            if outline.needs_human_input:
                self._run_repo.save_step(
                    RunStep.new(
                        run_id=run.id, agent_name="curriculum_designer", step_index=step_index,
                        status=StepStatus.FAILED, input_summary=repr(gap_report),
                        output_summary="", error=outline.reason or "needs_human_input",
                    )
                )
                raise RuntimeError(outline.reason or "curriculum designer needs human input")

            all_items = []
            for module in outline.modules:
                items = self._run_step(
                    run.id, "item_generator", step_index,
                    self._item_generator.execute, module,
                )
                step_index += 1
                for item in items:
                    validate_item(item, self._document_repo)
                    self._submit(item)  # the one write/side-effecting tool call
                all_items.extend(items)

            run.status = RunStatus.SUCCEEDED
        except Exception as e:  # noqa: BLE001 - any pipeline failure triggers graceful degradation
            self._degrade(run, run.id, step_index, e)
        finally:
            run.finished_at = _now()
            self._run_repo.save_run(run)

        return run