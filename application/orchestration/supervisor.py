"""
Supervisor orchestrator (FR-5, plus FR-6 streaming/cancellation).

Pattern: Named Supervisor State Machine.

Mandatory controls:
  - max-iteration breaker: MaxIterationsExceeded raised once step_index
    would exceed config.max_iterations (default: 6 iterations), before the step runs.
  - per-step timeout: hard 5.0-second limit per agent step.
  - retry with backoff: failed steps retry up to config.max_retries times
    with exponential backoff.
  - graceful degradation to plain RAG: if pipeline fails, degrade to a basic RAG
    fallback that outputs a standard syllabus outline from cached repository templates.
  - Lead Instructor Approval Gate: pauses workflow in `waiting_approval` state
    once assessment draft items are generated and validated.
  - every run inspectable step-by-step by run ID via GET /runs/{run_id}.
"""
from __future__ import annotations

import concurrent.futures as cf
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterator, Optional, Union

from domain.ports import DocumentRepository
from domain.workflow_entities import (
    ProgressEvent,
    Run,
    RunStatus,
    RunStep,
    StepStatus,
    TargetRoleContract,
)
from domain.workflow_ports import RunRepository
from application.agents.curriculum_designer import CurriculumDesignerAgent
from application.agents.item_generator import ItemGeneratorAgent
from application.agents.standards_mapper import StandardsMapperAgent
from application.orchestration.cancellation import CancellationToken, RunCancelled
from application.retrieve import AnswerQueryUseCase
from application.tools import PublishAssessmentBankTool
from application.validation import validate_item


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MaxIterationsExceeded(Exception):
    pass


class StepTimeoutError(Exception):
    pass


@dataclass(frozen=True)
class SupervisorConfig:
    max_iterations: int = 6
    step_timeout_seconds: float = 5.0
    max_retries: int = 2
    backoff_base_seconds: float = 0.05


class Supervisor:
    def __init__(
        self,
        standards_mapper: StandardsMapperAgent,
        curriculum_designer: CurriculumDesignerAgent,
        item_generator: ItemGeneratorAgent,
        submit_for_approval: PublishAssessmentBankTool,
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

    def _run_step_streaming(
        self,
        run_id: str,
        agent_name: str,
        step_index: int,
        cancellation_token: Optional[CancellationToken],
        fn,
        *args,
        **kwargs,
    ) -> Iterator[ProgressEvent]:
        if cancellation_token is not None and cancellation_token.is_cancelled():
            raise RunCancelled(f"cancelled before step {step_index} ({agent_name})")

        if step_index >= self._cfg.max_iterations:
            raise MaxIterationsExceeded(
                f"step {step_index} would exceed max_iterations={self._cfg.max_iterations}"
            )

        yield ProgressEvent(
            run_id=run_id, event_type="step_started", step_index=step_index,
            agent_name=agent_name, message=f"{agent_name} starting",
        )

        input_summary = f"args={args!r} kwargs={kwargs!r}"
        last_error: Exception | None = None
        for attempt in range(1, self._cfg.max_retries + 2):
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
                yield ProgressEvent(
                    run_id=run_id, event_type="step_succeeded", step_index=step_index,
                    agent_name=agent_name, message=f"{agent_name} succeeded (attempt {attempt})",
                )
                return result
            except cf.TimeoutError:
                last_error = StepTimeoutError(
                    f"{agent_name} exceeded {self._cfg.step_timeout_seconds}s (attempt {attempt})"
                )
            except Exception as e:
                last_error = e
            finally:
                executor.shutdown(wait=False)

            self._run_repo.save_step(
                RunStep.new(
                    run_id=run_id, agent_name=agent_name, step_index=step_index,
                    status=StepStatus.FAILED, input_summary=input_summary,
                    output_summary="", attempt=attempt, error=str(last_error),
                )
            )
            yield ProgressEvent(
                run_id=run_id, event_type="step_failed", step_index=step_index,
                agent_name=agent_name, message=f"attempt {attempt} failed: {last_error}",
            )
            if attempt <= self._cfg.max_retries:
                yield ProgressEvent(
                    run_id=run_id, event_type="step_retrying", step_index=step_index,
                    agent_name=agent_name, message=f"retrying (attempt {attempt + 1})",
                )
                time.sleep(self._cfg.backoff_base_seconds * (2 ** (attempt - 1)))

        assert last_error is not None
        raise last_error

    def _degrade(self, run: Run, run_id: str, step_index: int, reason: Exception) -> str:
        """Graceful degradation: output a standard syllabus outline from cached templates."""
        fallback_query = (
            f"Generate standard course syllabus outline and core competencies "
            f"for role: {run.target_role}"
        )
        try:
            answer = self._fallback_answer_uc.execute(fallback_query)
            output = answer.text
        except Exception as e:
            output = f"(fallback also failed: {e})"
        self._run_repo.save_step(
            RunStep.new(
                run_id=run_id, agent_name="supervisor.degrade", step_index=step_index,
                status=StepStatus.DEGRADED, input_summary=f"pipeline failure: {reason}",
                output_summary=output,
            )
        )
        run.status = RunStatus.DEGRADED
        return output

    def run_streaming(
        self,
        target_role: Union[TargetRoleContract, str],
        competencies: Optional[list[str]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> Iterator[ProgressEvent]:
        role_name = target_role.role_name if isinstance(target_role, TargetRoleContract) else target_role
        target_contract = target_role if isinstance(target_role, TargetRoleContract) else TargetRoleContract(
            role_name=target_role, existing_prerequisites=competencies or []
        )

        run = Run.new(role_name)
        self._run_repo.save_run(run)
        yield ProgressEvent(
            run_id=run.id, event_type="run_started", step_index=None,
            agent_name=None, message=f"run started for role={role_name!r}",
        )
        step_index = 0

        try:
            gap_report = yield from self._run_step_streaming(
                run.id, "standards_mapper", step_index, cancellation_token,
                self._standards_mapper.execute, target_contract, competencies,
            )
            step_index += 1

            outline = yield from self._run_step_streaming(
                run.id, "curriculum_designer", step_index, cancellation_token,
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

            for module in outline.modules:
                items = yield from self._run_step_streaming(
                    run.id, "item_generator", step_index, cancellation_token,
                    self._item_generator.execute, module,
                )
                step_index += 1
                for item in items:
                    validate_item(item, self._document_repo)
                    self._submit(item)

            # Lead Instructor Approval Gate: set status to waiting_approval
            run.status = RunStatus.WAITING_APPROVAL
            yield ProgressEvent(
                run_id=run.id, event_type="waiting_approval", step_index=step_index,
                agent_name="orchestrator", message="Workflow paused in waiting_approval state for Lead Instructor review.",
            )

        except RunCancelled as e:
            run.status = RunStatus.CANCELLED
            yield ProgressEvent(
                run_id=run.id, event_type="run_cancelled", step_index=step_index,
                agent_name=None, message=str(e),
            )
        except Exception as e:
            output = self._degrade(run, run.id, step_index, e)
            yield ProgressEvent(
                run_id=run.id, event_type="degraded", step_index=step_index,
                agent_name="supervisor.degrade", message=output,
            )
        finally:
            run.finished_at = _now()
            self._run_repo.save_run(run)
            yield ProgressEvent(
                run_id=run.id, event_type="run_finished", step_index=step_index,
                agent_name=None, message=f"status={run.status.value}",
            )

    def run(self, target_role: Union[TargetRoleContract, str], competencies: Optional[list[str]] = None) -> Run:
        run_id: Optional[str] = None
        for event in self.run_streaming(target_role, competencies):
            if run_id is None:
                run_id = event.run_id
        assert run_id is not None
        run = self._run_repo.get_run(run_id)
        assert run is not None
        return run