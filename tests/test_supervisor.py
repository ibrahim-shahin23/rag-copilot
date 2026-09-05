import time

import pytest

from domain.entities import Chunk, ChunkMetadata
from domain.workflow_entities import (
    CompetencyGap,
    CompetencyGapReport,
    ItemApprovalStatus,
    Module,
    ModuleOutline,
    Run,
    RunStatus,
    RunStep,
)
from domain.workflow_ports import ApprovalGateRepository, RunRepository
from application.orchestration.cancellation import CancellationToken
from application.orchestration.supervisor import Supervisor, SupervisorConfig
from application.tools import SubmitForApprovalTool


class InMemoryWorkflowRepo(RunRepository, ApprovalGateRepository):
    """Fast in-memory fake — avoids disk I/O so the timeout/backoff tests
    (which use real, tiny sleeps) stay quick."""

    def __init__(self):
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, list[RunStep]] = {}
        self.items: dict[str, "AssessmentItem"] = {}

    def save_run(self, run):
        self.runs[run.id] = run

    def get_run(self, run_id):
        return self.runs.get(run_id)

    def save_step(self, step):
        self.steps.setdefault(step.run_id, []).append(step)

    def get_steps(self, run_id):
        return list(self.steps.get(run_id, []))

    def submit(self, item):
        self.items[item.id] = item

    def get(self, item_id):
        return self.items.get(item_id)

    def list_pending(self):
        return [i for i in self.items.values() if i.approval_status == ItemApprovalStatus.PENDING]

    def list_all(self):
        return list(self.items.values())

    def decide(self, item_id, decision, decided_by, approved_text=None):
        item = self.items[item_id]
        item.approval_status = decision
        item.decided_by = decided_by
        if approved_text:
            item.approved_text = approved_text
        return item


class _FakeAgent:
    """Wraps a plain function as an object with .execute, since Supervisor
    only ever calls .execute(...) — matches how the real agents are shaped
    without needing the real agent classes in these orchestration tests."""

    def __init__(self, fn):
        self.execute = fn


class _FakeDocumentRepo:
    def __init__(self, chunks: dict):
        self._chunks = chunks

    def find_chunk_by_id(self, chunk_id):
        return self._chunks.get(chunk_id)


class _FakeAnswer:
    def __init__(self, text):
        self.text = text


class _FakeAnswerUseCase:
    def __init__(self):
        self.calls = []

    def execute(self, query):
        self.calls.append(query)
        return _FakeAnswer(f"fallback answer for: {query}")


def _chunk(cid="c1", text="FR-1 requires at least 2 things."):
    return Chunk(
        id=cid, document_id="d1", text=text,
        metadata=ChunkMetadata(source="spec.md", section=None, position=0,
                                char_start=0, char_end=len(text), version="1"),
    )


def _happy_path_agents():
    gap_report = CompetencyGapReport(
        target_role="Role",
        gaps=(CompetencyGap(name="x", description="d", citation_chunk_ids=("c1",), matched=True),),
    )
    outline = ModuleOutline(
        target_role="Role",
        modules=(Module(id="m1", title="Module: x", gap_names=("x",), order=0),),
    )

    def gen_items(module):
        from domain.workflow_entities import AssessmentItem
        return [
            AssessmentItem.new(
                module_id=module.id, question="Fill in: ___",
                options=["2", "3"], correct_option_index=0,
                citation_chunk_id="c1", citation_source="spec.md",
            )
        ]

    standards_mapper = _FakeAgent(lambda target_role, competencies: gap_report)
    curriculum_designer = _FakeAgent(lambda gr: outline)
    item_generator = _FakeAgent(gen_items)
    return standards_mapper, curriculum_designer, item_generator


def _build_supervisor(standards_mapper, curriculum_designer, item_generator, config=None, chunks=None):
    repo = InMemoryWorkflowRepo()
    submit = SubmitForApprovalTool(repo)
    doc_repo = _FakeDocumentRepo(chunks or {"c1": _chunk()})
    fallback_uc = _FakeAnswerUseCase()
    supervisor = Supervisor(
        standards_mapper=standards_mapper,
        curriculum_designer=curriculum_designer,
        item_generator=item_generator,
        submit_for_approval=submit,
        run_repo=repo,
        document_repo=doc_repo,
        fallback_answer_uc=fallback_uc,
        config=config or SupervisorConfig(max_retries=0, step_timeout_seconds=5.0, backoff_base_seconds=0.001),
    )
    return supervisor, repo, fallback_uc


# --- happy path ---

def test_happy_path_submits_validated_items_and_succeeds():
    sm, cd, ig = _happy_path_agents()
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert run.status == RunStatus.SUCCEEDED
    assert fallback_uc.calls == []  # no degradation needed
    pending = repo.list_pending()
    assert len(pending) == 1
    assert pending[0].validation_passed is True  # "2" really is in the cited chunk text
    assert pending[0].citation_chunk_id == "c1"


def test_run_is_inspectable_step_by_step_by_run_id():
    sm, cd, ig = _happy_path_agents()
    supervisor, repo, _ = _build_supervisor(sm, cd, ig)
    run = supervisor.run(target_role="Role", competencies=["x"])

    fetched_run = repo.get_run(run.id)
    assert fetched_run is not None
    assert fetched_run.status == RunStatus.SUCCEEDED

    steps = repo.get_steps(run.id)
    agent_names = [s.agent_name for s in steps]
    assert "standards_mapper" in agent_names
    assert "curriculum_designer" in agent_names
    assert "item_generator" in agent_names
    assert all(s.run_id == run.id for s in steps)


# --- max-iteration breaker ---

def test_max_iterations_breaker_triggers_degradation():
    sm, cd, ig = _happy_path_agents()
    config = SupervisorConfig(max_iterations=1, max_retries=0, step_timeout_seconds=5.0)
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig, config=config)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert run.status == RunStatus.DEGRADED
    assert len(fallback_uc.calls) == 1  # graceful degradation actually invoked plain RAG
    steps = repo.get_steps(run.id)
    assert any(s.agent_name == "supervisor.degrade" for s in steps)
    # step 0 (standards_mapper) should have succeeded before the breaker fired at step 1
    assert any(s.agent_name == "standards_mapper" and s.status.value == "succeeded" for s in steps)


# --- per-step timeout (real, preemptive) ---

def test_step_timeout_is_actually_enforced_not_just_measured():
    def slow_fn(target_role, competencies):
        time.sleep(0.3)
        return CompetencyGapReport(target_role=target_role, gaps=())

    sm = _FakeAgent(slow_fn)
    cd = _FakeAgent(lambda gr: ModuleOutline(target_role="Role", modules=()))
    ig = _FakeAgent(lambda module: [])
    config = SupervisorConfig(max_retries=0, step_timeout_seconds=0.05, backoff_base_seconds=0.001)
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig, config=config)

    started = time.monotonic()
    run = supervisor.run(target_role="Role", competencies=["x"])
    elapsed = time.monotonic() - started

    assert run.status == RunStatus.DEGRADED
    steps = repo.get_steps(run.id)
    timeout_steps = [s for s in steps if s.agent_name == "standards_mapper"]
    assert len(timeout_steps) == 1
    assert timeout_steps[0].status.value == "failed"
    assert "exceeded" in timeout_steps[0].error.lower()
    # The whole run should return promptly despite slow_fn sleeping 0.3s,
    # because future.result(timeout=...) + shutdown(wait=False) stop this
    # caller from blocking on it — this is what makes the fix real (before
    # the fix, `with ThreadPoolExecutor()` blocked on __exit__ until the
    # slow call finished anyway, silently defeating the timeout).
    assert elapsed < 0.2


# --- retry with backoff ---

def test_retries_then_succeeds_within_max_retries():
    attempts = {"n": 0}

    def flaky_fn(target_role, competencies):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ValueError("transient failure")
        return CompetencyGapReport(target_role=target_role, gaps=())

    sm = _FakeAgent(flaky_fn)
    cd = _FakeAgent(lambda gr: ModuleOutline(target_role="Role", modules=()))
    ig = _FakeAgent(lambda module: [])
    config = SupervisorConfig(max_retries=2, step_timeout_seconds=5.0, backoff_base_seconds=0.001)
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig, config=config)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert attempts["n"] == 3
    assert fallback_uc.calls == []  # recovered without ever needing to degrade
    steps = [s for s in repo.get_steps(run.id) if s.agent_name == "standards_mapper"]
    assert [s.status.value for s in steps] == ["failed", "failed", "succeeded"]
    assert [s.attempt for s in steps] == [1, 2, 3]


def test_exhausting_retries_triggers_graceful_degradation():
    def always_fails(target_role, competencies):
        raise ValueError("permanent failure")

    sm = _FakeAgent(always_fails)
    cd = _FakeAgent(lambda gr: ModuleOutline(target_role="Role", modules=()))
    ig = _FakeAgent(lambda module: [])
    config = SupervisorConfig(max_retries=2, step_timeout_seconds=5.0, backoff_base_seconds=0.001)
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig, config=config)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert run.status == RunStatus.DEGRADED
    assert len(fallback_uc.calls) == 1
    assert "Role" in fallback_uc.calls[0]
    steps = [s for s in repo.get_steps(run.id) if s.agent_name == "standards_mapper"]
    assert len(steps) == 3  # 1 initial + 2 retries, all failed
    assert all(s.status.value == "failed" for s in steps)


# --- needs_human_input path also degrades gracefully ---

def test_needs_human_input_from_curriculum_designer_triggers_degradation():
    sm = _FakeAgent(lambda target_role, competencies: CompetencyGapReport(target_role=target_role, gaps=()))
    cd = _FakeAgent(
        lambda gr: ModuleOutline(
            target_role=gr.target_role, modules=(), needs_human_input=True, reason="nothing matched"
        )
    )
    ig = _FakeAgent(lambda module: [])
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert run.status == RunStatus.DEGRADED
    assert len(fallback_uc.calls) == 1


# --- approval gate: items always go through validation before submission ---

def test_invalid_item_is_still_submitted_but_flagged_not_silently_dropped():
    """An item whose 'correct' answer isn't actually in the cited chunk
    should still reach the approval gate (so a human sees it and can
    reject it) — validation flags it, it doesn't vanish."""
    from domain.workflow_entities import AssessmentItem

    def gen_bad_item(module):
        return [
            AssessmentItem.new(
                module_id=module.id, question="Fill in: ___",
                options=["999", "3"], correct_option_index=0,  # 999 is NOT in the chunk text
                citation_chunk_id="c1", citation_source="spec.md",
            )
        ]

    sm = _FakeAgent(
        lambda target_role, competencies: CompetencyGapReport(
            target_role=target_role,
            gaps=(CompetencyGap(name="x", description="d", citation_chunk_ids=("c1",), matched=True),),
        )
    )
    cd = _FakeAgent(
        lambda gr: ModuleOutline(
            target_role=gr.target_role,
            modules=(Module(id="m1", title="x", gap_names=("x",), order=0),),
        )
    )
    ig = _FakeAgent(gen_bad_item)
    supervisor, repo, _ = _build_supervisor(sm, cd, ig)

    run = supervisor.run(target_role="Role", competencies=["x"])

    assert run.status == RunStatus.SUCCEEDED  # pipeline itself didn't fail
    pending = repo.list_pending()
    assert len(pending) == 1
    assert pending[0].validation_passed is False
    assert "not found verbatim" in pending[0].validation_notes


# --- FR-6: streaming progress events ---

def test_run_streaming_yields_events_in_order_for_happy_path():
    sm, cd, ig = _happy_path_agents()
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)

    events = list(supervisor.run_streaming(target_role="Role", competencies=["x"]))

    event_types = [e.event_type for e in events]
    assert event_types[0] == "run_started"
    assert event_types[-1] == "run_finished"
    assert "step_started" in event_types
    assert "step_succeeded" in event_types
    # every event carries the same run_id — a client can distinguish
    # concurrent runs by this field alone
    assert len({e.run_id for e in events}) == 1
    # step_started for an agent must precede its step_succeeded
    sm_started = event_types.index("step_started")
    sm_succeeded_indices = [i for i, t in enumerate(event_types) if t == "step_succeeded"]
    assert sm_succeeded_indices[0] > sm_started


def test_run_streaming_emits_retry_events_matching_run_behavior():
    attempts = {"n": 0}

    def flaky_fn(target_role, competencies):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise ValueError("transient")
        return CompetencyGapReport(target_role=target_role, gaps=())

    sm = _FakeAgent(flaky_fn)
    cd = _FakeAgent(lambda gr: ModuleOutline(target_role="Role", modules=()))
    ig = _FakeAgent(lambda module: [])
    config = SupervisorConfig(max_retries=1, step_timeout_seconds=5.0, backoff_base_seconds=0.001)
    supervisor, repo, _ = _build_supervisor(sm, cd, ig, config=config)

    events = list(supervisor.run_streaming(target_role="Role", competencies=["x"]))
    event_types = [e.event_type for e in events]
    assert event_types.count("step_failed") == 1
    assert event_types.count("step_retrying") == 1
    # two steps run in total (standards_mapper, curriculum_designer; no
    # modules -> no item_generator step), both eventually succeed
    assert event_types.count("step_succeeded") == 2
    sm_events = [e for e in events if e.agent_name == "standards_mapper"]
    assert [e.event_type for e in sm_events] == [
        "step_started", "step_failed", "step_retrying", "step_succeeded",
    ]


def test_run_and_run_streaming_produce_identical_final_run_state():
    """run() is implemented on top of run_streaming() specifically so
    they can't drift apart — this pins that down directly."""
    sm, cd, ig = _happy_path_agents()
    supervisor1, repo1, _ = _build_supervisor(sm, cd, ig)
    run_via_run = supervisor1.run(target_role="Role", competencies=["x"])

    sm2, cd2, ig2 = _happy_path_agents()
    supervisor2, repo2, _ = _build_supervisor(sm2, cd2, ig2)
    events = list(supervisor2.run_streaming(target_role="Role", competencies=["x"]))
    run_via_streaming = repo2.get_run(events[0].run_id)

    assert run_via_run.status == run_via_streaming.status == RunStatus.SUCCEEDED
    assert len(repo1.list_pending()) == len(repo2.list_pending())


# --- FR-6: cancellation ---

def test_cancellation_before_first_step_stops_all_work():
    sm, cd, ig = _happy_path_agents()
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)
    token = CancellationToken()
    token.cancel()  # cancelled before the run even starts

    events = list(supervisor.run_streaming(target_role="Role", competencies=["x"], cancellation_token=token))

    event_types = [e.event_type for e in events]
    assert "run_cancelled" in event_types
    assert "step_started" not in event_types  # no agent work was ever started
    run = repo.get_run(events[0].run_id)
    assert run.status == RunStatus.CANCELLED
    assert fallback_uc.calls == []  # cancellation must NOT trigger graceful degradation
    assert repo.list_pending() == []  # nothing generated, nothing submitted


def test_cancellation_between_steps_stops_remaining_steps():
    """Cancel after standards_mapper has already succeeded — item
    generation for later modules must never start."""
    sm, cd, ig = _happy_path_agents()
    token = CancellationToken()

    call_log = []
    original_cd_execute = cd.execute

    def cd_execute_and_cancel(gr):
        result = original_cd_execute(gr)
        token.cancel()  # simulate the client disconnecting right after this step
        call_log.append("curriculum_designer ran")
        return result

    cd.execute = cd_execute_and_cancel

    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)
    events = list(supervisor.run_streaming(target_role="Role", competencies=["x"], cancellation_token=token))

    assert call_log == ["curriculum_designer ran"]  # curriculum_designer DID run
    event_types = [e.event_type for e in events]
    assert "run_cancelled" in event_types
    # item_generator's step_started must never appear — cancellation caught
    # at the next step boundary, before item_generator (the next agent) runs
    item_gen_started = [
        e for e in events if e.event_type == "step_started" and e.agent_name == "item_generator"
    ]
    assert item_gen_started == []
    run = repo.get_run(events[0].run_id)
    assert run.status == RunStatus.CANCELLED
    assert fallback_uc.calls == []
    assert repo.list_pending() == []  # item_generator never ran, nothing to submit


def test_cancellation_takes_precedence_over_degradation():
    """A cancelled run must end as CANCELLED, never DEGRADED — even
    though both paths originate from an exception caught in run_streaming."""
    sm, cd, ig = _happy_path_agents()
    token = CancellationToken()
    token.cancel()
    supervisor, repo, fallback_uc = _build_supervisor(sm, cd, ig)

    events = list(supervisor.run_streaming(target_role="Role", competencies=["x"], cancellation_token=token))
    run = repo.get_run(events[0].run_id)

    assert run.status == RunStatus.CANCELLED
    assert run.status != RunStatus.DEGRADED
    assert fallback_uc.calls == []