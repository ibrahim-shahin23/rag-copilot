# ADR-004: Supervisor Orchestration Pattern

## Status
Accepted

## Context
FR-5 requires a named, justified orchestration pattern (supervisor,
planner-executor, pipeline, or state machine) with mandatory controls:
max-iteration breaker, per-step timeout, retry with backoff, graceful
degradation to plain RAG, every run inspectable step-by-step by run ID,
and an approval gate supporting approve / reject / edit-and-approve, all
audited.

## Decision: supervisor pattern
The curriculum/assessment pipeline (Standards Mapper -> Curriculum
Designer -> Item Generator -> validation -> submit_for_approval) is a
fixed, known sequence — it doesn't need to be discovered per-request the
way a planner-executor's plan would. A supervisor's simpler control flow
(a single coordinator calling a known sequence of steps, each with a
uniform retry/timeout wrapper) fits that better than a general planner
that reasons about what to do next. `application/orchestration/supervisor.py`
is the implementation; `Supervisor.run()` is the entire pattern in one
method, deliberately — there's no separate planning phase to get wrong.

## Decision: all mandatory controls are real mechanisms, not just fields
Each control below is enforced by actual code, not represented as
metadata someone could forget to check:
- **Max-iteration breaker**: `_run_step` raises `MaxIterationsExceeded`
  before running a step once `step_index >= max_iterations` — checked at
  the top of every step, not just at the end of a loop.
- **Per-step timeout**: `future.result(timeout=...)` on a
  single-worker `ThreadPoolExecutor`, with `executor.shutdown(wait=False)`
  in a `finally` block. This distinction mattered in practice: an earlier
  version used `with ThreadPoolExecutor() as executor:`, and the context
  manager's default `shutdown(wait=True)` on `__exit__` silently blocked
  the caller until the slow call finished anyway — defeating the timeout
  entirely. Caught by a test that measured wall-clock time end-to-end
  (`tests/test_supervisor.py::test_step_timeout_is_actually_enforced_not_just_measured`),
  not by reading the code. Python cannot forcibly kill a running thread,
  so "enforced" here means the orchestrator stops waiting and moves on —
  not that the underlying call is preemptively terminated. True hard
  preemption would need process-level isolation, out of scope for this
  slice.
- **Retry with backoff**: up to `max_retries` retries per step, sleeping
  `backoff_base_seconds * 2**(attempt-1)` between attempts. Every attempt
  — failed or not — is persisted as its own `RunStep`.
- **Graceful degradation to plain RAG**: any exception that survives all
  retries (including `MaxIterationsExceeded` and the curriculum designer's
  `needs_human_input` case) is caught once at the top of `Supervisor.run()`
  and triggers a direct `AnswerQueryUseCase.execute()` call against the
  target role, recorded as its own `DEGRADED` step, rather than the run
  simply failing with nothing to show.
- **Run inspectable step-by-step by run ID**: every `_run_step` call
  persists a `RunStep` via `RunRepository` before returning or raising —
  `get_steps(run_id)` returns the full, ordered history including failed
  attempts, not just the final outcome. `python cli.py trace <run_id>`
  exposes this directly.
- **Approval gate, audited**: `decide()` on `ApprovalGateRepository`
  records `decided_by` and `decided_at` on every decision
  (approve / reject / edit-and-approve), and `edited_and_approved` is the
  only path that populates `approved_text` — an approval or rejection
  can't accidentally carry stale edited text.

## Decision: Standards Mapper's matching uses execute(), not raw retrieve()
Caught during real CLI testing, not simulated: an earlier version had
`StandardsMapperAgent` call `SearchCorpusTool` (wrapping
`AnswerQueryUseCase.retrieve()`), which has no relevance threshold at all
by design (see `application/retrieve.py`). That made "unmapped" nearly
impossible to ever produce — a nonsense competency like "quantum
telepathy" still retrieved *something*, because retrieval always returns
its top-k candidates regardless of whether any of them are actually
relevant. Fixed with a dedicated `AssessCompetencyMatchTool`
(`application/tools.py`) built on `AnswerQueryUseCase.execute()` instead,
reusing the refusal decision the rest of the system has actually been
evaluated against (`eval/results/report.md`) rather than inventing a
second, unvalidated threshold just for this one agent.

This inherits, rather than fixes, the threshold-calibration limitation
FR-3's evaluation already documented (ADR-002): no single scalar
threshold cleanly separates relevant from irrelevant on a small corpus.
Re-running the "quantum telepathy" case after this fix still matches
incorrectly, for the same root-cause reason out-of-corpus refusal fails
2/2 in the eval report — confirmed by direct inspection
(`answer.refused == False`, `citations` all from `spec.md` with scores
~0.03), not assumed. The fix is architecturally correct (Standards Mapper
now shares one tested relevance signal instead of having its own
ungated one) even though the underlying threshold problem is unsolved;
a real embedding model or re-ranker (ADR-002's stated next step) would
improve both FR-2's refusal and Standards Mapper's matching at once,
which is the point of not inventing a parallel heuristic here.

## Decision: submit_for_approval is the one write tool; validation runs on everything
Every generated item — whether the automated validation pass marks it
`validation_passed=True` or `False` — is submitted to the approval gate.
An item that fails validation is not silently dropped; it reaches a human
reviewer with the validation failure reason attached
(`validation_notes`), so they can reject it or fix it via
edit-and-approve. Silently filtering failed items would mean the
"plausible-but-wrong items" catch-and-flag requirement degrades into
catch-and-hide, which defeats the point of a human-in-the-loop gate.

## Consequences / limitations, stated plainly
- The deterministic fallback drafter (`DraftItemTool`, used when no real
  LLM is configured) masks the *first* number it finds in a chunk, which
  is sometimes just a requirement's own ID number (e.g. "FR-1") rather
  than a substantively meaningful fact. The validation pass still
  correctly confirms the masked number appears verbatim in the source —
  the item is technically valid, just pedagogically weak. A real LLM
  (Gemini, once configured) replaces this drafting strategy entirely via
  the same tool's LLM-first path; documented as a known quality
  ceiling of the offline fallback, not hidden.
- Per-step timeout cannot forcibly kill a hung thread (see above) — a
  genuinely malicious or infinite-looping tool call would keep consuming
  a thread in the background even after the orchestrator moves on.
  Acceptable for this slice's trust model (tools call read-only corpus
  search, deterministic drafting, or a single local DB write — nothing
  that should hang); would need reconsidering before running
  less-trusted tools.

## Alternatives considered
- **Planner-executor** — rejected: the pipeline's sequence is fixed and
  known in advance; a planner's extra flexibility (and extra failure
  surface — a bad plan) buys nothing here.
- **State machine** — a reasonable alternative encoding of the same fixed
  sequence with more formal state transitions; supervisor was chosen for
  being the more direct mapping to "one coordinator calls N steps," with
  less ceremony for a pipeline this linear.