# Education Copilot — Full Build Plan

Vertical: **curriculum & assessment design**. Target role → competency gaps
→ module outline → assessment items + answer keys, produced by three
specialised agents (Standards Mapper, Curriculum Designer, Item Generator)
behind an orchestrator, with the lead instructor approving generated items
before they're usable, and an automated validation pass catching
plausible-but-wrong items before a human ever sees them.

This plan sequences the entire spec. **One slice is fully built already**:
ingestion + hybrid retrieval with citations (FR-1/FR-2) — see `README.md`
and the code in `domain/`, `application/`, `infrastructure/`. Everything
below is architected for, referenced against, but not yet built, except
where marked ✅.

---

## 1. Architecture decision (ADR-000, implicit)

**Clean Architecture**, four rings: `domain` → `application` →
`infrastructure` → `interface`. Chosen over Hexagonal/Onion/Vertical Slice
because this system has one dominant cross-cutting concern (swap any
LLM/vector-store/embedding provider without touching business logic) and
Clean Architecture's explicit dependency-rule ("dependencies point inward
only") is the most direct way to make that testable, not just aspirational.
Vertical Slice was the runner-up — better for teams shipping independent
features in parallel, worse for enforcing one shared provider-swap
boundary across ingestion, retrieval, and the agent layer at once.

**Acceptance test** (already demonstrated in the built slice, and the bar
every future PR must clear): swapping the LLM provider, embedding model, or
vector store is configuration plus one new adapter file. `domain/` and
`application/` never import an LLM SDK, vector-store SDK, or web framework.
CI should eventually enforce this with an import-linter rule, not just
code review discipline.

## 2. Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Team familiarity (backend work is already Python/Flask per prior context); rich RAG/agent ecosystem |
| Web framework | FastAPI | OpenAPI generation for free (FR-7), native async for streaming (FR-6) |
| Relational store | Postgres (SQLite adapter ✅ built first as the cheapest thing that proves the port) | Row-level security fits the multi-tenancy isolation requirement directly |
| Vector store | pgvector to start (numpy adapter ✅ built as the MVP swap-in); Qdrant if scale demands a dedicated index | One fewer moving service for a capstone-scale deployment; same Postgres instance can enforce tenant isolation at the DB layer |
| Embeddings | Hosted API (OpenAI or Anthropic-compatible) with local TF-IDF/sentence-transformer fallback ✅ | Provider abstraction requirement; demonstrated fallback chain |
| LLM | Anthropic Claude (primary), extractive fallback ✅ for degraded mode | Matches team's existing Anthropic API experience |
| Orchestration pattern | **Supervisor** (see §4) | Named, justified below |
| Agent contracts | Pydantic models (typed, not free-form text) | FR-4 requires typed I/O between agents |
| Streaming | SSE (not WebSocket) | Simpler client story for a single-direction token stream + progress events; no need for bidirectional |
| Tracing | OpenTelemetry + a lightweight self-hosted trace store (Postgres table is enough at this scale) | FR-9, avoids standing up a separate tracing backend for a capstone deliverable |
| Frontend | Minimal server-rendered pages or a CLI/TUI first, React only if time allows | Spec explicitly says "spend no time on visual design" |

## 3. Requirement → component map

| Req | Component | Status |
|---|---|---|
| FR-1 Ingestion | `application/ingest.py` + adapters | ✅ built |
| FR-2 Retrieval | `application/retrieve.py` + ADR-001/002 | ✅ built |
| FR-3 Evaluation | golden set + harness (`eval/`) | Not built — next priority |
| FR-4 Multi-agent | Standards Mapper, Curriculum Designer, Item Generator + orchestrator | Not built |
| FR-5 Orchestration | Supervisor pattern, approval gate, run inspector | Not built |
| FR-6 Real-time | SSE endpoint, cancellation token propagated to agent loop | Not built |
| FR-7 Surface | FastAPI + OpenAPI + minimal UI | Not built |
| FR-8 Access | Auth (JWT) + instructor/reviewer vs. contributor roles | Not built |
| FR-9 Observability | Correlation ID middleware, cost ledger table | Not built |
| Multi-tenancy | tenant_id column + Postgres RLS policy on every table | Not built |
| Security (§5) | `docs/SECURITY.md` control-to-threat mapping | Not built |
| Engineering process (§6) | Git/PR/CI discipline | Applies from commit 1 |

## 4. Multi-agent design (education vertical)

**Agents** (each: explicit role, restricted tool set, defined I/O, termination condition):

1. **Standards Mapper** — input: target role + raw competency framework
   text (from the ingested corpus). Output: a typed `CompetencyGapReport`
   (list of gaps, each cited back to the source standard via FR-2's
   citation mechanism). Tools: `search_corpus` (read-only, wraps
   `AnswerQueryUseCase`). Terminates when every named competency in the
   target role has either a matched standard or an explicit "unmapped" flag
   — never silently drops one.
2. **Curriculum Designer** — input: `CompetencyGapReport`. Output: a typed
   `ModuleOutline` (ordered modules, each tied to one or more gaps). Tools:
   `search_corpus`, `read_prior_curricula` (read-only). Terminates on a
   complete outline covering every gap, or an explicit "needs human input"
   result if gaps can't be reconciled into a coherent sequence.
3. **Item Generator** — input: one `ModuleOutline` module at a time.
   Output: typed `AssessmentItem[]` (question, options, answer key,
   citation back to source material). Tools: `search_corpus`,
   `draft_item` (pure generation, no side effects), and the one
   write/side-effecting tool: `submit_for_approval` (writes to the
   approval-gate queue — **never executed without passing the gate**, per
   FR-4). Terminates per-module when item count reaches the configured
   target or the module's source material is exhausted.

**Orchestrator**: supervisor pattern (named, per FR-5) — a single
supervisor agent sequences Standards Mapper → Curriculum Designer → Item
Generator, inspects each agent's typed output before advancing, and owns
the mandatory controls: max-iteration breaker per agent, per-step timeout,
retry-with-backoff on transient tool failures, and graceful degradation to
plain RAG (skip straight to `AnswerQueryUseCase` against the corpus) if any
agent in the chain fails its termination condition after retries.
Supervisor over planner-executor: the pipeline here is a fixed, known
sequence (map → design → generate), not a plan that needs to be discovered
per-request, so a supervisor's simpler control flow is a better fit than a
general planner.

**Automated validation pass** (the "plausible-but-wrong items" requirement):
a fourth, non-generative step — deterministic checks (does the marked
correct answer actually appear verbatim in the cited source chunk? are
distractors non-duplicate? is the citation chunk-id real?) plus one LLM-as-
judge pass scoring groundedness before an item reaches the approval gate.
This is validation, not a fourth "specialised agent" in the FR-4 sense —
it's a quality gate the orchestrator runs, not a role with its own tool set.

**Approval gate**: instructor sees each `AssessmentItem` with its citation
and validation-pass result; approve / reject / edit-and-approve, all
audited with correlation ID (FR-9) tying the decision back to the run that
produced it.

## 5. Evaluation harness (FR-3) — next build priority

25+ Q/A pairs against the real corpus once assembled, ≥5 adversarial:
out-of-corpus (✅ pattern demonstrated in the built slice's integration
test), ambiguous (a question matching two conflicting source chunks),
prompt injection (✅ pattern demonstrated — indirect injection via an
ingested document), conflicting sources (two ingested versions of the same
policy disagreeing). Harness reports retrieval hit-rate (did the correct
chunk appear in the fused top-k?), groundedness (does every claim in the
answer trace to a citation?), and refusal correctness (did the system
refuse exactly the questions it should have) — and the plan is to publish
the actual first-run numbers, including the bad ones, the same way ADR-002
already documents the refusal-threshold calibration problem found while
building FR-2, rather than tuning until numbers look good and calling that
the baseline.

## 6. Security (FR-5 §5 requirements, planned control set)

`docs/SECURITY.md` will map each control to the threat it addresses:
object-ownership checks on every resource fetch (broken access control);
parameterised queries throughout, upload validation (injection); rate
limiting per tenant and per user; strict CORS + security headers; `pip-audit`
/ `npm audit` in CI (dependency scanning); structured audit logging with a
secret-redaction filter. LLM-specific: strict separation between the system
prompt and retrieved content (already the pattern in the built
`AnswerQueryUseCase` — retrieved text is only ever interpolated into the
user-role excerpt block, never concatenated into the system prompt); ≥3
injection cases in the golden set (one pattern already demonstrated);
output never rendered as raw HTML or passed to a shell/SQL/file path
unvalidated; per-agent tool allow-lists (already designed into §4 above);
token caps and iteration limits (already designed into the supervisor's
mandatory controls); pinned dependencies + committed lockfile + CI
scanning (`requirements.txt` is pinned ✅ for the built slice already).

## 7. Multi-tenancy

`tenant_id` on every table from the first migration, Postgres row-level
security policy enforcing `tenant_id = current_setting('app.tenant_id')` on
every query — enforced at the data layer, not just filtered in application
code, so a bug in a use case can't leak across tenants. Isolation test:
seed two tenants, attempt a cross-tenant read via every repository method,
assert zero rows returned.

## 8. Roadmap / milestones

1. ✅ **Ingestion + hybrid retrieval + citations** (this repo, done)
2. Evaluation harness + golden set (next — needed before agent work can be
   scored meaningfully)
3. Multi-agent pipeline (Standards Mapper → Curriculum Designer → Item
   Generator) + supervisor + approval gate
4. Streaming + cancellation
5. FastAPI surface + OpenAPI + minimal UI + auth/roles
6. Multi-tenancy + Postgres migration off SQLite/numpy MVP adapters
7. Observability (correlation IDs, cost ledger, tracing) + SECURITY.md
8. Packaging (`docker compose up`, `.env.example`), CI, branch protection,
   release tags — engineering-process discipline (§6) applies from
   milestone 1's first commit, not bolted on here; this line marks when
   it's *complete*, not when it *starts*.

## 9. Engineering process discipline (applies throughout, not just at milestone 8)

Conventional Commits, atomic commits explaining why; every change via PR
with a self-review; GitHub Issues linked via `Closes #N`; CI (build, lint,
test, dependency + secret scan) green on every PR; ≥30 commits across ≥6
days by nature of following this milestone sequence rather than batching
work into single dumps.

## 10. Agentic workflow (this build)

This plan and the built slice were produced with: (1) a project-instruction
file (this repo's `README.md` + ADRs function as that — architecture rules
an assistant must respect); (2) reusable chunking/fusion logic written as
versioned modules, not inline prompts; (3) a clear domain/application/
infrastructure boundary that any future sub-agent (security reviewer, test
writer) can be scoped against without re-deriving it. Where it failed:
TF-IDF's "fit on first call" behavior (documented directly in
`tfidf_provider.py`) wasn't obvious until integration testing surfaced it —
a reminder that even a deliberately-scoped slice needs a real test run
before its adapter is trusted for the next milestone to build on.
