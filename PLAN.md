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

**ADR status** (spec requires ≥4, covering chunking/retrieval,
orchestration pattern, vector store choice, and the twist's central
decision): ADR-001 (chunking) ✅, ADR-002 (retrieval fusion) ✅, ADR-003
(evaluation methodology — bonus, not one of the four required topics) ✅,
ADR-004 (orchestration pattern) ✅. Still missing: a dedicated vector store
choice ADR (the numpy MVP vs. pgvector/Qdrant tradeoff is described in §2's
stack table but not yet formalized as its own ADR) and a twist's-central-
decision ADR (the education vertical's agent design is documented in §4
but, similarly, not yet its own ADR). Tracked here rather than silently
assumed done.

## 2. Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.12 | Team familiarity (backend work is already Python/Flask per prior context); rich RAG/agent ecosystem |
| Web framework | FastAPI | OpenAPI generation for free (FR-7), native async for streaming (FR-6) |
| Relational store | Postgres (SQLite adapter ✅ built first as the cheapest thing that proves the port) | Row-level security fits the multi-tenancy isolation requirement directly |
| Vector store | pgvector to start (numpy adapter ✅ built as the MVP swap-in); Qdrant if scale demands a dedicated index | One fewer moving service for a capstone-scale deployment; same Postgres instance can enforce tenant isolation at the DB layer |
| Embeddings | Gemini's free-tier hosted API (`gemini-embedding-001`) with local TF-IDF fallback ✅ | Free-tier friendly, same key as the LLM; provider abstraction still swaps to OpenAI/other in one adapter |
| LLM | Google Gemini (primary), extractive fallback ✅ for degraded mode | Free-tier friendly |
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
| FR-3 Evaluation | golden set + harness (`eval/`) | ✅ built |
| FR-4 Multi-agent | Standards Mapper, Curriculum Designer, Item Generator + orchestrator | ✅ built |
| FR-5 Orchestration | Supervisor pattern, approval gate, run inspector | ✅ built |
| FR-6 Real-time | SSE endpoint, cancellation token propagated to agent loop | Not built |
| FR-7 Surface | FastAPI + OpenAPI + minimal UI | Not built |
| FR-8 Access | Auth (JWT) + instructor/reviewer vs. contributor roles | Not built |
| FR-9 Observability | Correlation ID middleware, cost ledger table | Not built |
| Multi-tenancy | tenant_id column + Postgres RLS policy on every table | Not built |
| Security (§5) | `docs/SECURITY.md` control-to-threat mapping | Not built |
| Engineering process (§6) | Git/PR/CI discipline | Applies from commit 1 |

## 4. Multi-agent design (education vertical) — ✅ built

**Agents** (each: explicit role, restricted tool set, defined I/O, termination condition):

1. **Standards Mapper** (`application/agents/standards_mapper.py`) —
   input: target role + a caller-supplied list of competencies (role→
   competency taxonomies are a real product surface on their own; out of
   scope for this slice). Output: a typed `CompetencyGapReport`. Tool:
   `AssessCompetencyMatchTool` only (read-only) — **not** the originally
   planned `search_corpus`; see the note below on why that changed.
   Terminates when every named competency has either a matched standard
   (with citation) or an explicit "unmapped" flag — enforced structurally
   by a single loop over the input list, not a convention to remember.
2. **Curriculum Designer** (`application/agents/curriculum_designer.py`)
   — input: `CompetencyGapReport`. Output: a typed `ModuleOutline`. Tools:
   `search_corpus`, `read_prior_curricula` (both read-only). Terminates on
   a module per matched gap, or an explicit `needs_human_input=True`
   result if zero gaps matched.
3. **Item Generator** (`application/agents/item_generator.py`) — input:
   one `Module` at a time. Output: typed `list[AssessmentItem]`. Tools:
   `search_corpus`, `draft_item` (pure generation — LLM-first with a
   deterministic numeric-masking fallback for the offline case, see
   `application/tools.py`). Does **not** hold `submit_for_approval` —
   the orchestrator calls that after validation, so even a misbehaving
   agent has no path to the one write-capable tool. Terminates at the
   configured item-count target or when the module's retrievable source
   material runs out; an un-draftable chunk is skipped, never forced into
   a bad item.

**Real bug found and fixed during implementation**: Standards Mapper
originally used `search_corpus` (raw `AnswerQueryUseCase.retrieve()`),
which has no relevance threshold by design — so a nonsense competency
like "quantum telepathy" still "matched" a chunk, and the "unmapped" path
could never actually trigger. Caught in live CLI testing, not simulated.
Fixed with `AssessCompetencyMatchTool`, built on
`AnswerQueryUseCase.execute()`'s refusal decision instead — the one
relevance signal the rest of the system has actually been evaluated
against (FR-3's `eval/results/report.md`). This *inherits* rather than
fixes FR-3's already-documented threshold-calibration limitation
(ADR-002) — re-testing "quantum telepathy" after the fix still matches
incorrectly, for the same root cause out-of-corpus refusal fails 2/2 in
the eval report. Full writeup: `docs/ADR-004-orchestration-pattern.md`.

**Orchestrator**: supervisor pattern (`application/orchestration/supervisor.py`)
— sequences Standards Mapper → Curriculum Designer → Item Generator,
inspects each agent's typed output before advancing, and owns the
mandatory controls (all real mechanisms, tested, not just documented —
see ADR-004): max-iteration breaker, per-step timeout (thread-pool
`future.result(timeout=...)`), retry-with-backoff, and graceful
degradation to plain RAG. A second real bug was caught here too: an
earlier version's `with ThreadPoolExecutor() as executor:` silently
blocked on `__exit__` until a timed-out call finished anyway, defeating
the timeout — found by a test that measured actual wall-clock time
end-to-end, fixed with explicit `executor.shutdown(wait=False)`.

**Automated validation pass** (`application/validation.py`) — the
"plausible-but-wrong items" requirement. Deterministic checks only (no
LLM-as-judge pass, to keep it as fast/free/deterministic as the rest of
the offline-testable stack): the correct option must appear verbatim in
the *canonically re-fetched* cited chunk (`DocumentRepository.find_chunk_by_id`,
never a possibly-stale local copy), options must be non-duplicate, and
the citation must resolve to a real stored chunk at all. Not a fourth
"specialised agent" in the FR-4 sense — no tool set, no role, just a
quality gate the orchestrator runs between generation and submission.
**Every** item is submitted regardless of validation outcome — a failed
item is flagged (`validation_passed=False`, `validation_notes`
explaining why) and still reaches a human reviewer, rather than being
silently dropped, which would turn "catch and flag" into "catch and hide."

**Approval gate**: `SqliteWorkflowRepository` implements
`ApprovalGateRepository`; `python cli.py approvals-list` /
`approvals-decide <id> approve|reject|edit` cover all three required
decisions, each audited with `decided_by` + `decided_at`
(`edited_and_approved` is the only path that populates `approved_text`).

## 5. Evaluation harness (FR-3) — ✅ built

`eval/harness.py` runs 31 golden Q/A pairs (25 standard + 6 adversarial,
exceeding both the ≥25 and ≥5 minimums) against the real pipeline, fully
offline and deterministic (see ADR-003). All four required adversarial
categories are covered: out-of-corpus (2 items), ambiguous (1), prompt
injection (2), conflicting sources (1).

**Actual baseline, first real run** (`eval/results/report.md` has the full
per-item breakdown and interpretation):

| metric | overall | standard | worst category |
|---|---|---|---|
| retrieval hit-rate | 96.3% | 100% | ambiguous: 0% (1 item) |
| groundedness | 78% | 78% | — |
| refusal correctness | 93.1% | 100% | out-of-corpus: 0% (2 items) |
| injection leak-free | 100% | — | prompt-injection: 100% |

The two real failures — out-of-corpus questions being answered instead of
refused, and groundedness sitting at 78% rather than ~100% — are kept as
documented findings, not tuned away: a refusal-threshold sweep (0.012 to
0.05) found no value that fixes out-of-corpus refusal without collapsing
standard-question refusal-correctness to 40%, which means the real fix is
an absolute similarity signal (a real embedding model, or a re-ranker),
not a better constant. The groundedness gap traces to the same root cause
— `spec.md`'s 12 chunks competing with each other under TF-IDF's weak
lexical matching — and is expected to close substantially once a real
embedding model replaces the offline TF-IDF fallback. Both are tracked in
§8's roadmap rather than being called "done."

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
2. ✅ **Evaluation harness + golden set** (this repo, done — see §5 above)
3. ✅ **Multi-agent pipeline** (Standards Mapper → Curriculum Designer →
   Item Generator) + supervisor + approval gate (this repo, done — see §4
   above)
4. Streaming + cancellation — next priority
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