# RAG Copilot — Ingestion, Hybrid Retrieval, Evaluation & Multi-Agent Workflow

Three fully-implemented pieces of the full capstone spec: **FR-1
(ingestion) + FR-2 (hybrid retrieval with citations and refusal)**,
**FR-3 (evaluation harness with a real golden set and recorded baseline
numbers)**, and **FR-4/FR-5 (a multi-agent curriculum/assessment
pipeline behind a supervisor orchestrator, with a real approval gate)**
— built with Clean Architecture as the foundation the rest of the system
(API surface, security controls, streaming) will sit on top of. See
`PLAN.md` at the repo root (one level up) for how this fits the full spec
and what's still to build.

## Architecture

```
domain/           entities.py, ports.py, errors.py,
                  workflow_entities.py, workflow_ports.py (FR-4/FR-5 typed contracts)
                  Zero imports from infrastructure/, no LLM/vector-store/web-framework SDKs.
application/      chunking.py, ingest.py, retrieve.py, tools.py, validation.py,
                  agents/ (standards_mapper, curriculum_designer, item_generator),
                  orchestration/ (supervisor.py)
                  Use cases + chunking strategy + the agent pipeline. Depends only on domain/ ports.
infrastructure/   embeddings/ (tfidf + gemini + hosted-openai), vectorstore/ (numpy),
                  keyword/ (bm25), relational/ (sqlite, workflow_repository),
                  llm/ (extractive + gemini), extraction/ (pdf/text),
                  resilience/ (call-time fallback), config.py
                  Concrete adapters implementing domain/ports.py. config.py is the
                  composition root — the only file that wires domain to infrastructure.
interface/        cli.py — thin, imports use cases + config only.
tests/            unit (chunking, fusion, agents, tools, validation, supervisor)
                  + integration (full pipeline, real adapters)
docs/             ADR-001 (chunking), ADR-002 (fusion), ADR-003 (eval methodology),
                  ADR-004 (orchestration pattern)
```

**Acceptance test this satisfies**: swap `TfidfEmbeddingProvider` for
`HostedEmbeddingProvider` (or a real neural embedder), or `NumpyVectorStore`
for a Qdrant/pgvector adapter — you change `infrastructure/config.py` and
add one adapter file. Nothing in `domain/` or `application/` changes. Try
it: `grep -rn "sklearn\|rank_bm25\|sqlite3\|numpy" domain/ application/`
returns nothing (the only stdlib-adjacent exception is
`concurrent.futures` in `application/orchestration/supervisor.py`, which
is Python stdlib, not an LLM/vector-store/web-framework SDK).

## Provider abstraction & fallback chain
Both chains are resolved **per-call** by `FallbackEmbeddingProvider` /
`FallbackLLMProvider` (`infrastructure/resilience/`), not once at startup.
An earlier version only checked "is a key present" at process start, which
missed the more common real failure: a valid key that fails at request
time (quota exhausted, rate limited, transient network error) — that
crashed the CLI outright instead of degrading. See
`tests/test_resilience.py` for the regression test (reproduces an HTTP 429
and confirms it degrades rather than raising).

- **Embeddings**: `GeminiEmbeddingProvider` (`gemini-embedding-001` over
  REST — needs `GEMINI_API_KEY` + network to
  `generativelanguage.googleapis.com`; free tier, no card required) falls
  back to `TfidfEmbeddingProvider` (local, always available) on any
  call-time failure. `infrastructure/embeddings/hosted_provider.py`
  (OpenAI, paid) still exists as an alternative but isn't wired by default
  — the default config needs only one free key for both embeddings and
  completion. This sandbox has no API key or external network access, so
  the local provider is what's actually exercised end-to-end by the tests
  below — the hosted adapter is real code, not a stub, but untested here
  for that reason.
- **Completion**: `GeminiLLMProvider` (Google Gemini, called over REST —
  needs `GEMINI_API_KEY` + network to `generativelanguage.googleapis.com`)
  falls back to `ExtractiveFallbackProvider` (returns the top-cited excerpt
  verbatim instead of synthesized prose) on any call-time failure. This is
  the "design for a free tier running out" requirement — the system
  degrades, it doesn't fail. `python interface/cli.py ask ...` prints
  `llm_provider=...` up front, and a `[fallback] ... degrading to ...`
  notice to stderr whenever a degradation actually happens, so it's never
  silent.
- **Known limitation**: degrading to a different embedding provider
  mid-session means chunks already indexed in the primary provider's
  vector space won't dimensionally match a query embedded by the
  secondary — `NumpyVectorStore.query` already handles that safely (skips
  mismatched vectors instead of crashing), so the result is reduced
  recall, not a crash, but it's a real gap. Embedding-provider versioning
  per chunk is the actual fix, tracked in `PLAN.md`'s roadmap, not solved
  here.

## Known bugs found and fixed since the initial slice
- **PDF ingestion crashed with `UnicodeDecodeError`** — the CLI originally
  called `path.read_text(encoding="utf-8")` unconditionally, which only
  ever worked for already-plain-text formats. Fixed by adding a real
  extraction stage (`infrastructure/extraction/text_extractor.py`,
  supporting `.txt`/`.md` and `.pdf` via `pypdf`), which is what FR-1's
  "extract" pipeline stage was always supposed to be.
- **Dense retrieval silently returned zero hits across separate CLI
  invocations** — `ingest` and `ask` run as two different OS processes.
  The TF-IDF vectorizer's fitted vocabulary lived only in memory, so `ask`
  re-fit a fresh vectorizer on the query text alone, producing a vector of
  the wrong dimensionality to compare against the persisted chunk vectors.
  Fixed by persisting the fitted vectorizer
  (`infrastructure/embeddings/tfidf_provider.py`) and loading it in later
  processes instead of re-fitting.
- **BM25 dropped relevant hits on tiny corpora** — with very few
  documents, a term appearing in most/all of them gets a negative IDF, so
  a naive `score > 0` filter discarded an otherwise-relevant chunk. Fixed
  by returning the full top-k regardless of sign
  (`infrastructure/keyword/bm25_index.py`), matching how RRF fusion
  already only consumes rank, never raw score.
- All three are covered by regression tests in `tests/test_extraction.py`
  and `tests/test_ingest_retrieve_integration.py`
  (`test_cross_process_retrieval_uses_persisted_tfidf_vocabulary`,
  `test_small_corpus_keyword_match_is_not_dropped_by_negative_bm25_idf`).

- **A configured-but-failing hosted provider crashed the CLI outright** —
  `is_configured()` only checked whether an API key was present at
  startup; it never accounted for a valid key failing at call time (HTTP
  429 quota/rate-limit, transient network error). An unhandled `HTTPError`
  propagated straight out of `AnswerQueryUseCase` and crashed the process.
  Fixed with `infrastructure/resilience/fallback_providers.py`, which
  wraps each provider pair and catches call-time failures, degrading to
  the secondary provider with a visible stderr notice instead of raising.
  Regression test: `tests/test_resilience.py`.

- **Gemini 1.5 models are fully shut down (2026), and OpenAI embeddings
  cost money** — real bugs hit in live testing, not simulated. Every call
  to `gemini-1.5-flash` now returns HTTP 404 (Google decommissioned the
  whole 1.5 line). Fixed by switching the default model to
  `gemini-flash-latest`, a Google-maintained alias that tracks whatever
  the current recommended flash model is, so the next model rotation
  doesn't reproduce this same bug. Separately, the OpenAI embedding
  adapter required a paid key; replaced as the default primary with
  `GeminiEmbeddingProvider` (`gemini-embedding-001`), which has a genuine
  free tier — the default config now needs only one free `GEMINI_API_KEY`
  for both embeddings and completion. See
  `infrastructure/embeddings/gemini_embedding_provider.py` and
  `tests/test_gemini_embedding_provider.py`.

## Running it

```bash
pip install -r requirements.txt   # numpy, scikit-learn, rank_bm25, pytest
python -m pytest tests/ -v        # 13 tests, all passing (unit + integration)

python interface/cli.py ingest data/corpus/spec.md
python interface/cli.py ask "What does FR-2 say about citations?"
```

## What's genuinely tested vs. what's a documented stub
- **Tested end-to-end**: extract → clean → chunk → embed (TF-IDF) → index
  (SQLite + numpy vector store + BM25) → hybrid retrieve → fuse (RRF +
  metadata boost) → refuse-or-answer → structured citations. Idempotent
  re-ingestion is tested. An indirect prompt-injection case (malicious
  instruction text embedded in an *ingested document*, not the system
  prompt) is tested to confirm it surfaces only as cited retrieved text,
  never as an executed instruction.
- **Documented but not exercised here**: `HostedEmbeddingProvider` and
  `GeminiLLMProvider` — real adapter code, correct against their
  respective APIs, but this sandbox has neither an API key nor network
  egress to those hosts. That's exactly the fallback-chain scenario they're
  built for.
- **Known limitation, called out rather than hidden**: the refusal
  threshold needs corpus-scale calibration (see ADR-002's Consequences
  section) — with only a handful of chunks, RRF signal from a single
  retriever can cross a loosely-set threshold even for an off-topic query.
  The test suite demonstrates correct refusal by setting an explicit,
  strict threshold for that case, rather than pretending the default
  constant is production-ready.

## Evaluation (FR-3)

```bash
python eval/harness.py
```

Runs fully offline (TF-IDF + BM25 + extractive fallback — no API key
needed, deterministic, safe for CI on every PR) against 31 golden Q/A
pairs (`eval/golden_set.json`: 25 standard + 6 adversarial, covering all
four required adversarial categories). Writes `eval/results/report.md`
(human-readable, with a hand-written Interpretation section) and
`eval/results/report.json` (machine-readable).

**Actual first-run baseline** — see `eval/results/report.md` for full
detail and interpretation, `docs/ADR-003-evaluation-methodology.md` for
why the harness is built this way:

| metric | overall | standard | worst finding |
|---|---|---|---|
| retrieval hit-rate | 96.3% | 100% | ambiguous-term item: 0% |
| groundedness | 78% | 78% | 6/25 items missed the exact chunk |
| refusal correctness | 93.1% | 100% | out-of-corpus: 0/2 |
| injection leak-free | 100% | — | 2/2 |

The two real failures are kept as documented findings, not tuned away — a
refusal-threshold sweep found no value that fixes out-of-corpus refusal
without collapsing standard-question accuracy, meaning the actual fix is
an absolute similarity signal (real embeddings or a re-ranker), not a
better constant. See the report's Interpretation section for the full
threshold-sweep data and root-cause analysis.

## Multi-agent workflow (FR-4/FR-5)

```bash
python interface/cli.py ingest sample_corpus/spec.md
python interface/cli.py ingest sample_corpus/prior_curriculum_notes.md
python interface/cli.py workflow-run "RAG Engineer" \
    --competencies "ingestion pipeline,hybrid retrieval,evaluation harness"
python interface/cli.py trace <run_id>              # full step-by-step detail
python interface/cli.py approvals-list              # items awaiting a human decision
python interface/cli.py approvals-decide <item_id> approve --reviewer "lead-instructor"
```

Three agents (Standards Mapper → Curriculum Designer → Item Generator)
behind a Supervisor orchestrator, education vertical (target role →
competency gaps → module outline → generated assessment items). Full
design writeup: `PLAN.md` §4, `docs/ADR-004-orchestration-pattern.md`.

**Two real bugs found and fixed while building this** (details in
ADR-004, not just asserted here):
1. Standards Mapper's "unmapped" path was unreachable — it used raw
   retrieval (no relevance threshold by design), so a nonsense competency
   like "quantum telepathy" still matched a chunk. Fixed by routing
   through `AnswerQueryUseCase.execute()`'s refusal decision instead —
   which means Standards Mapper now inherits FR-3's already-documented
   threshold-calibration limitation rather than having a silent, separate
   bug of its own.
2. The Supervisor's per-step timeout wasn't actually enforced —
   `with ThreadPoolExecutor() as executor:` blocks on `__exit__` until a
   timed-out call finishes anyway, silently defeating the timeout. Caught
   by a test that measured real wall-clock time end-to-end, fixed with
   explicit `executor.shutdown(wait=False)`.

Mandatory FR-5 controls are real, tested mechanisms, not just recorded
fields: max-iteration breaker, per-step timeout, retry-with-backoff, and
graceful degradation to plain RAG — see `tests/test_supervisor.py`, which
exercises each one directly (a fake step that hangs, one that fails N
times then recovers, one that always fails, etc.) rather than only
testing the happy path.

**Known limitation, not swept under the rug**: the deterministic
item-drafting fallback (used when no real LLM is configured) masks the
*first* number it finds in a source chunk, which is sometimes a
requirement's own ID (e.g. "FR-1") rather than a meaningful fact — the
item is technically valid (the validation pass confirms the masked number
is genuinely in the source) but pedagogically weak. A real LLM replaces
this entirely once `GEMINI_API_KEY` is configured, via the same tool's
LLM-first path.

## Explicitly out of scope for this slice
Streaming (FR-6), the full HTTP API/UI (FR-7 — the CLI covers the same
operations, just not over HTTP with OpenAPI docs), auth/roles (FR-8),
tracing/cost accounting (FR-9), multi-tenancy, and the OWASP Top 10
controls beyond the one injection test in `tests/test_ingest_retrieve_integration.py`.
These are sequenced in `PLAN.md`.