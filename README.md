# RAG Copilot — Ingestion & Hybrid Retrieval Slice

One fully-implemented feature out of the full capstone spec: **FR-1
(ingestion) + FR-2 (hybrid retrieval with citations and refusal)**, built
with Clean Architecture as the foundation the rest of the system (agents,
API surface, security controls) will sit on top of. See `PLAN.md` at the
repo root (one level up) for how this slice fits the full spec and what's
still to build.

## Architecture

```
domain/           entities.py, ports.py, errors.py
                  Zero imports from infrastructure/, no LLM/vector-store/web-framework SDKs.
application/      chunking.py, ingest.py, retrieve.py
                  Use cases + chunking strategy. Depends only on domain/ ports.
infrastructure/   embeddings/ (tfidf + hosted-stub), vectorstore/ (numpy),
                  keyword/ (bm25), relational/ (sqlite), llm/ (extractive + gemini), config.py
                  Concrete adapters implementing domain/ports.py. config.py is the
                  composition root — the only file that wires domain to infrastructure.
interface/        cli.py — thin, imports use cases + config only.
tests/            unit (chunking, fusion) + integration (full pipeline, real adapters)
docs/             ADR-001 (chunking), ADR-002 (fusion + enhancement)
```

**Acceptance test this satisfies**: swap `TfidfEmbeddingProvider` for
`HostedEmbeddingProvider` (or a real neural embedder), or `NumpyVectorStore`
for a Qdrant/pgvector adapter — you change `infrastructure/config.py` and
add one adapter file. Nothing in `domain/` or `application/` changes. Try
it: `grep -rn "sklearn\|rank_bm25\|sqlite3\|numpy" domain/ application/`
returns nothing.

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

## Explicitly out of scope for this slice
Multi-agent orchestration (FR-4/5), streaming (FR-6), the full HTTP API/UI
(FR-7), auth/roles (FR-8), tracing/cost accounting (FR-9), and the OWASP
Top 10 controls beyond the one injection test above. These are sequenced in
`PLAN.md`.