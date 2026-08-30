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
                  keyword/ (bm25), relational/ (sqlite), llm/ (extractive + anthropic), config.py
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
- **Embeddings**: `HostedEmbeddingProvider` (OpenAI API, needs
  `OPENAI_API_KEY` + network) falls back to `TfidfEmbeddingProvider` (local,
  always available) when not configured. This sandbox has no API key or
  external network access, so the local provider is what's actually
  exercised by the tests below — the hosted adapter is real code, not a
  stub, but untested here for that reason.
- **Completion**: `AnthropicLLMProvider` falls back to
  `ExtractiveFallbackProvider` (returns the top-cited excerpt verbatim
  instead of synthesized prose) under the same conditions. This is the
  "design for a free tier running out" requirement — the system degrades,
  it doesn't fail.

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
  `AnthropicLLMProvider` — real adapter code, correct against their
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
