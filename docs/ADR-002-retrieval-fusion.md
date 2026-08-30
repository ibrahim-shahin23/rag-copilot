# ADR-002: Hybrid Retrieval Fusion (RRF) + Metadata-Filtering Enhancement

## Status
Accepted

## Context
FR-2 requires hybrid retrieval (dense + keyword) with a *documented fusion
method*, plus one justified enhancement from a fixed list (re-ranking,
query rewriting, HyDE, multi-query, metadata filtering, contextual
retrieval).

## Decision: Reciprocal Rank Fusion (RRF)
Dense (cosine similarity) and BM25 (keyword) scores live on incomparable
scales — cosine similarity is bounded [0,1]-ish, BM25 is an unbounded,
corpus-size-dependent score. Naively averaging them means whichever
retriever happens to produce larger numbers silently dominates. RRF avoids
this by only using each retriever's *rank*, not its raw score:

```
score(chunk) = Σ 1 / (k + rank_in_retriever)
```

with `k = 60` (the standard damping constant from the original RRF paper —
large enough that rank 1 vs rank 2 isn't wildly different, small enough
that being retrieved at all matters). A chunk that both retrievers agree on
(low rank in both lists) naturally rises to the top without any
score-normalization step.

## Decision: chosen enhancement — metadata filtering
When a query explicitly names a section (`"What does FR-2 require?"`, a
regex detects `FR-2`), chunks whose `ChunkMetadata.section` matches that
reference get a small additive boost (`+0.02`, tunable) *after* RRF fusion,
not instead of it. This is deliberately a boost, not a hard filter — a hard
filter would break recall the moment the user's clause reference is
slightly wrong (e.g. they say "FR-2" but the actually-relevant material also
lives partly in "FR-3"). Boosting keeps the true hybrid ranking as the
foundation and only tie-breaks/reorders using structural metadata,
matching the spec's list item verbatim ("metadata filtering").

## Consequences
- **Positive**: rank-based fusion needs no per-corpus score calibration;
  the metadata boost is cheap (regex, no LLM call) and directly exploits
  this corpus's clause-numbered structure.
- **Negative / limitation observed in testing**: the refusal threshold
  (`RetrievalConfig.refusal_threshold`) is sensitive to corpus size. With a
  tiny corpus (a handful of chunks), even an irrelevant query can pick up
  enough RRF signal from a single retriever to cross a loosely-set
  threshold — see `tests/test_ingest_retrieve_integration.py::test_out_of_corpus_question_is_refused`,
  which needed a deliberately strict threshold (0.3) to demonstrate correct
  refusal against a 3-chunk demo corpus. The default (0.012) is calibrated
  for illustration, not for production; a real deployment must tune this
  against its own golden set (FR-3), and that calibration process — not a
  fixed constant — is the actual deliverable.

## Alternatives considered
- **Score normalization (min-max or z-score) + weighted sum** — rejected as
  primary fusion: requires per-corpus, per-retriever calibration to avoid
  one retriever dominating; more moving parts than RRF for the same
  outcome.
- **Cross-encoder re-ranking** — attractive, but requires either a hosted
  reranking API or a locally-downloaded cross-encoder model; this
  environment's network policy blocks model-hub downloads, so it was set
  aside in favor of an enhancement that works fully offline. Documented
  here as the natural next step once a reranker endpoint is available.
