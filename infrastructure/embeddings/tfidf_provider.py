"""
Local embedding provider — works fully offline, no API key required.

This is the "local model" leg of the required 2-implementation provider
abstraction (the other is hosted_provider.py). Swapping between them is a
one-line config change (see infrastructure/config.py); nothing in
domain/ or application/ changes.

Implementation note: TF-IDF is not a neural embedding, but it satisfies the
EmbeddingProvider contract (text -> fixed-length dense vector) and needs no
model download, which matters in network-restricted environments. A real
deployment would point EMBEDDING_PROVIDER at a proper sentence-embedding
model or a hosted API; the interface does not change either way.

Persistence note: the fitted vocabulary MUST be persisted. The CLI runs
`ingest` and `ask` as two separate processes — without persistence, `ask`
fits a brand-new vectorizer on the query text alone, producing a vector
space with a different dimensionality than the one the ingested chunks
were embedded into, and every query silently returns zero hits. This was
caught by testing the real two-process CLI workflow, not the in-process
integration tests (which happened to share one Python process end to end).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from domain.ports import EmbeddingProvider


class TfidfEmbeddingProvider(EmbeddingProvider):
    def __init__(self, max_features: int = 4096, persist_path: str | None = None) -> None:
        self._max_features = max_features
        self._persist_path = Path(persist_path) if persist_path else None
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False
        if self._persist_path and self._persist_path.exists():
            self._load()

    @property
    def name(self) -> str:
        return "local-tfidf"

    def _load(self) -> None:
        with open(self._persist_path, "rb") as f:
            self._vectorizer = pickle.load(f)
        self._fitted = True

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump(self._vectorizer, f)

    def fit(self, corpus: Sequence[str]) -> None:
        """Must be called once with a representative corpus before embed().
        A production swap-in (neural embeddings) would not need this step;
        it exists only because TF-IDF's vocabulary is corpus-dependent."""
        self._vectorizer.fit(corpus)
        self._fitted = True
        self._save()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._fitted:
            # Bootstrap: fit on whatever we're given first (typically the
            # ingestion call, since ingest always runs before ask). Once
            # fitted, the vocabulary is persisted so a later process (e.g.
            # the `ask` CLI invocation) reuses it via transform() below
            # instead of re-fitting on a single query and drifting out of
            # the indexed vector space.
            self._vectorizer.fit(texts)
            self._fitted = True
            self._save()
            return self._vectorizer.transform(texts).toarray().tolist()
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()
