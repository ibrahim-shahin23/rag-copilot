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
"""
from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer

from domain.ports import EmbeddingProvider


class TfidfEmbeddingProvider(EmbeddingProvider):
    def __init__(self, max_features: int = 4096) -> None:
        self._vectorizer = TfidfVectorizer(max_features=max_features)
        self._fitted = False

    @property
    def name(self) -> str:
        return "local-tfidf"

    def fit(self, corpus: Sequence[str]) -> None:
        """Must be called once with a representative corpus before embed().
        A production swap-in (neural embeddings) would not need this step;
        it exists only because TF-IDF's vocabulary is corpus-dependent."""
        self._vectorizer.fit(corpus)
        self._fitted = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not self._fitted:
            # Bootstrap: fit on whatever we're given first (single-doc demo
            # / first ingestion). Re-fitting on every batch is a known
            # limitation of TF-IDF as a stand-in embedder — documented in
            # docs/ADR-002-retrieval-fusion.md's "limitations" section.
            self._vectorizer.fit(texts)
            self._fitted = True
        matrix = self._vectorizer.transform(texts)
        return matrix.toarray().tolist()
