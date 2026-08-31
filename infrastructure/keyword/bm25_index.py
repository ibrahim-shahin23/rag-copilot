"""BM25 keyword/lexical index — the sparse leg of hybrid retrieval."""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Sequence

from rank_bm25 import BM25Okapi

from domain.entities import Chunk
from domain.ports import KeywordIndex

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25KeywordIndex(KeywordIndex):
    def __init__(self, persist_path: str | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._chunks: dict[str, Chunk] = {}
        self._order: list[str] = []
        self._bm25: BM25Okapi | None = None
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._persist_path, "rb") as f:
            state = pickle.load(f)
        self._chunks = state["chunks"]
        self._order = state["order"]
        self._rebuild()

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump({"chunks": self._chunks, "order": self._order}, f)

    def _rebuild(self) -> None:
        if not self._order:
            self._bm25 = None
            return
        corpus = [_tokenize(self._chunks[cid].text) for cid in self._order]
        self._bm25 = BM25Okapi(corpus)

    def index(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            if chunk.id not in self._chunks:
                self._order.append(chunk.id)
            self._chunks[chunk.id] = chunk
        self._rebuild()
        self._save()

    def delete_by_document(self, document_id: str) -> None:
        to_remove = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
        for cid in to_remove:
            self._chunks.pop(cid, None)
            if cid in self._order:
                self._order.remove(cid)
        self._rebuild()
        self._save()

    def query(self, text: str, top_k: int) -> list[tuple[Chunk, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._order, scores), key=lambda x: x[1], reverse=True)
        # Do NOT filter by score > 0: with a very small corpus, BM25's IDF
        # goes negative for terms that appear in most/all documents, which
        # can push an otherwise-relevant chunk's score below zero. RRF
        # fusion (application/retrieve.py) only consumes *rank*, never the
        # raw score, so returning the full top_k regardless of sign is both
        # safe and necessary — filtering here was silently discarding
        # legitimate hits on small corpora. Mirrors NumpyVectorStore.query,
        # which already returns top_k unconditionally.
        return [(self._chunks[cid], float(score)) for cid, score in ranked[:top_k]]
