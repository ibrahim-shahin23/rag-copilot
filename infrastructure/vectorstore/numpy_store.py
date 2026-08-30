"""
Vector store adapter — in-process numpy cosine-similarity index, persisted
to disk as a flat file.

This is the MVP swap-in for a real vector database (Qdrant / pgvector,
per the spec's "relational store + vector store with migrations"
requirement at full-system scope). It implements the same VectorStore
port, so promoting to Qdrant later is a new adapter file plus one config
line — never a change to application/retrieve.py or application/ingest.py.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from domain.entities import Chunk
from domain.ports import VectorStore


class NumpyVectorStore(VectorStore):
    def __init__(self, persist_path: str | None = None) -> None:
        self._persist_path = Path(persist_path) if persist_path else None
        self._ids: list[str] = []
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, np.ndarray] = {}
        if self._persist_path and self._persist_path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._persist_path, "rb") as f:
            state = pickle.load(f)
        self._ids = state["ids"]
        self._chunks = state["chunks"]
        self._vectors = state["vectors"]

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._persist_path, "wb") as f:
            pickle.dump(
                {"ids": self._ids, "chunks": self._chunks, "vectors": self._vectors}, f
            )

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors):
            arr = np.asarray(vector, dtype=np.float32)
            norm = np.linalg.norm(arr)
            if norm > 0:
                arr = arr / norm
            if chunk.id not in self._chunks:
                self._ids.append(chunk.id)
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = arr
        self._save()

    def delete_by_document(self, document_id: str) -> None:
        to_remove = [cid for cid, c in self._chunks.items() if c.document_id == document_id]
        for cid in to_remove:
            self._chunks.pop(cid, None)
            self._vectors.pop(cid, None)
            if cid in self._ids:
                self._ids.remove(cid)
        self._save()

    def query(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if not self._ids:
            return []
        q = np.asarray(vector, dtype=np.float32)
        qnorm = np.linalg.norm(q)
        if qnorm > 0:
            q = q / qnorm
        scored: list[tuple[str, float]] = []
        for cid in self._ids:
            v = self._vectors[cid]
            # vectors may differ in dimensionality across a re-fit TF-IDF
            # vocabulary; guard defensively rather than crash mid-query.
            if v.shape != q.shape:
                continue
            score = float(np.dot(v, q))
            scored.append((cid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(self._chunks[cid], score) for cid, score in scored[:top_k]]
