"""
Ports (interfaces) that the application layer depends on. Infrastructure
adapters implement these. This is the seam that makes the provider swap-out
acceptance test possible: change an adapter + config, never this file or
the use cases that consume it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

from domain.entities import Chunk, Document, RetrievedChunk


class EmbeddingProvider(ABC):
    """Turns text into dense vectors. Implementations: local (offline) or a
    hosted API — selected by configuration (see infrastructure/embeddings/)."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...


class VectorStore(ABC):
    """Persists chunk vectors and answers dense similarity queries."""

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        """Returns (chunk, similarity_score) pairs, highest score first."""
        ...


class KeywordIndex(ABC):
    """Sparse / lexical index (BM25 or equivalent) for hybrid retrieval."""

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def query(self, text: str, top_k: int) -> list[tuple[Chunk, float]]:
        ...


class DocumentRepository(ABC):
    """Relational persistence for documents and chunks (source of truth)."""

    @abstractmethod
    def save_document(self, document: Document) -> None:
        ...

    @abstractmethod
    def find_by_content_hash(self, content_hash: str) -> Document | None:
        ...

    @abstractmethod
    def save_chunks(self, chunks: Iterable[Chunk]) -> None:
        ...

    @abstractmethod
    def delete_chunks_for_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def find_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        """Canonical chunk lookup by id — used by the FR-3/FR-4 validation
        passes to verify a claim against the *current* stored chunk text,
        rather than trusting a copy an agent may have cached earlier."""
        ...


class LLMProvider(ABC):
    """Chat/completion provider used to synthesize a grounded answer."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...