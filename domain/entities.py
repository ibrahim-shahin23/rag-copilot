"""
Domain entities for the RAG Copilot.

These are plain dataclasses with zero dependency on any LLM SDK, vector-store
SDK, or web framework. This is the "acceptance test" line from the spec:
swapping providers must never touch this file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import hashlib
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)


class IngestStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class DocumentMetadata:
    """Provenance captured at ingestion time (FR-1)."""
    source: str                 # original filename / URL
    doc_type: str                # e.g. "pdf", "md", "txt"
    version: str = "1"           # bumped on re-ingestion of same source


@dataclass
class Document:
    id: str
    metadata: DocumentMetadata
    raw_text: str
    content_hash: str
    status: IngestStatus = IngestStatus.PENDING
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_now)

    @staticmethod
    def new(source: str, doc_type: str, raw_text: str, version: str = "1") -> "Document":
        content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
        return Document(
            id=str(uuid.uuid4()),
            metadata=DocumentMetadata(source=source, doc_type=doc_type, version=version),
            raw_text=raw_text,
            content_hash=content_hash,
        )


@dataclass(frozen=True)
class ChunkMetadata:
    """Everything a citation needs to be traceable to an exact chunk (FR-2)."""
    source: str
    section: Optional[str]
    position: int          # ordinal position within the document
    char_start: int
    char_end: int
    version: str


@dataclass
class Chunk:
    id: str
    document_id: str
    text: str
    metadata: ChunkMetadata

    @staticmethod
    def new(document: Document, text: str, section: Optional[str],
            position: int, char_start: int, char_end: int) -> "Chunk":
        return Chunk(
            id=str(uuid.uuid4()),
            document_id=document.id,
            text=text,
            metadata=ChunkMetadata(
                source=document.metadata.source,
                section=section,
                position=position,
                char_start=char_start,
                char_end=char_end,
                version=document.metadata.version,
            ),
        )


@dataclass(frozen=True)
class Citation:
    """Structured, traceable citation — never a bare string (FR-2)."""
    chunk_id: str
    source: str
    section: Optional[str]
    position: int
    score: float


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    dense_score: float
    keyword_score: float
    fused_score: float


@dataclass(frozen=True)
class Answer:
    """Result of a query. If `refused` is True, `text` explains why and
    `citations` will be empty — the domain enforces this invariant."""
    query: str
    text: str
    citations: tuple[Citation, ...]
    refused: bool

    def __post_init__(self):
        if self.refused and self.citations:
            raise ValueError("A refused answer must not carry citations")
