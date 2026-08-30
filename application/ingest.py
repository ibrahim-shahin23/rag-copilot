"""
IngestDocumentUseCase — application layer.

Pipeline (FR-1): extract -> clean -> chunk -> embed -> index, with per-document
status/failure reporting and idempotent re-ingestion (same content_hash short-
circuits re-processing; a changed hash re-indexes and bumps version).

Depends only on domain ports — no LLM SDK, vector-store SDK, or web framework
import here. That is the acceptance test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from domain.entities import Document, IngestStatus
from domain.errors import DomainError
from domain.ports import DocumentRepository, EmbeddingProvider, KeywordIndex, VectorStore
from application.chunking import ChunkingConfig, chunk_document

_WHITESPACE_RE = re.compile(r"[ \t]+")
_BLANK_LINES_RE = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Minimal, deterministic cleaning: normalize whitespace, collapse
    excessive blank lines, strip trailing spaces. Deliberately does not
    strip punctuation or case — retrieval and citation exactness both
    benefit from preserving original wording."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = _WHITESPACE_RE.sub(" ", text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


@dataclass(frozen=True)
class IngestResult:
    document_id: str
    status: IngestStatus
    chunk_count: int
    reused_existing: bool
    error: str | None = None


class IngestDocumentUseCase:
    def __init__(
        self,
        repo: DocumentRepository,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._repo = repo
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._chunking_config = chunking_config or ChunkingConfig()

    def execute(self, source: str, doc_type: str, raw_text: str) -> IngestResult:
        cleaned = clean_text(raw_text)
        candidate = Document.new(source=source, doc_type=doc_type, raw_text=cleaned)

        existing = self._repo.find_by_content_hash(candidate.content_hash)
        if existing is not None:
            # Idempotent re-ingestion: identical content already indexed.
            return IngestResult(
                document_id=existing.id,
                status=existing.status,
                chunk_count=-1,  # not recomputed; see repo for detail if needed
                reused_existing=True,
            )

        document = candidate
        document.status = IngestStatus.PROCESSING
        self._repo.save_document(document)

        try:
            chunks = chunk_document(document, self._chunking_config)
            vectors = self._embedder.embed([c.text for c in chunks])

            # Replace any prior chunks for this document id (defensive; new
            # documents won't have any, but keeps the op idempotent per-doc).
            self._repo.delete_chunks_for_document(document.id)
            self._vector_store.delete_by_document(document.id)
            self._keyword_index.delete_by_document(document.id)

            self._repo.save_chunks(chunks)
            self._vector_store.upsert(chunks, vectors)
            self._keyword_index.index(chunks)

            document.status = IngestStatus.SUCCEEDED
            self._repo.save_document(document)
            return IngestResult(
                document_id=document.id,
                status=document.status,
                chunk_count=len(chunks),
                reused_existing=False,
            )
        except DomainError as e:
            document.status = IngestStatus.FAILED
            document.error = str(e)
            self._repo.save_document(document)
            return IngestResult(
                document_id=document.id,
                status=document.status,
                chunk_count=0,
                reused_existing=False,
                error=str(e),
            )
