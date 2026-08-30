"""
SQLite repository — relational source of truth for documents and chunks.

Swap-in for Postgres at full-system scope (spec asks for "a relational
store... with migrations"); this adapter implements DocumentRepository so
that promotion is a new adapter class, not a change to application/ingest.py.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable

from domain.entities import (
    Chunk,
    ChunkMetadata,
    Document,
    DocumentMetadata,
    IngestStatus,
)
from domain.ports import DocumentRepository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    version TEXT NOT NULL,
    raw_text TEXT NOT NULL,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id),
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    section TEXT,
    position INTEGER NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
"""


class SqliteDocumentRepository(DocumentRepository):
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_document(self, document: Document) -> None:
        self._conn.execute(
            """
            INSERT INTO documents (id, source, doc_type, version, raw_text,
                                    content_hash, status, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, error=excluded.error
            """,
            (
                document.id,
                document.metadata.source,
                document.metadata.doc_type,
                document.metadata.version,
                document.raw_text,
                document.content_hash,
                document.status.value,
                document.error,
                document.created_at.isoformat(),
            ),
        )
        self._conn.commit()

    def find_by_content_hash(self, content_hash: str) -> Document | None:
        row = self._conn.execute(
            "SELECT id, source, doc_type, version, raw_text, content_hash, "
            "status, error, created_at FROM documents WHERE content_hash = ?",
            (content_hash,),
        ).fetchone()
        if row is None:
            return None
        return Document(
            id=row[0],
            metadata=DocumentMetadata(source=row[1], doc_type=row[2], version=row[3]),
            raw_text=row[4],
            content_hash=row[5],
            status=IngestStatus(row[6]),
            error=row[7],
        )

    def save_chunks(self, chunks: Iterable[Chunk]) -> None:
        rows = [
            (
                c.id,
                c.document_id,
                c.text,
                c.metadata.source,
                c.metadata.section,
                c.metadata.position,
                c.metadata.char_start,
                c.metadata.char_end,
                c.metadata.version,
            )
            for c in chunks
        ]
        self._conn.executemany(
            """
            INSERT OR REPLACE INTO chunks
            (id, document_id, text, source, section, position, char_start, char_end, version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        self._conn.commit()

    def delete_chunks_for_document(self, document_id: str) -> None:
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        self._conn.commit()

    def chunk_count_for_document(self, document_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row[0] if row else 0
