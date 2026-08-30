"""
Chunking strategy — ADR-001 (see docs/ADR-001-chunking.md).

Decision: section-aware chunking. The corpus this system targets (specs,
policy documents, curricula) is structured around headings/numbered clauses
("FR-2", "4. Architecture", "## Module 3"). Splitting blindly by character
count would routinely sever a requirement from its own clause number, which
is exactly the kind of citation error the spec's FR-3 evaluation set
penalizes ("groundedness").

Algorithm:
  1. Split the raw text into sections using a heading regex (markdown `#`
     headers, numbered clauses like "FR-1", "4.", "Section 4", or a blank
     line before an ALL-CAPS/Title-Case short line).
  2. Within a section, if it fits inside `max_chars`, keep it as one chunk.
  3. If a section is longer, slide a window of `max_chars` with
     `overlap_chars` overlap, breaking on sentence boundaries where
     possible so a chunk never starts or ends mid-sentence.
  4. Every chunk records its originating section label, its ordinal
     position, and its exact char offsets in the source document — this is
     what makes citations traceable to the exact chunk (FR-2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from domain.entities import Chunk, Document
from domain.errors import ChunkingError, EmptyDocumentError

_HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|(?:FR|ADR|NFR)-\d+\b.*|\d+(?:\.\d+)*\.\s+.+|[A-Z][A-Za-z0-9 &/'\-]{2,60})$",
    re.MULTILINE,
)
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ChunkingConfig:
    max_chars: int = 800
    overlap_chars: int = 150
    min_chars: int = 40  # drop degenerate trailing slivers


def _split_into_sections(text: str) -> list[tuple[str | None, str, int]]:
    """Returns list of (section_label, section_text, char_start_offset)."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [(None, text, 0)]

    sections: list[tuple[str | None, str, int]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        label = m.group().strip()[:80]
        body = text[start:end]
        if body.strip():
            sections.append((label, body, start))

    # Capture any preamble before the first heading.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()]
        if preamble.strip():
            sections.insert(0, (None, preamble, 0))
    return sections


def _sliding_window(section_text: str, cfg: ChunkingConfig) -> list[tuple[str, int, int]]:
    """Returns list of (chunk_text, local_start, local_end) within section_text."""
    if len(section_text) <= cfg.max_chars:
        return [(section_text, 0, len(section_text))]

    sentences = _SENTENCE_BOUNDARY_RE.split(section_text)
    windows: list[tuple[str, int, int]] = []
    cursor = 0
    buf_start = 0
    buf = ""
    for sent in sentences:
        if not sent:
            continue
        # locate sentence in original text starting from cursor to track offsets
        idx = section_text.index(sent, cursor)
        cursor = idx + len(sent)
        if len(buf) + len(sent) + 1 > cfg.max_chars and buf:
            windows.append((buf.strip(), buf_start, buf_start + len(buf)))
            # start new buffer with overlap: back up overlap_chars worth of prior text
            overlap_start = max(0, len(buf) - cfg.overlap_chars)
            carry = buf[overlap_start:]
            buf_start = buf_start + overlap_start
            buf = carry + " " + sent
        else:
            if not buf:
                buf_start = idx
            buf = (buf + " " + sent) if buf else sent
    if buf.strip() and len(buf.strip()) >= cfg.min_chars:
        windows.append((buf.strip(), buf_start, buf_start + len(buf)))
    elif buf.strip() and windows:
        # merge tiny trailing sliver into previous window
        prev_text, prev_start, _ = windows[-1]
        windows[-1] = (prev_text + " " + buf.strip(), prev_start, buf_start + len(buf))
    return windows


def chunk_document(document: Document, cfg: ChunkingConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkingConfig()
    text = document.raw_text
    if not text or not text.strip():
        raise EmptyDocumentError(f"Document {document.id} has no extractable text")

    try:
        sections = _split_into_sections(text)
        chunks: list[Chunk] = []
        position = 0
        for label, section_text, section_offset in sections:
            for chunk_text, local_start, local_end in _sliding_window(section_text, cfg):
                if len(chunk_text.strip()) < cfg.min_chars:
                    continue
                chunks.append(
                    Chunk.new(
                        document=document,
                        text=chunk_text.strip(),
                        section=label,
                        position=position,
                        char_start=section_offset + local_start,
                        char_end=section_offset + local_end,
                    )
                )
                position += 1
        if not chunks:
            raise ChunkingError(f"Document {document.id} produced zero valid chunks")
        return chunks
    except EmptyDocumentError:
        raise
    except Exception as e:  # noqa: BLE001 - convert unexpected failures to domain error
        raise ChunkingError(f"Failed to chunk document {document.id}: {e}") from e
