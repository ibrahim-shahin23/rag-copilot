"""
Extraction adapter — the "extract" stage of FR-1's extract -> clean ->
chunk -> embed -> index pipeline.

This was previously collapsed into a bare `path.read_text()` call in the
CLI, which only ever worked for already-plain-text formats (.md, .txt) and
throws a raw UnicodeDecodeError on anything binary, like a PDF. That's a
real bug, not a documented limitation, so it's fixed here rather than
caveated: this module is what FR-1's "≥2 input formats" requirement
actually depends on.
"""
from __future__ import annotations

from pathlib import Path

from domain.errors import UnsupportedFormatError

_SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def extract_text(path: Path) -> str:
    """Extract raw text from a file based on its extension. Raises
    UnsupportedFormatError for anything not yet supported, rather than
    letting a low-level decode error leak up to the CLI."""
    suffix = path.suffix.lower()

    if suffix in _SUPPORTED_TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8")

    if suffix == ".pdf":
        return _extract_pdf(path)

    raise UnsupportedFormatError(
        f"No extractor registered for '{suffix}' files "
        f"(supported: {sorted(_SUPPORTED_TEXT_EXTENSIONS | {'.pdf'})})"
    )


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader  # local import: keeps pypdf optional for
                                   # environments that only ingest text/markdown

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            # Tag each page so the downstream section-splitter in
            # application/chunking.py has a natural boundary to key off,
            # and so citations can eventually report a page number.
            pages.append(f"[Page {i + 1}]\n{page_text}")
    if not pages:
        raise UnsupportedFormatError(
            f"'{path.name}' produced no extractable text — it may be a "
            f"scanned/image-only PDF, which needs OCR (not yet supported here)"
        )
    return "\n\n".join(pages)
