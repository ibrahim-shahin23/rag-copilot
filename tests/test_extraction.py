from pathlib import Path

import pytest
from pypdf import PdfWriter

from domain.errors import UnsupportedFormatError
from infrastructure.extraction.text_extractor import extract_text


def _make_pdf(tmp_path: Path, text: str) -> Path:
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    # pypdf's writer has no direct text-drawing API without reportlab; for
    # a deterministic test we instead round-trip through a real page that
    # already contains extractable text by using the annotation approach
    # is overkill — simplest reliable path is generating via reportlab if
    # available, else skip with a clear reason.
    pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas

    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=(612, 792))
    c.drawString(72, 720, text)
    c.save()
    return pdf_path


def test_pdf_extraction_returns_text_with_page_marker(tmp_path):
    pdf_path = _make_pdf(tmp_path, "FR-1 Ingestion requirement text.")
    text = extract_text(pdf_path)
    assert "[Page 1]" in text
    assert "FR-1" in text


def test_txt_and_md_pass_through_unchanged(tmp_path):
    p = tmp_path / "note.md"
    p.write_text("plain markdown content", encoding="utf-8")
    assert extract_text(p) == "plain markdown content"


def test_unsupported_extension_raises_domain_error(tmp_path):
    p = tmp_path / "data.docx"
    p.write_bytes(b"not really a docx")
    with pytest.raises(UnsupportedFormatError):
        extract_text(p)
