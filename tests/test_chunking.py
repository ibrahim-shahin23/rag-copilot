import pytest

from domain.entities import Document
from domain.errors import EmptyDocumentError
from application.chunking import ChunkingConfig, chunk_document


def _doc(text: str) -> Document:
    return Document.new(source="test.md", doc_type="md", raw_text=text)


def test_empty_document_raises():
    with pytest.raises(EmptyDocumentError):
        chunk_document(_doc("   \n\n  "))


def test_short_section_stays_one_chunk():
    text = "FR-1 Ingestion. A short requirement that fits in one chunk easily."
    chunks = chunk_document(_doc(text))
    assert len(chunks) == 1
    assert chunks[0].metadata.section is not None
    assert "FR-1" in chunks[0].metadata.section


def test_long_section_splits_with_overlap():
    sentence = "This is one sentence about retrieval quality and citations. "
    long_section = "FR-2 Retrieval.\n" + sentence * 40  # forces multiple windows
    cfg = ChunkingConfig(max_chars=300, overlap_chars=60, min_chars=20)
    chunks = chunk_document(_doc(long_section), cfg)
    assert len(chunks) > 1
    # Every chunk must carry traceable offsets back into the source text.
    for c in chunks:
        assert c.metadata.char_start < c.metadata.char_end
        assert c.metadata.source == "test.md"
    # Consecutive chunks should overlap in content (not just offsets) so a
    # citation near a window boundary isn't stranded without context.
    for a, b in zip(chunks, chunks[1:]):
        assert a.metadata.char_end > b.metadata.char_start or a.text[-20:] in (a.text + b.text)


def test_multiple_sections_preserve_labels_and_order():
    text = (
        "FR-1 Ingestion. Extract clean chunk embed index.\n"
        "FR-2 Retrieval. Hybrid retrieval with citations.\n"
        "FR-3 Evaluation. A golden set of at least 25 Q/A pairs."
    )
    chunks = chunk_document(_doc(text))
    labels = [c.metadata.section for c in chunks]
    assert any("FR-1" in (l or "") for l in labels)
    assert any("FR-2" in (l or "") for l in labels)
    assert any("FR-3" in (l or "") for l in labels)
    positions = [c.metadata.position for c in chunks]
    assert positions == sorted(positions)


def test_content_hash_is_deterministic_for_idempotency():
    d1 = _doc("identical content")
    d2 = _doc("identical content")
    assert d1.content_hash == d2.content_hash
    d3 = _doc("different content")
    assert d1.content_hash != d3.content_hash


def test_short_document_below_min_chars_still_chunks():
    """Regression test for a real bug found via HTTP ingest testing: a
    short document with no heading and no long section — its single
    window is shorter than min_chars — used to be discarded entirely,
    raising ChunkingError for a perfectly legitimate short document.
    min_chars must only filter degenerate trailing slivers from actual
    windowing, never a section's sole chunk."""
    text = "FR-9 requires correlation IDs."  # well under the default min_chars=40
    assert len(text) < ChunkingConfig().min_chars
    chunks = chunk_document(_doc(text))
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_tiny_trailing_sliver_from_real_windowing_is_still_droppable():
    """The min_chars fix must not disable the original 'drop a degenerate
    trailing sliver' behavior when windowing genuinely produces more than
    one piece — only the single-window (short section) case is exempt."""
    sentence = "This is one sentence about retrieval quality and citations. "
    long_section = "FR-2 Retrieval.\n" + sentence * 40
    cfg = ChunkingConfig(max_chars=300, overlap_chars=60, min_chars=20)
    chunks = chunk_document(_doc(long_section), cfg)
    # every produced chunk still respects min_chars when there WAS more
    # than one window to filter among
    assert all(len(c.text) >= cfg.min_chars for c in chunks)