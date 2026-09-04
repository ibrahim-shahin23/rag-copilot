from domain.entities import Chunk, ChunkMetadata
from domain.workflow_entities import AssessmentItem
from application.validation import validate_item


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        id=cid, document_id="d1", text=text,
        metadata=ChunkMetadata(source="spec.md", section=None, position=0,
                                char_start=0, char_end=len(text), version="1"),
    )


class _FakeRepo:
    def __init__(self, chunks: dict):
        self._chunks = chunks

    def find_chunk_by_id(self, chunk_id):
        return self._chunks.get(chunk_id)


def test_valid_item_passes():
    chunk = _chunk("c1", "FR-1 requires at least two input formats.")
    item = AssessmentItem.new(
        module_id="m1", question="Fill in: ___",
        options=["two", "three"], correct_option_index=0,
        citation_chunk_id="c1", citation_source="spec.md",
    )
    result = validate_item(item, _FakeRepo({"c1": chunk}))
    assert result.validation_passed is True
    assert result.validation_notes == "all checks passed"


def test_correct_answer_not_in_chunk_fails():
    chunk = _chunk("c1", "FR-1 requires at least two input formats.")
    item = AssessmentItem.new(
        module_id="m1", question="Fill in: ___",
        options=["nine hundred", "three"], correct_option_index=0,
        citation_chunk_id="c1", citation_source="spec.md",
    )
    result = validate_item(item, _FakeRepo({"c1": chunk}))
    assert result.validation_passed is False
    assert "not found verbatim" in result.validation_notes


def test_duplicate_options_fail():
    chunk = _chunk("c1", "FR-1 requires at least two input formats.")
    item = AssessmentItem.new(
        module_id="m1", question="Fill in: ___",
        options=["two", "two"], correct_option_index=0,
        citation_chunk_id="c1", citation_source="spec.md",
    )
    result = validate_item(item, _FakeRepo({"c1": chunk}))
    assert result.validation_passed is False
    assert "duplicate options" in result.validation_notes


def test_unresolvable_citation_fails():
    item = AssessmentItem.new(
        module_id="m1", question="Fill in: ___",
        options=["two", "three"], correct_option_index=0,
        citation_chunk_id="does-not-exist", citation_source="spec.md",
    )
    result = validate_item(item, _FakeRepo({}))
    assert result.validation_passed is False
    assert "does not resolve" in result.validation_notes


def test_out_of_range_correct_index_fails():
    chunk = _chunk("c1", "FR-1 requires at least two input formats.")
    item = AssessmentItem.new(
        module_id="m1", question="Fill in: ___",
        options=["two", "three"], correct_option_index=5,
        citation_chunk_id="c1", citation_source="spec.md",
    )
    result = validate_item(item, _FakeRepo({"c1": chunk}))
    assert result.validation_passed is False
    assert "out of range" in result.validation_notes