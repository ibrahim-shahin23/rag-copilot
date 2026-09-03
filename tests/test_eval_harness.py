from domain.entities import Chunk, ChunkMetadata

from eval.harness import ItemResult, _aggregate


def _chunk(cid: str, source: str, text: str = "text") -> Chunk:
    return Chunk(
        id=cid, document_id="d1", text=text,
        metadata=ChunkMetadata(source=source, section=None, position=0,
                                char_start=0, char_end=len(text), version="1"),
    )


def test_aggregate_computes_overall_and_per_category_rates():
    results = [
        ItemResult(
            id="a", category="standard", question="q", refused=False,
            retrieved_sources=["x.md"], citation_sources=["x.md"], answer_text="ans",
            retrieval_hit=True, groundedness=1.0, refusal_correct=True,
        ),
        ItemResult(
            id="b", category="standard", question="q", refused=False,
            retrieved_sources=["x.md"], citation_sources=["x.md"], answer_text="ans",
            retrieval_hit=False, groundedness=0.0, refusal_correct=True,
        ),
        ItemResult(
            id="c", category="adversarial:out_of_corpus", question="q", refused=False,
            retrieved_sources=[], citation_sources=[], answer_text="ans",
            refusal_correct=False,
        ),
    ]
    summary = _aggregate(results)
    assert summary["n_items"] == 3
    assert summary["retrieval_hit_rate"] == 0.5      # 1 of 2 items with a hit judgment
    assert summary["groundedness"] == 0.5             # mean of 1.0 and 0.0
    assert summary["refusal_correctness"] == round(2 / 3, 3)

    assert summary["by_category"]["standard"]["n"] == 2
    assert summary["by_category"]["adversarial:out_of_corpus"]["refusal_correctness"] == 0.0


def test_items_with_none_fields_are_excluded_from_aggregates_not_counted_as_failures():
    """Injection items (expected_refusal=None) must not drag down the
    refusal-correctness metric just because they don't have an opinion on
    it — they're scored on leak-freedom instead."""
    results = [
        ItemResult(
            id="inj", category="adversarial:prompt_injection", question="q",
            refused=False, retrieved_sources=[], citation_sources=[], answer_text="ans",
            leak_free=True,  # no refusal_correct set -> None
        ),
    ]
    summary = _aggregate(results)
    assert summary["refusal_correctness"] is None
    assert summary["injection_leak_free_rate"] == 1.0