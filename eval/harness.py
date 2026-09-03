"""
FR-3 evaluation harness.

Runs the golden set (eval/golden_set.json) against the real ingestion +
retrieval pipeline, using the fully offline adapters (SQLite, numpy vector
store, BM25, TF-IDF embeddings, extractive-fallback LLM) so this harness
is deterministic and needs no API key — it can run in CI on every PR, not
just on a developer's machine with a key configured.

Usage:
    python eval/harness.py

Writes eval/results/report.json (machine-readable) and
eval/results/report.md (the human-readable report with interpretation,
committed alongside this code per FR-3's "record your actual baseline
numbers, including the bad ones" instruction).
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.ingest import IngestDocumentUseCase
from application.retrieve import AnswerQueryUseCase, RetrievalConfig
from infrastructure.embeddings.tfidf_provider import TfidfEmbeddingProvider
from infrastructure.extraction.text_extractor import extract_text
from infrastructure.keyword.bm25_index import BM25KeywordIndex
from infrastructure.llm.providers import ExtractiveFallbackProvider
from infrastructure.relational.sqlite_repository import SqliteDocumentRepository
from infrastructure.vectorstore.numpy_store import NumpyVectorStore

_ROOT = Path(__file__).resolve().parent.parent
_CORPUS_DIR = _ROOT / "eval" / "corpus"
_GOLDEN_SET_PATH = _ROOT / "eval" / "golden_set.json"
_DATA_DIR = _ROOT / "eval" / "_data"  # rebuilt fresh on every run
_RESULTS_DIR = _ROOT / "eval" / "results"

# Written by hand after inspecting the first real run's per-item results
# (see the "By category" table and per-item detail below this section in
# the generated report) — this is deliberately not auto-generated text,
# per FR-3's "record your actual baseline numbers, including the bad
# ones, with interpretation." The numbers are real; this is the human
# read of what they mean.
_INTERPRETATION = """\
## Interpretation

**Retrieval hit-rate (standard): 100%, but groundedness: 78%.** These two
metrics measuring different things is itself the finding. Hit-rate only
checks whether *some* chunk from the right document made the top-5 fused
list; groundedness checks whether the *specific fact* the question asked
about is actually present in those top-5 chunks. `spec.md` is chunked into
12 pieces, and for 6 of 25 standard questions (q03, q05, q07, q13, q19,
q23), a `spec.md` chunk was retrieved but not the one containing the exact
answer — other `spec.md` chunks about unrelated FRs out-ranked it. This is
a direct, quantified cost of using TF-IDF as the embedding stand-in: it
matches on lexical overlap, so a question phrased differently from the
source text's wording (e.g. "how many pull requests" vs. the source's "At
least 8 PRs") competes on weaker signal than a lucky lexical match. A real
semantic embedding model would be expected to close most of this gap;
that's the concrete, measurable case for prioritizing an embedding-model
swap on this line item specifically once a hosted key with quota is
available, over further hand-tuning the offline fallback.

**Out-of-corpus refusal: 0/2 — a real failure, not fixed by threshold
tuning.** Both out-of-corpus questions (Mars mercury boiling point, World
Cup winner) were confidently *answered* instead of refused. A threshold
sweep (0.012 through 0.05) found no value that fixes this without
collateral damage: standard-question refusal-correctness stays at 100%
from 0.012 up to 0.03, then collapses to 40% at 0.035 — the exact point
where out-of-corpus finally starts refusing correctly. There is no gap
between those two regimes; on this corpus, a single scalar RRF-sum
threshold cannot separate "confidently relevant" from "not relevant at
all," because RRF only encodes rank, and with a small corpus even an
irrelevant chunk can rank #1 in one retriever and accumulate enough score
to clear a low bar. The default ships at 0.012 (favoring not refusing
legitimate questions, the more costly failure mode for a corpus this
size) with this limitation stated plainly rather than papered over with a
threshold that would look better on this one subcategory while breaking
real answers. The actual fix is architectural, not a tuned constant: an
absolute similarity-based confidence signal (e.g. a real embedding
model's cosine similarity, which is bounded and semantically meaningful,
unlike RRF's rank-sum) or a re-ranker cross-encoder score, per ADR-002's
"alternatives considered." Tracked in PLAN.md's roadmap.

**Ambiguous-term retrieval: missed at the default top-k.** Both senses of
"sprint" exist in the corpus, but `wellness_notes.md` (1 chunk) only
enters the fused top-10 once `fused_top_k` is widened from 5 to 10 —
`spec.md`'s 12 competing chunks crowd it out by default. Same root cause
as the groundedness gap above (weak lexical embedding, imbalanced chunk
counts across documents), not a separate bug.

**Conflicting sources: passed (1/1), but by a weak proxy.** The system
correctly surfaced citations from both `policy_review_v1.md` and
`policy_review_v2.md` for the pull-request review question. That's the
metric this harness checks — but the system has no actual conflict
*detection*; it doesn't flag to the user that the two cited sources
disagree, it just happens to cite both. Genuine conflict-awareness (e.g.
an LLM-side instruction to explicitly call out disagreement between
cited excerpts) is future work, not implemented.

**Prompt injection: 2/2 leak-free.** Neither injected instruction (both
worded differently) caused the literal system-prompt text to appear in
the answer. This corpus's injection payload is a single unsophisticated
attempt, though, in a 1-chunk document that's easy to keep separate from
the 12-chunk main corpus — it is not a stress test of harder adversarial
phrasing or a payload embedded deep inside a large, otherwise-legitimate
document.
"""


@dataclass
class ItemResult:
    id: str
    category: str
    question: str
    refused: bool
    retrieved_sources: list[str]
    citation_sources: list[str]
    answer_text: str

    retrieval_hit: bool | None = None       # None = not applicable to this item
    groundedness: float | None = None       # None = not applicable
    refusal_correct: bool | None = None     # None = not applicable (injection items)
    leak_free: bool | None = None           # None = not applicable (non-injection items)
    notes: list[str] = field(default_factory=list)


def _fresh_wiring():
    if _DATA_DIR.exists():
        shutil.rmtree(_DATA_DIR)
    _DATA_DIR.mkdir(parents=True)
    repo = SqliteDocumentRepository(str(_DATA_DIR / "eval.db"))
    embedder = TfidfEmbeddingProvider(persist_path=str(_DATA_DIR / "tfidf.pkl"))
    vector_store = NumpyVectorStore(str(_DATA_DIR / "vectors.pkl"))
    keyword_index = BM25KeywordIndex(str(_DATA_DIR / "bm25.pkl"))
    llm = ExtractiveFallbackProvider()
    return repo, embedder, vector_store, keyword_index, llm


def _ingest_corpus(repo, embedder, vector_store, keyword_index) -> None:
    ingest = IngestDocumentUseCase(repo, embedder, vector_store, keyword_index)
    for path in sorted(_CORPUS_DIR.glob("*.md")):
        text = extract_text(path)
        result = ingest.execute(source=path.name, doc_type="md", raw_text=text)
        print(f"  ingested {path.name}: status={result.status.value} chunks={result.chunk_count}")


def _score_item(item: dict, answer_uc: AnswerQueryUseCase) -> ItemResult:
    question = item["question"]
    retrieved = answer_uc.retrieve(question)
    answer = answer_uc.execute(question)

    retrieved_sources = [c.metadata.source for c, _ in retrieved]
    citation_sources = [c.source for c in answer.citations]

    result = ItemResult(
        id=item["id"],
        category=item["category"],
        question=question,
        refused=answer.refused,
        retrieved_sources=retrieved_sources,
        citation_sources=citation_sources,
        answer_text=answer.text,
    )

    expected_sources = item.get("expected_sources") or []
    if expected_sources:
        if len(expected_sources) > 1:
            # Conflicting-sources / ambiguous items: pass requires ALL
            # named sources to be retrievable (both sides of the conflict
            # surfaced), not just one.
            result.retrieval_hit = all(s in retrieved_sources for s in expected_sources)
        else:
            result.retrieval_hit = expected_sources[0] in retrieved_sources

    expected_keywords = item.get("expected_answer_contains") or []
    if expected_keywords and not answer.refused:
        cited_text = " ".join(c_text for c_text, _ in [(c[0].text, c[1]) for c in retrieved]).lower()
        found = sum(1 for kw in expected_keywords if kw.lower() in cited_text)
        result.groundedness = found / len(expected_keywords)

    expected_refusal = item.get("expected_refusal")
    if expected_refusal is not None:
        result.refusal_correct = answer.refused == expected_refusal

    must_not_contain = item.get("must_not_contain") or []
    if must_not_contain:
        lowered = answer.text.lower()
        result.leak_free = not any(phrase.lower() in lowered for phrase in must_not_contain)

    if item.get("note"):
        result.notes.append(item["note"])

    return result


def _aggregate(results: list[ItemResult]) -> dict:
    def _rate(vals: list[bool]) -> float | None:
        return round(sum(vals) / len(vals), 3) if vals else None

    hit_vals = [r.retrieval_hit for r in results if r.retrieval_hit is not None]
    ground_vals = [r.groundedness for r in results if r.groundedness is not None]
    refusal_vals = [r.refusal_correct for r in results if r.refusal_correct is not None]
    leak_vals = [r.leak_free for r in results if r.leak_free is not None]

    by_category: dict[str, dict] = {}
    categories = sorted({r.category for r in results})
    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_hits = [r.retrieval_hit for r in cat_results if r.retrieval_hit is not None]
        cat_ground = [r.groundedness for r in cat_results if r.groundedness is not None]
        cat_refusal = [r.refusal_correct for r in cat_results if r.refusal_correct is not None]
        cat_leak = [r.leak_free for r in cat_results if r.leak_free is not None]
        by_category[cat] = {
            "n": len(cat_results),
            "retrieval_hit_rate": _rate(cat_hits),
            "groundedness": round(sum(cat_ground) / len(cat_ground), 3) if cat_ground else None,
            "refusal_correctness": _rate(cat_refusal),
            "injection_leak_free_rate": _rate(cat_leak),
        }

    return {
        "n_items": len(results),
        "retrieval_hit_rate": _rate(hit_vals),
        "groundedness": round(sum(ground_vals) / len(ground_vals), 3) if ground_vals else None,
        "refusal_correctness": _rate(refusal_vals),
        "injection_leak_free_rate": _rate(leak_vals),
        "by_category": by_category,
    }


def _write_reports(results: list[ItemResult], summary: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = _RESULTS_DIR / "report.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "items": [
                    {
                        "id": r.id,
                        "category": r.category,
                        "question": r.question,
                        "refused": r.refused,
                        "retrieved_sources": r.retrieved_sources,
                        "citation_sources": r.citation_sources,
                        "retrieval_hit": r.retrieval_hit,
                        "groundedness": r.groundedness,
                        "refusal_correct": r.refusal_correct,
                        "leak_free": r.leak_free,
                    }
                    for r in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = ["# FR-3 Evaluation Report", ""]
    lines.append(
        "Generated by `eval/harness.py` against the fully offline adapter "
        "stack (TF-IDF embeddings, BM25 keyword search, extractive-fallback "
        "LLM) — no API key required, reproducible in CI."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Items: {summary['n_items']}")
    lines.append(f"- Retrieval hit-rate: {summary['retrieval_hit_rate']}")
    lines.append(f"- Groundedness: {summary['groundedness']}")
    lines.append(f"- Refusal correctness: {summary['refusal_correctness']}")
    lines.append(f"- Injection leak-free rate: {summary['injection_leak_free_rate']}")
    lines.append("")
    lines.append("## By category")
    lines.append("")
    lines.append("| category | n | hit-rate | groundedness | refusal correctness | leak-free |")
    lines.append("|---|---|---|---|---|---|")
    for cat, stats in summary["by_category"].items():
        lines.append(
            f"| {cat} | {stats['n']} | {stats['retrieval_hit_rate']} | "
            f"{stats['groundedness']} | {stats['refusal_correctness']} | "
            f"{stats['injection_leak_free_rate']} |"
        )
    lines.append("")
    lines.append(_INTERPRETATION)
    lines.append("")
    lines.append("## Per-item detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r.id} ({r.category})")
        lines.append(f"- Question: {r.question}")
        lines.append(f"- Refused: {r.refused}")
        lines.append(f"- Retrieved sources: {r.retrieved_sources}")
        lines.append(f"- Citation sources: {r.citation_sources}")
        if r.retrieval_hit is not None:
            lines.append(f"- Retrieval hit: {r.retrieval_hit}")
        if r.groundedness is not None:
            lines.append(f"- Groundedness: {r.groundedness}")
        if r.refusal_correct is not None:
            lines.append(f"- Refusal correct: {r.refusal_correct}")
        if r.leak_free is not None:
            lines.append(f"- Leak-free: {r.leak_free}")
        lines.append("")

    (_RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    golden_set = json.loads(_GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(golden_set)} golden items")

    print("Ingesting eval corpus...")
    repo, embedder, vector_store, keyword_index, llm = _fresh_wiring()
    _ingest_corpus(repo, embedder, vector_store, keyword_index)

    answer_uc = AnswerQueryUseCase(
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=llm,
        config=RetrievalConfig(refusal_threshold=0.012),
    )

    print("Scoring golden set...")
    results = [_score_item(item, answer_uc) for item in golden_set]
    summary = _aggregate(results)

    _write_reports(results, summary)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nFull report: {_RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()