from domain.entities import Chunk, ChunkMetadata
from domain.workflow_entities import CompetencyGap, CompetencyGapReport, Module
from application.agents.curriculum_designer import CurriculumDesignerAgent
from application.agents.item_generator import ItemGeneratorAgent
from application.agents.standards_mapper import StandardsMapperAgent


def _chunk(cid: str, source: str, text: str) -> Chunk:
    return Chunk(
        id=cid, document_id="d1", text=text,
        metadata=ChunkMetadata(source=source, section=None, position=0,
                                char_start=0, char_end=len(text), version="1"),
    )


# --- StandardsMapperAgent ---

def test_standards_mapper_matches_every_competency_or_flags_unmapped():
    chunk = _chunk("c1", "spec.md", "FR-1 requires at least two input formats.")

    def assess(query):
        if "ingestion" in query:
            return True, chunk
        return False, None  # unmatched competency

    agent = StandardsMapperAgent(assess_competency_match=assess)
    report = agent.execute("RAG Engineer", ["ingestion pipeline", "quantum computing"])

    assert isinstance(report, CompetencyGapReport)
    assert len(report.gaps) == 2  # termination: every input competency produces exactly one gap
    matched = {g.name: g for g in report.gaps}
    assert matched["ingestion pipeline"].matched is True
    assert matched["ingestion pipeline"].citation_chunk_ids == ("c1",)
    assert matched["quantum computing"].matched is False
    assert matched["quantum computing"].citation_chunk_ids == ()


def test_standards_mapper_never_drops_a_competency_even_with_zero_matches():
    agent = StandardsMapperAgent(assess_competency_match=lambda query: (False, None))
    report = agent.execute("Some Role", ["a", "b", "c"])
    assert {g.name for g in report.gaps} == {"a", "b", "c"}
    assert all(not g.matched for g in report.gaps)


# --- CurriculumDesignerAgent ---

def test_curriculum_designer_produces_one_module_per_matched_gap_in_order():
    gap_report = CompetencyGapReport(
        target_role="RAG Engineer",
        gaps=(
            CompetencyGap(name="ingestion", description="d", citation_chunk_ids=("c1",), matched=True),
            CompetencyGap(name="unmapped-thing", description="d", citation_chunk_ids=(), matched=False),
            CompetencyGap(name="retrieval", description="d", citation_chunk_ids=("c2",), matched=True),
        ),
    )
    agent = CurriculumDesignerAgent(
        search_corpus=lambda q, top_k=5: [],
        read_prior_curricula=lambda q, top_k=2: [],
    )
    outline = agent.execute(gap_report)
    assert outline.needs_human_input is False
    assert [m.gap_names[0] for m in outline.modules] == ["ingestion", "retrieval"]
    assert [m.order for m in outline.modules] == [0, 1]


def test_curriculum_designer_flags_needs_human_input_when_nothing_matched():
    gap_report = CompetencyGapReport(
        target_role="Role",
        gaps=(CompetencyGap(name="x", description="d", citation_chunk_ids=(), matched=False),),
    )
    agent = CurriculumDesignerAgent(
        search_corpus=lambda q, top_k=5: [],
        read_prior_curricula=lambda q, top_k=2: [],
    )
    outline = agent.execute(gap_report)
    assert outline.needs_human_input is True
    assert outline.modules == ()
    assert outline.reason is not None


# --- ItemGeneratorAgent ---

def test_item_generator_stops_at_target_count():
    chunks = [_chunk(f"c{i}", "spec.md", f"FR-{i} requires at least {i} things.") for i in range(1, 6)]

    def search(query, top_k=5):
        return [(c, 1.0) for c in chunks][:top_k]

    def draft(text, source):
        return {"question": f"Fill in: {text}", "options": ["1", "2", "3"], "correct_index": 0}

    agent = ItemGeneratorAgent(search_corpus=search, draft_item=draft, target_items_per_module=2)
    module = Module(id="m1", title="Module: x", gap_names=("x",), order=0)
    items = agent.execute(module)
    assert len(items) == 2  # termination: stops at target, doesn't drain all 5 candidates


def test_item_generator_skips_undraftable_chunks_without_forcing_bad_items():
    chunks = [
        _chunk("c1", "spec.md", "No digits here at all."),
        _chunk("c2", "spec.md", "FR-2 requires 25 pairs."),
    ]

    def search(query, top_k=5):
        return [(c, 1.0) for c in chunks]

    def draft(text, source):
        if "digits" in text:
            return None  # can't draft -> must be skipped, not forced
        return {"question": "Fill in", "options": ["25", "26"], "correct_index": 0}

    agent = ItemGeneratorAgent(search_corpus=search, draft_item=draft, target_items_per_module=5)
    module = Module(id="m1", title="Module: x", gap_names=("x",), order=0)
    items = agent.execute(module)
    assert len(items) == 1  # only the draftable chunk produced an item
    assert items[0].citation_chunk_id == "c2"


def test_item_generator_deduplicates_repeated_chunks():
    chunk = _chunk("c1", "spec.md", "FR-1 requires 2 things.")

    def search(query, top_k=5):
        return [(chunk, 1.0), (chunk, 0.9), (chunk, 0.8)]  # same chunk, hypothetically re-ranked

    def draft(text, source):
        return {"question": "Q", "options": ["2", "3"], "correct_index": 0}

    agent = ItemGeneratorAgent(search_corpus=search, draft_item=draft, target_items_per_module=3)
    module = Module(id="m1", title="x", gap_names=("x",), order=0)
    items = agent.execute(module)
    assert len(items) == 1  # deduped, not 3 identical items