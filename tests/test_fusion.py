from domain.entities import Chunk, ChunkMetadata, Document
from domain.ports import EmbeddingProvider, KeywordIndex, LLMProvider, VectorStore
from application.retrieve import AnswerQueryUseCase, RetrievalConfig


def _chunk(cid: str, text: str, section: str | None = None, position: int = 0) -> Chunk:
    return Chunk(
        id=cid,
        document_id="doc-1",
        text=text,
        metadata=ChunkMetadata(
            source="spec.md", section=section, position=position,
            char_start=0, char_end=len(text), version="1",
        ),
    )


class FakeEmbedder(EmbeddingProvider):
    name = "fake"
    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeVectorStore(VectorStore):
    def __init__(self, hits):
        self._hits = hits
    def upsert(self, chunks, vectors): ...
    def delete_by_document(self, document_id): ...
    def query(self, vector, top_k):
        return self._hits[:top_k]


class FakeKeywordIndex(KeywordIndex):
    def __init__(self, hits):
        self._hits = hits
    def index(self, chunks): ...
    def delete_by_document(self, document_id): ...
    def query(self, text, top_k):
        return self._hits[:top_k]


class FakeLLM(LLMProvider):
    name = "fake-llm"
    def complete(self, system_prompt, user_prompt):
        return "synthesized grounded answer"


def test_agreement_between_dense_and_keyword_wins_fusion():
    a = _chunk("a", "chunk A content")
    b = _chunk("b", "chunk B content")
    c = _chunk("c", "chunk C content")
    # 'a' ranks #1 in both retrievers -> should come out on top after RRF
    dense_hits = [(a, 0.9), (b, 0.5), (c, 0.3)]
    keyword_hits = [(a, 5.0), (c, 2.0), (b, 1.0)]

    use_case = AnswerQueryUseCase(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(dense_hits),
        keyword_index=FakeKeywordIndex(keyword_hits),
        llm=FakeLLM(),
        config=RetrievalConfig(refusal_threshold=0.0),
    )
    answer = use_case.execute("what does chunk A say?")
    assert not answer.refused
    assert answer.citations[0].chunk_id == "a"


def test_metadata_filter_boosts_explicitly_referenced_section():
    fr2 = _chunk("fr2", "Hybrid retrieval with citations.", section="FR-2 Retrieval")
    fr1 = _chunk("fr1", "Extract clean chunk embed index.", section="FR-1 Ingestion")
    # Both retrievers rank them identically -> without the boost it's a tie;
    # with the boost, the query's explicit "FR-2" reference should win.
    dense_hits = [(fr1, 0.5), (fr2, 0.5)]
    keyword_hits = [(fr1, 1.0), (fr2, 1.0)]

    use_case = AnswerQueryUseCase(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(dense_hits),
        keyword_index=FakeKeywordIndex(keyword_hits),
        llm=FakeLLM(),
        config=RetrievalConfig(refusal_threshold=0.0),
    )
    answer = use_case.execute("What does FR-2 require?")
    assert answer.citations[0].chunk_id == "fr2"


def test_low_evidence_query_is_refused_with_no_citations():
    a = _chunk("a", "irrelevant content")
    use_case = AnswerQueryUseCase(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore([(a, 0.01)]),
        keyword_index=FakeKeywordIndex([]),
        llm=FakeLLM(),
        config=RetrievalConfig(refusal_threshold=0.5),  # deliberately high bar
    )
    answer = use_case.execute("something completely unrelated to the corpus")
    assert answer.refused is True
    assert answer.citations == ()


def test_no_hits_at_all_is_refused_not_crashed():
    use_case = AnswerQueryUseCase(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore([]),
        keyword_index=FakeKeywordIndex([]),
        llm=FakeLLM(),
    )
    answer = use_case.execute("anything")
    assert answer.refused is True
