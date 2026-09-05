from domain.entities import Chunk, ChunkMetadata
from domain.ports import EmbeddingProvider, KeywordIndex, LLMProvider, StreamingLLMProvider, VectorStore
from application.retrieve import AnswerQueryUseCase, RetrievalConfig


def _chunk(cid: str, text: str, section: str | None = None) -> Chunk:
    return Chunk(
        id=cid, document_id="doc-1", text=text,
        metadata=ChunkMetadata(source="spec.md", section=section, position=0,
                                char_start=0, char_end=len(text), version="1"),
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


class FakeStreamingLLM(StreamingLLMProvider):
    name = "fake-streaming-llm"
    def complete(self, system_prompt, user_prompt):
        return "".join(self.stream_complete(system_prompt, user_prompt))
    def stream_complete(self, system_prompt, user_prompt):
        yield "streamed "
        yield "grounded "
        yield "answer"


class FakeNonStreamingLLM(LLMProvider):
    name = "fake-non-streaming-llm"
    def complete(self, system_prompt, user_prompt):
        return "answer"


def _use_case(hits, llm, threshold=0.0):
    return AnswerQueryUseCase(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(hits),
        keyword_index=FakeKeywordIndex([]),
        llm=llm,
        config=RetrievalConfig(refusal_threshold=threshold),
    )


def test_streaming_yields_tokens_then_a_done_event_with_citations():
    chunk = _chunk("c1", "some grounded content")
    uc = _use_case([(chunk, 0.9)], FakeStreamingLLM())

    events = list(uc.execute_streaming("a question"))

    token_events = [e for e in events if e.kind == "token"]
    done_events = [e for e in events if e.kind == "done"]
    assert len(done_events) == 1
    assert len(token_events) >= 1
    assert "".join(e.text for e in token_events) == "streamed grounded answer"

    final = done_events[0].answer
    assert final is not None
    assert final.refused is False
    assert final.text == "streamed grounded answer"
    assert len(final.citations) == 1
    assert final.citations[0].chunk_id == "c1"


def test_streaming_refusal_path_yields_a_token_and_done_with_refused_answer():
    uc = _use_case([], FakeStreamingLLM(), threshold=0.5)  # nothing retrieved -> refuse
    events = list(uc.execute_streaming("unanswerable question"))

    assert events[-1].kind == "done"
    final = events[-1].answer
    assert final.refused is True
    assert final.citations == ()
    # the refusal message itself is still delivered via the same event stream
    assert any(e.kind == "token" for e in events)


def test_streaming_raises_clear_error_for_non_streaming_llm_before_yielding_anything():
    chunk = _chunk("c1", "content")
    uc = _use_case([(chunk, 0.9)], FakeNonStreamingLLM())

    gen = uc.execute_streaming("a question")
    try:
        next(gen)
        raised = False
    except TypeError as e:
        raised = True
        assert "StreamingLLMProvider" in str(e)
    assert raised