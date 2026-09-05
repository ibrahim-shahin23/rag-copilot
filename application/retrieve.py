"""
AnswerQueryUseCase — application layer.

FR-2 requirements implemented here:
  - Hybrid retrieval (dense + keyword) fused via Reciprocal Rank Fusion (RRF).
    RRF is chosen over score-normalization fusion because dense cosine
    similarity and BM25 scores live on incomparable scales; RRF only needs
    each retriever's *ranking*, which sidesteps that problem entirely
    (see docs/ADR-002-retrieval-fusion.md).
  - Chosen enhancement: metadata filtering. This corpus is clause/section
    structured (FR-2, ADR-001, "Section 4", ...). When a query explicitly
    names a section, we boost (not strictly filter, to stay recall-safe)
    chunks whose ChunkMetadata.section matches, ranking them above
    equally-scored competitors.
  - Correct refusal on low-evidence questions: if the fused top score is
    below `refusal_threshold`, the use case returns a refusal Answer with
    zero citations rather than letting the LLM hallucinate an answer.
  - Citations are structured and traceable to the exact chunk (chunk id,
    source, section, position) — never a bare string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

from domain.entities import Answer, AnswerStreamEvent, Chunk, Citation
from domain.ports import EmbeddingProvider, KeywordIndex, LLMProvider, StreamingLLMProvider, VectorStore

_SECTION_REF_RE = re.compile(r"\b(?:FR|ADR|NFR)-\d+\b|\bsection\s+\d+(?:\.\d+)*\b", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a grounded document copilot. Answer ONLY using the provided "
    "excerpts. Every factual claim must be attributable to one of the "
    "numbered excerpts. If the excerpts do not contain the answer, say so "
    "plainly. Do not use outside knowledge."
)


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 10
    keyword_top_k: int = 10
    fused_top_k: int = 5
    rrf_k: int = 60                 # standard RRF damping constant
    section_boost: float = 0.02     # small nudge, must not override real relevance
    refusal_threshold: float = 0.012  # fused RRF score floor; see tests for calibration


class AnswerQueryUseCase:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        llm: LLMProvider,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._keyword_index = keyword_index
        self._llm = llm
        self._cfg = config or RetrievalConfig()

    def _fuse(
        self,
        query: str,
        dense_hits: list[tuple[Chunk, float]],
        keyword_hits: list[tuple[Chunk, float]],
    ) -> list[tuple[Chunk, float]]:
        cfg = self._cfg
        referenced_sections = {m.group().upper() for m in _SECTION_REF_RE.finditer(query)}

        scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}

        for rank, (chunk, _score) in enumerate(dense_hits, start=1):
            chunks_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (cfg.rrf_k + rank)

        for rank, (chunk, _score) in enumerate(keyword_hits, start=1):
            chunks_by_id[chunk.id] = chunk
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (cfg.rrf_k + rank)

        # Metadata-filtering enhancement: boost chunks whose section matches
        # an explicit section reference in the query.
        if referenced_sections:
            for cid, chunk in chunks_by_id.items():
                section = (chunk.metadata.section or "").upper()
                if any(ref in section for ref in referenced_sections):
                    scores[cid] += cfg.section_boost

        ranked = sorted(chunks_by_id.values(), key=lambda c: scores[c.id], reverse=True)
        return [(c, scores[c.id]) for c in ranked[: cfg.fused_top_k]]

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Hybrid retrieval + fusion, exposed separately from execute() so
        callers (notably the FR-3 evaluation harness, eval/harness.py) can
        measure retrieval quality — did the right chunk get retrieved at
        all? — independently of whether the refusal threshold then chose
        to answer. Conflating the two would make it impossible to tell
        "retrieval failed" apart from "retrieval succeeded but we correctly
        declined," which are very different failure modes to score."""
        cfg = self._cfg
        [query_vector] = self._embedder.embed([query])
        dense_hits = self._vector_store.query(query_vector, top_k=cfg.dense_top_k)
        keyword_hits = self._keyword_index.query(query, top_k=cfg.keyword_top_k)
        return self._fuse(query, dense_hits, keyword_hits)

    def execute(self, query: str) -> Answer:
        cfg = self._cfg
        fused = self.retrieve(query)

        if not fused or fused[0][1] < cfg.refusal_threshold:
            return Answer(
                query=query,
                text=(
                    "I don't have enough evidence in the ingested corpus to "
                    "answer that confidently, so I'm declining rather than "
                    "guessing. Try rephrasing, or confirm the relevant "
                    "document has been ingested."
                ),
                citations=(),
                refused=True,
            )

        excerpts_block = "\n\n".join(
            f"[{i+1}] (source={chunk.metadata.source}, "
            f"section={chunk.metadata.section or 'n/a'}): {chunk.text}"
            for i, (chunk, _score) in enumerate(fused)
        )
        user_prompt = f"Question: {query}\n\nExcerpts:\n{excerpts_block}"
        answer_text = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

        citations = tuple(
            Citation(
                chunk_id=chunk.id,
                source=chunk.metadata.source,
                section=chunk.metadata.section,
                position=chunk.metadata.position,
                score=round(score, 6),
            )
            for chunk, score in fused
        )
        return Answer(query=query, text=answer_text, citations=citations, refused=False)

    def execute_streaming(self, query: str) -> Iterator[AnswerStreamEvent]:
        """FR-6 streaming counterpart to execute(). Retrieval and the
        refusal decision happen synchronously first, exactly as in
        execute() — citations and the refuse-or-answer call both depend
        on the fully-fused retrieval result, so there's nothing to stream
        incrementally about them. Only the LLM's answer text is actually
        streamed, token-by-token (provider-dependent chunk granularity),
        via a StreamingLLMProvider. Requires the configured llm to
        support streaming; raises a clear TypeError immediately (before
        yielding anything) if it doesn't, rather than failing confusingly
        partway through."""
        cfg = self._cfg
        fused = self.retrieve(query)

        if not fused or fused[0][1] < cfg.refusal_threshold:
            answer = Answer(
                query=query,
                text=(
                    "I don't have enough evidence in the ingested corpus to "
                    "answer that confidently, so I'm declining rather than "
                    "guessing. Try rephrasing, or confirm the relevant "
                    "document has been ingested."
                ),
                citations=(),
                refused=True,
            )
            yield AnswerStreamEvent(kind="token", text=answer.text)
            yield AnswerStreamEvent(kind="done", text="", answer=answer)
            return

        if not isinstance(self._llm, StreamingLLMProvider):
            raise TypeError(
                f"execute_streaming() requires a StreamingLLMProvider; "
                f"{type(self._llm).__name__} only implements LLMProvider.complete(). "
                f"Use execute() instead, or configure a streaming-capable provider."
            )

        excerpts_block = "\n\n".join(
            f"[{i+1}] (source={chunk.metadata.source}, "
            f"section={chunk.metadata.section or 'n/a'}): {chunk.text}"
            for i, (chunk, _score) in enumerate(fused)
        )
        user_prompt = f"Question: {query}\n\nExcerpts:\n{excerpts_block}"

        full_text_parts: list[str] = []
        for token_chunk in self._llm.stream_complete(_SYSTEM_PROMPT, user_prompt):
            full_text_parts.append(token_chunk)
            yield AnswerStreamEvent(kind="token", text=token_chunk)

        citations = tuple(
            Citation(
                chunk_id=chunk.id,
                source=chunk.metadata.source,
                section=chunk.metadata.section,
                position=chunk.metadata.position,
                score=round(score, 6),
            )
            for chunk, score in fused
        )
        answer = Answer(
            query=query, text="".join(full_text_parts), citations=citations, refused=False,
        )
        yield AnswerStreamEvent(kind="done", text="", answer=answer)