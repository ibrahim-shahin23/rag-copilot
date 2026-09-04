"""
The multi-agent system's tool set (FR-4: >=4 tools, >=1 write/side-effecting).

1. SearchCorpusTool         — read-only
2. ReadPriorCurriculaTool   — read-only, restricted to a source-name filter
3. DraftItemTool            — pure generation, no side effects
4. SubmitForApprovalTool    — the ONE write/side-effecting tool

Each agent below is constructed with only the tool instances its role
needs (see application/agents/*.py) — that's the "restricted tool set"
requirement enforced by construction, not by convention: an agent simply
has no reference to a tool it isn't supposed to use.
"""
from __future__ import annotations

import json
import random
import re
from typing import Optional

from domain.entities import Chunk
from domain.ports import LLMProvider
from domain.workflow_entities import AssessmentItem, ItemApprovalStatus
from domain.workflow_ports import ApprovalGateRepository
from application.retrieve import AnswerQueryUseCase

_NUMBER_RE = re.compile(r"\b\d+\b")


class SearchCorpusTool:
    """Read-only. Wraps AnswerQueryUseCase.retrieve — agents get retrieved
    chunks with citations, never raw write access to the corpus or store."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        return self._uc.retrieve(query)[:top_k]


class AssessCompetencyMatchTool:
    """Read-only, and deliberately NOT built on SearchCorpusTool's raw
    retrieve(). Standards Mapper's job is to decide matched vs. unmapped,
    not just fetch candidates — and retrieve() has no relevance threshold
    at all (by design; see application/retrieve.py's docstring), so a
    first version of this tool built on raw retrieval could never actually
    produce 'unmapped', even for a nonsense competency like 'quantum
    telepathy' — caught in live CLI testing, not simulated. Reusing
    AnswerQueryUseCase.execute()'s refusal decision here means Standards
    Mapper inherits the one relevance signal the rest of this system has
    actually been evaluated against (eval/results/report.md) — including
    its documented limitation (ADR-002: no single threshold cleanly
    separates relevant from irrelevant on a small corpus) rather than
    inventing a second, unvalidated threshold just for this tool."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, query: str) -> tuple[bool, Optional[Chunk]]:
        answer = self._uc.execute(query)
        if answer.refused or not answer.citations:
            return False, None
        top_citation = answer.citations[0]
        for chunk, _score in self._uc.retrieve(query):
            if chunk.id == top_citation.chunk_id:
                return True, chunk
        return False, None  # citation existed but chunk lookup failed — treat as unmatched, not a crash


class ReadPriorCurriculaTool:
    """Read-only, and restricted beyond SearchCorpusTool: only chunks whose
    source name matches the 'curriculum' naming convention are returned,
    so this tool can't be used as a backdoor to pull arbitrary corpus
    content under the 'prior curricula' label."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, topic: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        hits = self._uc.retrieve(topic)
        filtered = [(c, s) for c, s in hits if "curriculum" in c.metadata.source.lower()]
        return filtered[:top_k]


class DraftItemTool:
    """Pure generation — no side effects, no persistence. Tries the
    configured LLM first (a structured JSON prompt); if that fails or
    returns something unusable — the expected outcome on the offline
    extractive-fallback provider, which just echoes excerpt text and
    can't produce JSON — falls back to a deterministic numeric-masking
    drafter, so item generation stays fully testable without an API key.
    Mirrors the same graceful-degradation shape already used for
    embeddings/completion (infrastructure/resilience/)."""

    def __init__(self, llm: LLMProvider, rng_seed: int = 42) -> None:
        self._llm = llm
        self._rng = random.Random(rng_seed)

    def __call__(self, chunk_text: str, source: str) -> Optional[dict]:
        drafted = self._try_llm(chunk_text, source)
        if drafted is not None:
            return drafted
        return self._deterministic_numeric_mask(chunk_text)

    def _try_llm(self, chunk_text: str, source: str) -> Optional[dict]:
        try:
            system = (
                "You write multiple-choice assessment items strictly from "
                "the given excerpt. Output ONLY valid JSON: "
                '{"question": str, "options": [str, str, str, str], '
                '"correct_index": int}. The correct option MUST be a fact '
                "stated verbatim in the excerpt. No prose outside the JSON."
            )
            user = f"Source: {source}\nExcerpt:\n{chunk_text}"
            raw = self._llm.complete(system, user)
            data = json.loads(raw)
            if (
                isinstance(data, dict)
                and isinstance(data.get("question"), str)
                and isinstance(data.get("options"), list)
                and len(data["options"]) >= 2
                and isinstance(data.get("correct_index"), int)
                and 0 <= data["correct_index"] < len(data["options"])
            ):
                return data
        except Exception:
            pass
        return None

    def _deterministic_numeric_mask(self, chunk_text: str) -> Optional[dict]:
        match = _NUMBER_RE.search(chunk_text)
        if not match:
            return None  # no numeric fact to mask -> caller must skip, not force a bad item
        number = int(match.group())
        blanked = (chunk_text[: match.start()] + "____" + chunk_text[match.end():]).strip()
        if len(blanked) > 220:
            blanked = blanked[:220] + "..."

        distractor_pool = sorted({number + 1, number + 2, max(0, number - 1)} - {number})
        offset = 3
        while len(distractor_pool) < 3:
            candidate = number + offset
            if candidate != number and candidate not in distractor_pool:
                distractor_pool.append(candidate)
            offset += 1
        distractors = distractor_pool[:3]

        options = [str(number)] + [str(d) for d in distractors]
        self._rng.shuffle(options)
        correct_index = options.index(str(number))
        return {
            "question": f"Fill in the blank: {blanked}",
            "options": options,
            "correct_index": correct_index,
        }


class SubmitForApprovalTool:
    """The one write/side-effecting tool. Persists a drafted, validated
    item as PENDING in the approval gate. This is a genuine write (new
    state is durably stored), but a PENDING item has no downstream effect
    — nothing reads a pending item as usable. The approval decision
    (approve / reject / edit-and-approve), not this call, is what
    'passing the approval gate' means; this tool is what makes an item
    reachable for that decision in the first place, never what finalizes
    it. See docs/ADR-004-orchestration-pattern.md for this interpretation
    stated explicitly."""

    def __init__(self, approval_gate: ApprovalGateRepository) -> None:
        self._gate = approval_gate

    def __call__(self, item: AssessmentItem) -> None:
        item.approval_status = ItemApprovalStatus.PENDING
        self._gate.submit(item)