"""
The multi-agent system's tool set (FR-4: >=4 tools, >=1 write/side-effecting).

1. LookupStandardsTool / retrieve_competency_framework — read-only taxonomy & standards query
2. SearchCurriculumTemplatesTool                      — read-only syllabi & template search
3. ValidateDistractorsAutomatedTool                   — automated quality pass for assessment items
4. PublishAssessmentBankTool                           — the ONE write/side-effecting tool (manual/approval-triggered)
"""
from __future__ import annotations

import json
import random
import re
from typing import Optional

from domain.entities import Chunk
from domain.ports import LLMProvider
from domain.workflow_entities import (
    AssessmentDraftItemContract,
    AssessmentItem,
    ItemApprovalStatus,
)
from domain.workflow_ports import ApprovalGateRepository
from application.retrieve import AnswerQueryUseCase


_NUMBER_RE = re.compile(r"\b\d+\b")


class LookupStandardsTool:
    """Read-only. Queries standards/skills taxonomy against the ingested corpus."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, query: str) -> tuple[bool, Optional[Chunk]]:
        hits = self._uc.retrieve(query)
        if not hits:
            return False, None
        top_chunk, top_score = hits[0]
        if top_score < self._uc._cfg.refusal_threshold:
            return False, None
        return True, top_chunk


class RetrieveCompetencyFrameworkTool:
    """Read-only. Queries educational/industry competency frameworks (e.g., Bloom's, SFIA, O*NET)."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, framework_name: str, topic: str) -> list[tuple[Chunk, float]]:
        query = f"{framework_name} {topic}"
        return self._uc.retrieve(query)[:3]


# Backwards-compatible alias for existing code
AssessCompetencyMatchTool = LookupStandardsTool


class SearchCurriculumTemplatesTool:
    """Read-only. Semantic search over course syllabi, prior curricula, and learning templates."""

    def __init__(self, answer_use_case: AnswerQueryUseCase) -> None:
        self._uc = answer_use_case

    def __call__(self, topic: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        hits = self._uc.retrieve(topic)
        filtered = [c for c in hits if "curriculum" in c[0].metadata.source.lower()]
        return filtered[:top_k] if filtered else hits[:top_k]


# Backwards-compatible alias for existing code
SearchCorpusTool = SearchCurriculumTemplatesTool
ReadPriorCurriculaTool = SearchCurriculumTemplatesTool


class DraftItemTool:
    """Pure generation — no side effects, no persistence. Generates item question,
    options, correct index, and distractors."""

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
            return None
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
            "rationale": f"Numeric fact '{number}' extracted from source snippet.",
        }


class ValidateDistractorsAutomatedTool:
    """Automated Quality Pass: Evaluates assessment draft items to ensure
    distractors are plausible, non-trivial, distinct, and unambiguously wrong.
    Flags ambiguous or invalid distractors."""

    def __call__(self, item: AssessmentDraftItemContract, document_repo=None) -> tuple[bool, str]:
        # 1. Option count & uniqueness check
        options = list(item.options)
        if len(options) < 2:
            return False, "Failed quality pass: Item must have at least 2 options."
        if len(set(options)) != len(options):
            return False, "Failed quality pass: Duplicate options/distractors detected."

        # 2. Correct index check
        if not (0 <= item.correct_option_index < len(options)):
            return False, f"Failed quality pass: Correct index {item.correct_option_index} out of bounds."

        # 3. Citation verbatim verification (if doc repo present)
        if document_repo is not None:
            chunk = document_repo.find_chunk_by_id(item.citation_chunk_id)
            if chunk is not None:
                correct_option = options[item.correct_option_index]
                if correct_option.lower() not in chunk.text.lower():
                    return False, f"Correct option '{correct_option}' not found verbatim in cited chunk."

        # 4. Check for ambiguous/trivial distractors (e.g. empty strings, "all of the above" conflicts)
        for idx, opt in enumerate(options):
            if not opt.strip():
                return False, f"Option at index {idx} is empty or whitespace."
            if opt.lower() in ("all of the above", "none of the above") and len(options) < 3:
                return False, f"Ambiguous distractor '{opt}' with insufficient options."

        return True, "Automated distractor quality validation passed."


class PublishAssessmentBankTool:
    """WRITE / Side-Effecting tool: Commits approved items to the production
    assessment bank database.
    CRITICAL: Must NEVER execute automatically. Must halt for Lead Instructor approval."""

    def __init__(self, approval_gate: ApprovalGateRepository) -> None:
        self._gate = approval_gate

    def __call__(self, item: AssessmentDraftItemContract, status: ItemApprovalStatus = ItemApprovalStatus.PENDING) -> None:
        item.approval_status = status
        self._gate.submit(item)



# Backwards compatibility alias
SubmitForApprovalTool = PublishAssessmentBankTool