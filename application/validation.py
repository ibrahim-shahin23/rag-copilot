"""
Automated validation pass (FR-4's "plausible-but-wrong items — automated
validation pass required", education vertical).

Deliberately NOT a fourth specialised agent: it's a deterministic quality
gate the orchestrator runs between item generation and submit_for_approval,
with no tool set, no LLM call, and no role of its own — see PLAN.md's
multi-agent design section for why this distinction matters (FR-4 asks
for >=3 agents with explicit roles; inflating a checklist function into a
fourth "agent" would be padding, not honesty about what the system does).

Checks, all against the canonical chunk text fetched by id (never a
locally-cached copy — see DocumentRepository.find_chunk_by_id):
  1. The correct option appears verbatim in the cited chunk.
  2. No duplicate options (a repeated distractor makes the item unscorable).
  3. correct_option_index is actually in range.
  4. The citation resolves to a real, currently-stored chunk at all.
"""
from __future__ import annotations

from domain.ports import DocumentRepository
from domain.workflow_entities import AssessmentItem


def validate_item(item: AssessmentItem, repo: DocumentRepository) -> AssessmentItem:
    notes: list[str] = []
    passed = True

    chunk = repo.find_chunk_by_id(item.citation_chunk_id)
    if chunk is None:
        passed = False
        notes.append(f"citation chunk_id {item.citation_chunk_id!r} does not resolve to a stored chunk")
    else:
        if not (0 <= item.correct_option_index < len(item.options)):
            passed = False
            notes.append("correct_option_index out of range")
        else:
            correct_text = item.options[item.correct_option_index]
            if correct_text not in chunk.text:
                passed = False
                notes.append(
                    f"correct option {correct_text!r} not found verbatim in cited chunk"
                )

    if len(set(item.options)) != len(item.options):
        passed = False
        notes.append("duplicate options present")

    item.validation_passed = passed
    item.validation_notes = "; ".join(notes) if notes else "all checks passed"
    return item