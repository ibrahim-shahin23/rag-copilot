"""
Item Generator agent (FR-4).

Role: generate assessment items for one module.
Tool set: search_corpus (read-only), draft_item (pure generation). This
agent does NOT hold submit_for_approval — the orchestrator calls that
tool after the validation pass, so even a compromised or misbehaving
agent has no path to the one write-capable tool. Defense in depth, not
just a convention.
Input: one Module at a time (typed contract from Curriculum Designer).
Output: list[AssessmentItem] (typed contract), each still
  validation_passed=False / PENDING — validation and approval happen
  downstream, not inside this agent.
Termination condition: item count reaches target_items_per_module, or the
  module's cited source material is exhausted (no more distinct,
  draftable chunks) — whichever comes first. A chunk that can't be
  drafted into an item (draft_item returns None — no numeric fact to
  mask, see application/tools.py) is skipped, never forced into a bad item.
"""
from __future__ import annotations

from domain.workflow_entities import AssessmentItem, Module
from application.tools import DraftItemTool, SearchCorpusTool


class ItemGeneratorAgent:
    def __init__(
        self,
        search_corpus: SearchCorpusTool,
        draft_item: DraftItemTool,
        target_items_per_module: int = 2,
    ) -> None:
        self._search = search_corpus
        self._draft = draft_item
        self._target = target_items_per_module

    def execute(self, module: Module) -> list[AssessmentItem]:
        query = " ".join(module.gap_names)
        hits = self._search(query, top_k=max(self._target * 3, 5))

        items: list[AssessmentItem] = []
        seen_chunk_ids: set[str] = set()
        for chunk, _score in hits:
            if len(items) >= self._target:
                break
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)

            drafted = self._draft(chunk.text, chunk.metadata.source)
            if drafted is None:
                continue  # source material exhausted for this chunk; try the next one

            items.append(
                AssessmentItem.new(
                    module_id=module.id,
                    question=drafted["question"],
                    options=drafted["options"],
                    correct_option_index=drafted["correct_index"],
                    citation_chunk_id=chunk.id,
                    citation_source=chunk.metadata.source,
                )
            )
        return items