"""
Item Generator agent (FR-4).

Role: Generates assessment questions, correct answer keys, and plausible-but-wrong distractors.
Input: ModuleOutlineContract or Module.
Output: AssessmentDraftContract (questions, options, correct_key, rationale).
Allowed Tools: validate_distractors_automated (quality validation pass).
"""
from __future__ import annotations

from typing import Optional, Union

from domain.workflow_entities import (
    AssessmentDraftContract,
    AssessmentDraftItemContract,
    AssessmentItem,
    Module,
    ModuleOutlineContract,
)
from application.tools import DraftItemTool, SearchCurriculumTemplatesTool, ValidateDistractorsAutomatedTool


class ItemGeneratorAgent:
    def __init__(
        self,
        search_corpus: Union[SearchCurriculumTemplatesTool, callable],
        draft_item: DraftItemTool,
        validate_distractors_automated: Optional[ValidateDistractorsAutomatedTool] = None,
        target_items_per_module: int = 2,
    ) -> None:
        self._search = search_corpus
        self._draft = draft_item
        self._validator = validate_distractors_automated or ValidateDistractorsAutomatedTool()
        self._target = target_items_per_module

    def execute(self, module_input: Union[Module, ModuleOutlineContract]) -> Union[AssessmentDraftContract, list[AssessmentItem]]:
        if isinstance(module_input, ModuleOutlineContract):
            all_drafts = []
            for module in module_input.modules:
                contract = self._generate_for_module(module)
                all_drafts.extend(contract.items)
            m_id = module_input.modules[0].id if module_input.modules else "unknown"
            return AssessmentDraftContract(module_id=m_id, items=all_drafts)
        else:
            draft_contract = self._generate_for_module(module_input)
            return draft_contract.items  # Returns list[AssessmentItem] for backwards compatibility

    def _generate_for_module(self, module: Module) -> AssessmentDraftContract:
        query = " ".join(module.gap_names)
        hits = self._search(query, top_k=max(self._target * 3, 5))

        items: list[AssessmentDraftItemContract] = []
        seen_chunk_ids: set[str] = set()
        for chunk_item in hits:
            chunk = chunk_item[0] if isinstance(chunk_item, (tuple, list)) else chunk_item
            if len(items) >= self._target:
                break
            if chunk.id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk.id)

            drafted = self._draft(chunk.text, chunk.metadata.source)
            if drafted is None:
                continue

            item = AssessmentDraftItemContract.new(
                module_id=module.id,
                question=drafted["question"],
                options=drafted["options"],
                correct_option_index=drafted["correct_index"],
                citation_chunk_id=chunk.id,
                citation_source=chunk.metadata.source,
                rationale=drafted.get("rationale", f"Verified against source {chunk.metadata.source}"),
            )

            # Automated Quality Pass for Distractors
            valid, notes = self._validator(item)
            item.validation_passed = valid
            item.validation_notes = notes

            items.append(item)

        return AssessmentDraftContract(module_id=module.id, items=items)