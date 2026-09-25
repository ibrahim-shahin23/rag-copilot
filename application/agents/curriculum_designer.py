"""
Curriculum Designer agent (FR-4).

Role: Converts competency gaps into structured module outlines and measurable learning objectives.
Input: CompetencyGapsContract.
Output: ModuleOutlineContract (modules with units, learning outcomes, and order).
Allowed Tools: search_curriculum_templates (read-only).
"""
from __future__ import annotations

import uuid
from typing import Union

from domain.workflow_entities import (
    CompetencyGapsContract,
    Module,
    ModuleOutlineContract,
)
from application.tools import SearchCurriculumTemplatesTool


class CurriculumDesignerAgent:
    def __init__(
        self,
        search_curriculum_templates: Union[SearchCurriculumTemplatesTool, callable, None] = None,
        read_prior_curricula: Union[SearchCurriculumTemplatesTool, callable, None] = None,
        search_corpus: Union[SearchCurriculumTemplatesTool, callable, None] = None,
    ) -> None:
        self._search_templates = search_curriculum_templates or search_corpus
        self._read_prior = read_prior_curricula or self._search_templates


    def execute(self, gap_report: CompetencyGapsContract) -> ModuleOutlineContract:
        matched = [g for g in gap_report.gaps if g.matched]
        if not matched:
            return ModuleOutlineContract(
                target_role=gap_report.target_role,
                modules=(),
                needs_human_input=True,
                reason=(
                    "No competency gaps could be matched to the standards "
                    "corpus — nothing to build a module outline from."
                ),
            )

        for gap in matched:
            self._search_templates(gap.name, top_k=2)

        modules = tuple(
            Module(
                id=str(uuid.uuid4()),
                title=f"Module: {gap.name}",
                gap_names=(gap.name,),
                order=i,
                units=[f"Unit 1: Fundamentals of {gap.name}", f"Unit 2: Applied {gap.name}"],
                learning_outcomes=[
                    f"Understand core concepts of {gap.name}",
                    f"Demonstrate practical skill in {gap.name}",
                ],
            )
            for i, gap in enumerate(matched)
        )
        return ModuleOutlineContract(
            target_role=gap_report.target_role,
            modules=modules,
            needs_human_input=False,
        )