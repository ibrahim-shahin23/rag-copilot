"""
Curriculum Designer agent (FR-4).

Role: turn a CompetencyGapReport into an ordered ModuleOutline.
Tool set: search_corpus, read_prior_curricula (both read-only) — this
agent cannot draft assessment items or write anything; it only ever
produces the outline.
Input: CompetencyGapReport (typed contract from Standards Mapper).
Output: ModuleOutline (typed contract).
Termination condition: a Module is produced for every matched gap, in
  order, OR — if there are zero matched gaps to build from — the outline
  is returned with needs_human_input=True and a reason, rather than
  fabricating a module out of nothing. This is the "explicit 'needs human
  input' result if gaps can't be reconciled" case from PLAN.md's design.
"""
from __future__ import annotations

import uuid

from domain.workflow_entities import CompetencyGapReport, Module, ModuleOutline
from application.tools import ReadPriorCurriculaTool, SearchCorpusTool


class CurriculumDesignerAgent:
    def __init__(
        self,
        search_corpus: SearchCorpusTool,
        read_prior_curricula: ReadPriorCurriculaTool,
    ) -> None:
        self._search = search_corpus
        self._read_prior = read_prior_curricula

    def execute(self, gap_report: CompetencyGapReport) -> ModuleOutline:
        matched = [g for g in gap_report.gaps if g.matched]
        if not matched:
            return ModuleOutline(
                target_role=gap_report.target_role,
                modules=(),
                needs_human_input=True,
                reason=(
                    "No competency gaps could be matched to the standards "
                    "corpus — nothing to build a module outline from."
                ),
            )

        # Read-only lookup against prior curricula, purely to demonstrate
        # the restricted second tool being exercised; this slice doesn't
        # yet reorder modules based on what it finds (a real
        # implementation would use overlap with prior modules to avoid
        # duplicating existing coverage — tracked as future work).
        for gap in matched:
            self._read_prior(gap.name, top_k=2)

        modules = tuple(
            Module(id=str(uuid.uuid4()), title=f"Module: {gap.name}", gap_names=(gap.name,), order=i)
            for i, gap in enumerate(matched)
        )
        return ModuleOutline(target_role=gap_report.target_role, modules=modules, needs_human_input=False)