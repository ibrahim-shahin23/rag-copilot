"""
Standards Mapper agent (FR-4).

Role: Ingests the target professional/academic role and analyzes competency gaps
against educational/industry standards (e.g., Bloom’s taxonomy, SFIA, O*NET).
Input: TargetRoleContract (role_name, target_seniority, existing_prerequisites).
Output: CompetencyGapsContract (target_role, mapped_standards, gaps, gap_areas).
Allowed Tools: lookup_standards, retrieve_competency_framework (read-only).
"""
from __future__ import annotations

from typing import Optional, Union

from domain.workflow_entities import (
    CompetencyGap,
    CompetencyGapsContract,
    TargetRoleContract,
)
from application.tools import LookupStandardsTool, RetrieveCompetencyFrameworkTool


class StandardsMapperAgent:
    def __init__(
        self,
        lookup_standards: Union[LookupStandardsTool, callable, None] = None,
        retrieve_competency_framework: Optional[RetrieveCompetencyFrameworkTool] = None,
        assess_competency_match: Union[LookupStandardsTool, callable, None] = None,
    ) -> None:
        self._lookup = lookup_standards or assess_competency_match
        self._framework = retrieve_competency_framework


    def execute(
        self,
        target_role_input: Union[TargetRoleContract, str],
        competencies: Optional[list[str]] = None,
    ) -> CompetencyGapsContract:
        if isinstance(target_role_input, TargetRoleContract):
            target_role = target_role_input.role_name
            comps = competencies if competencies is not None else target_role_input.existing_prerequisites
        else:
            target_role = target_role_input
            comps = competencies or []

        gaps: list[CompetencyGap] = []
        mapped_standards = []
        gap_areas = []

        for competency in comps:
            matched, chunk = self._lookup(competency)
            if matched and chunk is not None:
                gap = CompetencyGap(
                    name=competency,
                    description=chunk.text[:200],
                    citation_chunk_ids=(chunk.id,),
                    matched=True,
                    framework_standard="SFIA / Bloom's Taxonomy",
                )
                gaps.append(gap)
                mapped_standards.append({"competency": competency, "standard": "Bloom's/SFIA", "chunk_id": chunk.id})
            else:
                gap = CompetencyGap(
                    name=competency,
                    description="No matching standard found in the ingested corpus",
                    citation_chunk_ids=(),
                    matched=False,
                    framework_standard=None,
                )
                gaps.append(gap)
                gap_areas.append(competency)

        return CompetencyGapsContract(
            target_role=target_role,
            mapped_standards=mapped_standards,
            gaps=tuple(gaps),
            gap_areas=gap_areas,
        )