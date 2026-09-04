"""
Standards Mapper agent (FR-4).

Role: map a target role's required competencies against the ingested
standards corpus.
Tool set: assess_competency_match only (read-only) — this agent has no
reference to any other tool, so it cannot draft items, submit anything,
or read prior-curricula content even if it wanted to.
Input: target role (str) + the list of competencies that role requires.
  Role-to-competency taxonomies are a real product surface on their own;
  out of scope for this slice, so the competency list is supplied by the
  caller rather than derived from anywhere.
Output: CompetencyGapReport (typed contract, domain/workflow_entities.py).
Termination condition: every competency in the input list ends up with
  exactly one CompetencyGap in the output — matched (with a citation) or
  explicitly matched=False ("unmapped"). The loop below makes silently
  dropping one structurally impossible, not just a convention to remember.
"""
from __future__ import annotations

from domain.workflow_entities import CompetencyGap, CompetencyGapReport
from application.tools import AssessCompetencyMatchTool


class StandardsMapperAgent:
    def __init__(self, assess_competency_match: AssessCompetencyMatchTool) -> None:
        self._assess = assess_competency_match

    def execute(self, target_role: str, competencies: list[str]) -> CompetencyGapReport:
        gaps: list[CompetencyGap] = []
        for competency in competencies:
            matched, chunk = self._assess(competency)
            if matched and chunk is not None:
                gaps.append(
                    CompetencyGap(
                        name=competency,
                        description=chunk.text[:200],
                        citation_chunk_ids=(chunk.id,),
                        matched=True,
                    )
                )
            else:
                gaps.append(
                    CompetencyGap(
                        name=competency,
                        description="No matching standard found in the ingested corpus",
                        citation_chunk_ids=(),
                        matched=False,
                    )
                )
        return CompetencyGapReport(target_role=target_role, gaps=tuple(gaps))