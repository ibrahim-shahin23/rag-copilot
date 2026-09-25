"""
Tests for the Education Curriculum & Assessment Design Multi-Agent Pipeline (FR-4 & FR-5).
"""
import pytest
from fastapi.testclient import TestClient

from domain.workflow_entities import (
    ApprovalAudit,
    AssessmentDraftContract,
    AssessmentDraftItemContract,
    CompetencyGap,
    CompetencyGapsContract,
    ItemApprovalStatus,
    Module,
    ModuleOutlineContract,
    RunStatus,
    TargetRoleContract,
)
from application.agents.curriculum_designer import CurriculumDesignerAgent
from application.agents.item_generator import ItemGeneratorAgent
from application.agents.standards_mapper import StandardsMapperAgent
from application.orchestration.supervisor import MaxIterationsExceeded, Supervisor, SupervisorConfig
from application.tools import (
    LookupStandardsTool,
    PublishAssessmentBankTool,
    RetrieveCompetencyFrameworkTool,
    SearchCurriculumTemplatesTool,
    ValidateDistractorsAutomatedTool,
)
from interface.http_api import app


def test_pydantic_io_contracts_structure():
    # 1. TargetRoleContract
    role = TargetRoleContract(
        role_name="Senior RAG Architect",
        target_seniority="Senior",
        existing_prerequisites=["Python", "Vector Databases"],
    )
    assert role.role_name == "Senior RAG Architect"

    # 2. CompetencyGapsContract
    gap = CompetencyGap(
        name="Hybrid Search",
        description="Combining dense & sparse retrieval",
        citation_chunk_ids=("chunk_1",),
        matched=True,
        framework_standard="SFIA",
    )
    gap_contract = CompetencyGapsContract(
        target_role=role.role_name,
        mapped_standards=[{"competency": "Hybrid Search", "standard": "SFIA"}],
        gaps=(gap,),
        gap_areas=[],
    )
    assert gap_contract.target_role == "Senior RAG Architect"
    assert len(gap_contract.gaps) == 1

    # 3. ModuleOutlineContract
    module = Module(
        id="mod_1",
        title="Module 1: Advanced Retrieval",
        gap_names=("Hybrid Search",),
        order=0,
        units=["Unit 1: BM25", "Unit 2: Dense Vectors"],
        learning_outcomes=["Implement hybrid search pipeline"],
    )
    outline = ModuleOutlineContract(
        target_role=role.role_name,
        modules=(module,),
        needs_human_input=False,
    )
    assert len(outline.modules) == 1
    assert outline.modules[0].units[0] == "Unit 1: BM25"

    # 4. AssessmentDraftContract
    item = AssessmentDraftItemContract.new(
        module_id="mod_1",
        question="Which score fusion method is reciprocal rank fusion?",
        options=["RRF", "BM25", "TF-IDF", "Linear Sum"],
        correct_option_index=0,
        citation_chunk_id="chunk_1",
        citation_source="spec.md",
        rationale="RRF fuses ranked lists deterministically.",
    )
    draft_contract = AssessmentDraftContract(module_id="mod_1", items=[item])
    assert len(draft_contract.items) == 1
    assert draft_contract.items[0].correct_key == "RRF"


def test_validate_distractors_automated_tool():
    validator = ValidateDistractorsAutomatedTool()

    # Valid item
    valid_item = AssessmentDraftItemContract.new(
        module_id="m1",
        question="What is 2 + 2?",
        options=["4", "3", "5", "6"],
        correct_option_index=0,
        citation_chunk_id="c1",
        citation_source="math.md",
    )
    passed, notes = validator(valid_item)
    assert passed is True
    assert "passed" in notes

    # Item with duplicate distractors
    invalid_item = AssessmentDraftItemContract.new(
        module_id="m1",
        question="What is 2 + 2?",
        options=["4", "3", "3", "6"],
        correct_option_index=0,
        citation_chunk_id="c1",
        citation_source="math.md",
    )
    passed_dup, notes_dup = validator(invalid_item)
    assert passed_dup is False
    assert "Duplicate options" in notes_dup


def test_publish_assessment_bank_tool_does_not_auto_approve():
    class DummyGateRepo:
        def __init__(self):
            self.submitted = []
        def submit(self, item):
            self.submitted.append(item)

    gate = DummyGateRepo()
    tool = PublishAssessmentBankTool(gate)

    item = AssessmentDraftItemContract.new(
        module_id="m1", question="Q?", options=["A", "B"], correct_option_index=0,
        citation_chunk_id="c1", citation_source="s1",
    )
    tool(item, status=ItemApprovalStatus.PENDING)

    assert len(gate.submitted) == 1
    assert gate.submitted[0].approval_status == ItemApprovalStatus.PENDING


def test_max_iterations_breaker_control():
    config = SupervisorConfig(max_iterations=2)
    assert config.max_iterations == 2
