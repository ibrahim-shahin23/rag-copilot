"""Minimal CLI surface (FR-7 scoped down to what this slice needs: ingest,
ask-with-citations, and the FR-4/FR-5 multi-agent workflow — run, trace,
and act on the approval gate). A full FastAPI+OpenAPI surface with a
proper UI is still out of scope; see PLAN.md."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.ingest import IngestDocumentUseCase
from application.retrieve import AnswerQueryUseCase
from domain.errors import UnsupportedFormatError
from domain.workflow_entities import ItemApprovalStatus
from infrastructure.config import build_supervisor, build_wiring
from infrastructure.extraction.text_extractor import extract_text


def cmd_ingest(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    use_case = IngestDocumentUseCase(
        repo=wiring.repo,
        embedder=wiring.embedder,
        vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index,
    )
    path = Path(args.file)
    try:
        text = extract_text(path)
    except UnsupportedFormatError as e:
        print(f"error={e}")
        return
    result = use_case.execute(source=path.name, doc_type=path.suffix.lstrip("."), raw_text=text)
    print(f"document_id={result.document_id}")
    print(f"status={result.status.value}")
    print(f"reused_existing={result.reused_existing}")
    print(f"chunk_count={result.chunk_count}")
    if result.error:
        print(f"error={result.error}")


def cmd_ask(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    print(f"llm_provider={wiring.llm.name}")
    use_case = AnswerQueryUseCase(
        embedder=wiring.embedder,
        vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index,
        llm=wiring.llm,
    )
    answer = use_case.execute(args.query)
    print(f"refused={answer.refused}")
    print(f"\n{answer.text}\n")
    if answer.citations:
        print("Citations:")
        for c in answer.citations:
            print(
                f"  - chunk={c.chunk_id[:8]} source={c.source} "
                f"section={c.section!r} position={c.position} score={c.score}"
            )


def cmd_workflow_run(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    supervisor = build_supervisor(wiring)
    competencies = [c.strip() for c in args.competencies.split(",") if c.strip()]
    run = supervisor.run(target_role=args.target_role, competencies=competencies)
    print(f"run_id={run.id}")
    print(f"status={run.status.value}")
    steps = wiring.workflow_repo.get_steps(run.id)
    print(f"steps={len(steps)}")
    for s in steps:
        marker = "OK" if s.status.value == "succeeded" else s.status.value.upper()
        print(f"  [{marker}] step={s.step_index} agent={s.agent_name} attempt={s.attempt}")
        if s.error:
            print(f"         error={s.error}")
    pending = wiring.workflow_repo.list_pending()
    print(f"\nitems_pending_approval={len(pending)}")
    print(f"Use `python cli.py trace {run.id}` for full step detail.")
    print("Use `python cli.py approvals-list` to see items awaiting a decision.")


def cmd_trace(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    run = wiring.workflow_repo.get_run(args.run_id)
    if run is None:
        print(f"error=no run found with id {args.run_id!r}")
        return
    print(f"run_id={run.id}")
    print(f"target_role={run.target_role}")
    print(f"status={run.status.value}")
    print(f"started_at={run.started_at.isoformat()}")
    print(f"finished_at={run.finished_at.isoformat() if run.finished_at else None}")
    print()
    for s in wiring.workflow_repo.get_steps(run.id):
        print(f"--- step {s.step_index} (attempt {s.attempt}) ---")
        print(f"agent:  {s.agent_name}")
        print(f"status: {s.status.value}")
        print(f"input:  {s.input_summary}")
        if s.error:
            print(f"error:  {s.error}")
        else:
            print(f"output: {s.output_summary}")
        print()


def cmd_approvals_list(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    items = wiring.workflow_repo.list_pending()
    if not items:
        print("No items pending approval.")
        return
    for item in items:
        print(f"id={item.id}")
        print(f"  question: {item.question}")
        print(f"  options: {list(item.options)}")
        print(f"  correct_option_index: {item.correct_option_index}")
        print(f"  citation: {item.citation_source} (chunk={item.citation_chunk_id[:8]})")
        print(f"  validation_passed: {item.validation_passed} ({item.validation_notes})")
        print()


def cmd_approvals_decide(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    decision_map = {
        "approve": ItemApprovalStatus.APPROVED,
        "reject": ItemApprovalStatus.REJECTED,
        "edit": ItemApprovalStatus.EDITED_AND_APPROVED,
    }
    decision = decision_map[args.decision]
    if decision == ItemApprovalStatus.EDITED_AND_APPROVED and not args.text:
        print("error=--text is required for an 'edit' decision")
        return
    item = wiring.workflow_repo.decide(
        item_id=args.item_id, decision=decision, decided_by=args.reviewer, approved_text=args.text,
    )
    print(f"item_id={item.id}")
    print(f"approval_status={item.approval_status.value}")
    print(f"decided_by={item.decided_by}")
    print(f"decided_at={item.decided_at.isoformat() if item.decided_at else None}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag-copilot")
    parser.add_argument("--data-dir", default="data")
    sub = parser.add_subparsers(required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("file")
    p_ingest.set_defaults(func=cmd_ingest)

    p_ask = sub.add_parser("ask", help="Ask a grounded question")
    p_ask.add_argument("query")
    p_ask.set_defaults(func=cmd_ask)

    p_run = sub.add_parser("workflow-run", help="Run the curriculum/assessment multi-agent workflow")
    p_run.add_argument("target_role")
    p_run.add_argument(
        "--competencies", required=True,
        help="Comma-separated list of competencies to map against the corpus",
    )
    p_run.set_defaults(func=cmd_workflow_run)

    p_trace = sub.add_parser("trace", help="Show step-by-step detail for a run ID")
    p_trace.add_argument("run_id")
    p_trace.set_defaults(func=cmd_trace)

    p_appr_list = sub.add_parser("approvals-list", help="List items pending approval")
    p_appr_list.set_defaults(func=cmd_approvals_list)

    p_appr_decide = sub.add_parser("approvals-decide", help="Approve, reject, or edit-and-approve an item")
    p_appr_decide.add_argument("item_id")
    p_appr_decide.add_argument("decision", choices=["approve", "reject", "edit"])
    p_appr_decide.add_argument("--reviewer", default="cli-user")
    p_appr_decide.add_argument("--text", default=None, help="Required for 'edit' — the corrected item text")
    p_appr_decide.set_defaults(func=cmd_approvals_decide)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()