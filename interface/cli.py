"""Minimal CLI surface (FR-7 scoped down to what this single feature needs:
ingest + ask-with-citations). A full FastAPI+OpenAPI surface covering the
approval gate and trace view belongs to the multi-agent feature, out of
scope for this slice."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from application.ingest import IngestDocumentUseCase
from application.retrieve import AnswerQueryUseCase
from infrastructure.config import build_wiring


def cmd_ingest(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
    use_case = IngestDocumentUseCase(
        repo=wiring.repo,
        embedder=wiring.embedder,
        vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index,
    )
    path = Path(args.file)
    text = path.read_text(encoding="utf-8")
    result = use_case.execute(source=path.name, doc_type=path.suffix.lstrip("."), raw_text=text)
    print(f"document_id={result.document_id}")
    print(f"status={result.status.value}")
    print(f"reused_existing={result.reused_existing}")
    print(f"chunk_count={result.chunk_count}")
    if result.error:
        print(f"error={result.error}")


def cmd_ask(args: argparse.Namespace) -> None:
    wiring = build_wiring(args.data_dir)
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
