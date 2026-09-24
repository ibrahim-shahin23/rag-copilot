"""
Seed script to ingest all evaluation corpus files into the relational/vector data store.

Usage:
    python scripts/seed.py [--data-dir data] [--corpus-dir eval/corpus]
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from application.ingest import IngestDocumentUseCase
from infrastructure.config import build_wiring
from infrastructure.extraction.text_extractor import extract_text


def seed_database(data_dir: str = "data", corpus_dir: str = "eval/corpus") -> None:
    corpus_path = PROJECT_ROOT / corpus_dir if not Path(corpus_dir).is_absolute() else Path(corpus_dir)
    if not corpus_path.exists():
        print(f"[seed] Corpus directory '{corpus_path}' not found.")
        return

    wiring = build_wiring(data_dir)
    use_case = IngestDocumentUseCase(
        repo=wiring.repo,
        embedder=wiring.embedder,
        vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index,
    )

    files = sorted([f for f in corpus_path.glob("*") if f.is_file() and not f.name.startswith(".")])
    print(f"[seed] Found {len(files)} documents in '{corpus_path}'. Beginning ingestion into '{data_dir}'...")

    for file_path in files:
        try:
            text = extract_text(file_path)
            doc_type = file_path.suffix.lstrip(".") or "txt"
            result = use_case.execute(source=file_path.name, doc_type=doc_type, raw_text=text)
            status_str = "reused" if result.reused_existing else result.status.value
            print(f"  [+] Ingested '{file_path.name}': status={status_str}, chunks={result.chunk_count}")
        except Exception as e:
            print(f"  [-] Failed to ingest '{file_path.name}': {e}")

    print("[seed] Database seeding completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed RAG Copilot database with sample corpus")
    parser.add_argument("--data-dir", default="data", help="Directory for database and index storage")
    parser.add_argument("--corpus-dir", default="eval/corpus", help="Directory containing corpus files to ingest")
    args = parser.parse_args()

    seed_database(data_dir=args.data_dir, corpus_dir=args.corpus_dir)
