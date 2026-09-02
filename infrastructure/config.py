"""
Composition root. This is the ONLY file that should import both domain
ports and concrete infrastructure adapters together, plus wire the
documented fallback chain:

  Embeddings:  HostedEmbeddingProvider  -> (fails at call time, any reason) -> TfidfEmbeddingProvider
  Completion:  GeminiLLMProvider        -> (fails at call time, any reason) -> ExtractiveFallbackProvider

Both chains are resolved per-call via FallbackEmbeddingProvider /
FallbackLLMProvider (infrastructure/resilience/), not once at startup —
see that module's docstring for why a startup-only check isn't enough.

Swapping the LLM provider, embedding model, or vector store is meant to be
"configuration plus one adapter" per the spec's acceptance test — this file
is the configuration half of that.

.env loading: this previously only read os.environ directly, so a key
placed in a `.env` file (the spec's own `.env.example` convention) was
silently ignored — the fallback chain would trigger and nothing would tell
you why. `load_dotenv` is called against the project root explicitly
(rather than the process's current working directory), because the CLI is
commonly invoked from inside `interface/`, and dotenv's default cwd-search
would miss a `.env` sitting at the repo root in that case.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from domain.ports import DocumentRepository, EmbeddingProvider, KeywordIndex, LLMProvider, VectorStore
from infrastructure.embeddings.hosted_provider import HostedEmbeddingProvider
from infrastructure.embeddings.tfidf_provider import TfidfEmbeddingProvider
from infrastructure.keyword.bm25_index import BM25KeywordIndex
from infrastructure.llm.providers import GeminiLLMProvider, ExtractiveFallbackProvider
from infrastructure.relational.sqlite_repository import SqliteDocumentRepository
from infrastructure.resilience.fallback_providers import FallbackEmbeddingProvider, FallbackLLMProvider
from infrastructure.vectorstore.numpy_store import NumpyVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Wiring:
    repo: DocumentRepository
    embedder: EmbeddingProvider
    vector_store: VectorStore
    keyword_index: KeywordIndex
    llm: LLMProvider


def build_wiring(data_dir: str = "data") -> Wiring:
    repo = SqliteDocumentRepository(f"{data_dir}/copilot.db")
    vector_store = NumpyVectorStore(f"{data_dir}/vectors.pkl")
    keyword_index = BM25KeywordIndex(f"{data_dir}/bm25.pkl")

    # Wrapped in a runtime fallback rather than chosen once via
    # is_configured(): a hosted provider can be configured (key present)
    # and still fail per-call (quota, rate limit, network) — that must
    # degrade gracefully, not crash. See infrastructure/resilience/.
    embedder: EmbeddingProvider = FallbackEmbeddingProvider(
        primary=HostedEmbeddingProvider(),
        secondary=TfidfEmbeddingProvider(persist_path=f"{data_dir}/tfidf_vectorizer.pkl"),
    )

    llm: LLMProvider = FallbackLLMProvider(
        primary=GeminiLLMProvider(),
        secondary=ExtractiveFallbackProvider(),
    )

    return Wiring(
        repo=repo,
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=llm,
    )
