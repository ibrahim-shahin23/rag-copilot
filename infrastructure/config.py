"""
Composition root. This is the ONLY file that should import both domain
ports and concrete infrastructure adapters together, plus wire the
documented fallback chain:

  Embeddings:  HostedEmbeddingProvider  -> (not configured / fails) -> TfidfEmbeddingProvider
  Completion:  GeminiLLMProvider        -> (not configured / fails) -> ExtractiveFallbackProvider

Swapping the LLM provider, embedding model, or vector store is meant to be
"configuration plus one adapter" per the spec's acceptance test — this file
is the configuration half of that.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from domain.ports import DocumentRepository, EmbeddingProvider, KeywordIndex, LLMProvider, VectorStore
from infrastructure.embeddings.hosted_provider import HostedEmbeddingProvider
from infrastructure.embeddings.tfidf_provider import TfidfEmbeddingProvider
from infrastructure.keyword.bm25_index import BM25KeywordIndex
from infrastructure.llm.providers import GeminiLLMProvider, ExtractiveFallbackProvider
from infrastructure.relational.sqlite_repository import SqliteDocumentRepository
from infrastructure.vectorstore.numpy_store import NumpyVectorStore


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

    hosted_embedder = HostedEmbeddingProvider()
    embedder: EmbeddingProvider = (
        hosted_embedder if hosted_embedder.is_configured() else TfidfEmbeddingProvider()
    )

    gemini_llm = GeminiLLMProvider()
    llm: LLMProvider = gemini_llm if gemini_llm.is_configured() else ExtractiveFallbackProvider()

    return Wiring(
        repo=repo,
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=llm,
    )