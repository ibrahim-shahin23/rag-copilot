"""
Composition root. This is the ONLY file that should import both domain
ports and concrete infrastructure adapters together, plus wire the
documented fallback chain:

  Embeddings:  GeminiEmbeddingProvider  -> (fails at call time, any reason) -> TfidfEmbeddingProvider
  Completion:  GeminiLLMProvider        -> (fails at call time, any reason) -> ExtractiveFallbackProvider

Embeddings default to Gemini's free-tier `gemini-embedding-001` rather than
OpenAI's paid endpoint (infrastructure/embeddings/hosted_provider.py is
still there as an alternative, just not wired by default) — both provider
and completion now run off the same free GEMINI_API_KEY, no paid key
needed at all for the default configuration.

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

from domain.ports import DocumentRepository, EmbeddingProvider, KeywordIndex, StreamingLLMProvider, VectorStore
from domain.workflow_ports import ApprovalGateRepository, RunRepository
from infrastructure.embeddings.gemini_embedding_provider import GeminiEmbeddingProvider
from infrastructure.embeddings.tfidf_provider import TfidfEmbeddingProvider
from infrastructure.keyword.bm25_index import BM25KeywordIndex
from infrastructure.llm.providers import GeminiLLMProvider, GemmaLocalLLMProvider, ExtractiveFallbackProvider
from infrastructure.relational.sqlite_repository import SqliteDocumentRepository
from infrastructure.relational.workflow_repository import SqliteWorkflowRepository
from infrastructure.resilience.fallback_providers import FallbackEmbeddingProvider, FallbackStreamingLLMProvider
from infrastructure.vectorstore.numpy_store import NumpyVectorStore

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Wiring:
    repo: DocumentRepository
    embedder: EmbeddingProvider
    vector_store: VectorStore
    keyword_index: KeywordIndex
    llm: StreamingLLMProvider  # a strict superset of LLMProvider (FR-6) — every
                               # existing .complete()-only call site still works unchanged
    workflow_repo: RunRepository  # also implements ApprovalGateRepository


def build_wiring(data_dir: str = "data") -> Wiring:
    repo = SqliteDocumentRepository(f"{data_dir}/copilot.db")
    vector_store = NumpyVectorStore(f"{data_dir}/vectors.pkl")
    keyword_index = BM25KeywordIndex(f"{data_dir}/bm25.pkl")
    workflow_repo = SqliteWorkflowRepository(f"{data_dir}/copilot.db")

    # Wrapped in a runtime fallback rather than chosen once via
    # is_configured(): a hosted provider can be configured (key present)
    # and still fail per-call (quota, rate limit, network) — that must
    # degrade gracefully, not crash. See infrastructure/resilience/.
    embedder: EmbeddingProvider = FallbackEmbeddingProvider(
        primary=GeminiEmbeddingProvider(),
        secondary=TfidfEmbeddingProvider(persist_path=f"{data_dir}/tfidf_vectorizer.pkl"),
    )

    llm: StreamingLLMProvider = FallbackStreamingLLMProvider(
        primary=GeminiLLMProvider(),
        secondary=FallbackLLMProvider(
            primary=GemmaLocalLLMProvider(),
            secondary=ExtractiveFallbackProvider(),
        ),
    )

    return Wiring(
        repo=repo,
        embedder=embedder,
        vector_store=vector_store,
        keyword_index=keyword_index,
        llm=llm,
        workflow_repo=workflow_repo,
    )


def build_supervisor(wiring: Wiring):
    """Assembles the FR-4/FR-5 multi-agent pipeline from an existing
    Wiring. Kept in the composition root, not in interface/cli.py, so the
    CLI stays a thin presentation layer — it only ever imports use cases
    and this function, never infrastructure adapters directly."""
    from application.agents.curriculum_designer import CurriculumDesignerAgent
    from application.agents.item_generator import ItemGeneratorAgent
    from application.agents.standards_mapper import StandardsMapperAgent
    from application.orchestration.supervisor import Supervisor
    from application.retrieve import AnswerQueryUseCase
    from application.tools import AssessCompetencyMatchTool, DraftItemTool, ReadPriorCurriculaTool, SearchCorpusTool, SubmitForApprovalTool

    answer_uc = AnswerQueryUseCase(
        embedder=wiring.embedder, vector_store=wiring.vector_store,
        keyword_index=wiring.keyword_index, llm=wiring.llm,
    )
    search_corpus = SearchCorpusTool(answer_uc)
    assess_competency_match = AssessCompetencyMatchTool(answer_uc)
    read_prior_curricula = ReadPriorCurriculaTool(answer_uc)
    draft_item = DraftItemTool(llm=wiring.llm)
    submit_for_approval = SubmitForApprovalTool(wiring.workflow_repo)

    return Supervisor(
        standards_mapper=StandardsMapperAgent(assess_competency_match),
        curriculum_designer=CurriculumDesignerAgent(search_corpus, read_prior_curricula),
        item_generator=ItemGeneratorAgent(search_corpus, draft_item),
        submit_for_approval=submit_for_approval,
        run_repo=wiring.workflow_repo,
        document_repo=wiring.repo,
        fallback_answer_uc=answer_uc,
    )