FR-1 Ingestion. Support at least two input formats. Separable, testable stages:
extract, clean, chunk, embed, index, with document and chunk metadata (source,
section, page/clause, version). Idempotent re-ingestion. Per-document status
and failure reporting.

FR-2 Retrieval. Chunking strategy must be a deliberate, documented decision
justified against your document structure. Hybrid retrieval (dense + keyword)
with a documented fusion method. One justified enhancement: re-ranking, query
rewriting, HyDE, multi-query, metadata filtering or contextual retrieval.
Citations mandatory — structured, traceable to the exact chunk. Correct
refusal on low-evidence questions.

FR-3 Evaluation. A golden set of at least 25 Q/A pairs including at least 5
adversarial cases (out-of-corpus, ambiguous, prompt injection, conflicting
sources), and a runnable harness reporting retrieval hit-rate, groundedness
and refusal correctness. Record your actual baseline numbers, including the
bad ones, with interpretation.

FR-4 Multi-agent. At least three specialised agents plus an orchestrator,
each with an explicit role, a restricted tool set, defined input/output and
a termination condition. At least four tools, of which at least one is
write or side-effecting — never executed without passing the approval gate.
Agents communicate through typed contracts, not free-form text.

FR-5 Orchestration. A named, justified pattern: supervisor, planner-executor,
pipeline, or state machine. Mandatory controls: max-iteration breaker,
per-step timeout, retry with backoff, graceful degradation to plain RAG.
Every run inspectable step-by-step by run ID. Approval gate supports
approve, reject, and edit-and-approve, all audited.

FR-6 Real-time. Token-level streaming over SSE or WebSocket, live agent
progress events rather than a frozen spinner, and client cancellation that
actually stops server-side work.

FR-7 Surface. Documented HTTP API using OpenAPI. A minimal working UI —
web or CLI/TUI — covering ingest, ask with citations, run the workflow,
act on the approval gate, and view a trace. Persistent session history.

FR-8 Access. Authentication plus at least two roles with genuinely
different permissions, enforced server-side, not by hiding buttons.

FR-9 Observability. Correlation ID flowing from request through
orchestrator, agent, and LLM call. Per-request token and cost accounting,
persisted and queryable. LLM tracing via a self-hosted tool, OpenTelemetry,
or a clean custom trace store. Health and readiness endpoints.

4. Architecture and Engineering. Clean Architecture is mandatory. Clean,
Hexagonal, Onion, or Vertical Slice — name and justify your choice. Your
domain and application layers must not depend on any LLM SDK, vector-store
SDK, or web framework. Provider abstraction is mandatory: one interface
covering completion, streaming, tool calling, and embeddings, with at
least two working implementations, selected by configuration, with a
documented fallback chain. Dependency injection is required throughout.
At least four ADRs are required, covering chunking and retrieval,
orchestration pattern, vector store choice, and your twist's central
decision.

5. Security. Document every control against the threat it addresses in
docs/SECURITY.md. Cover the OWASP Web Top 10 and the OWASP LLM Top 10,
including prompt injection (direct and indirect via ingested documents),
insecure output handling, sensitive information disclosure, excessive
agency, unbounded consumption, and supply chain risk. Secrets must never
appear in the repository, including in history.

6. Engineering Process. At least 30 meaningful commits across at least 6
distinct days. All work merges via Pull Request, even solo. At least 8
PRs with real descriptions, each self-reviewed with inline comments.
GitHub Issues linked to PRs. GitHub Actions CI on every PR: build, lint,
tests, dependency and secret scanning, green on main at submission.

Multi-tenancy. At least two tenants with fully isolated corpora, users,
and runs; isolation enforced at the data layer; a test proving cross-tenant
leakage is impossible.