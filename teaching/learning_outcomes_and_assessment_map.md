# Learning Outcomes and Assessment Map

**Domain Focus:** Education — Curriculum & Assessment Design Copilot  
**Course Level:** Post-Graduate (M.Sc. Advanced Software & AI Systems)  
**Session Title:** Deterministic Multi-Agent State Machines, Safety Gates, & Clean Architecture  
**Duration:** 90 Minutes  

---

## 1. Pedagogical Alignment

| # | Topic Area | Bloom's Level | Measurable Learning Outcome (LO) | Formative Checkpoint (Lab Exercise) | Summative Assessment Criteria |
|---|---|---|---|---|---|
| **LO-1** | **Clean Architecture & Domain Ports** | Analyze (Level 4) | Critique coupling between agent pipelines and external SDKs; enforce the Clean Architecture dependency rule where `domain/` and `application/` have zero imports from `infrastructure/` or web frameworks. | Exercise 1: Run architectural linter/grep against `domain/` and `application/` to verify zero framework imports. | Explains why domain entities (`workflow_entities.py`) must remain decoupled from specific LLM, vector store, and HTTP drivers. |
| **LO-2** | **Deterministic Agent Boundaries** | Evaluate (Level 5) | Evaluate failure modes in autonomous loops; construct specialized agents with restricted tool sets, typed Pydantic contracts, and explicit structural termination criteria. | Exercise 2: Run workflow via CLI/API and inspect the step boundary contract transitions (`CompetencyGapReport` → `ModuleOutline` → `AssessmentItem`). | Analyzes why `StandardsMapper` must terminate over a fixed competency list rather than relying on unstructured LLM stopping conditions. |
| **LO-3** | **Automated Validation vs. LLM-as-Judge** | Apply (Level 3) | Contrast fast, deterministic, offline-testable verification with non-deterministic LLM-as-judge passes to catch plausible-but-wrong assessment items. | Exercise 3: Inspect `application/validation.py` catching non-verbatim keys or unresolved chunk IDs. | Formulates deterministic rules verifying verbatim key occurrences inside canonically re-fetched chunks (`DocumentRepository.find_chunk_by_id`). |
| **LO-4** | **Human-in-the-Loop & Role Segregation** | Create (Level 6) | Architect a persistent approval gate with strict role segregation, preventing `CONTRIBUTOR` roles from approving items and enforcing `REVIEWER` authentication. | Exercise 4: Attempt cross-role approval (`CONTRIBUTOR` key getting 403 on `/approvals/{id}/decide`). | Implements the triad of decisions (`approve`, `reject`, `edit`) where `decided_by` is derived strictly from server-authenticated context. |
| **LO-5** | **Supervised Resilience & Cancellation** | Evaluate (Level 5) | Implement runtime controls (max-iteration breakers, per-step timeouts with unblocking thread pools, backoff, and SSE cancellation tokens) that degrade gracefully to plain RAG. | Exercise 5 & Stretch Challenges: Disconnect during `/workflow/stream` or induce step timeouts to observe fallback handling. | Solves the thread pool `__exit__` blocking bug and verifies graceful degradation to `ExtractiveFallbackProvider`. |