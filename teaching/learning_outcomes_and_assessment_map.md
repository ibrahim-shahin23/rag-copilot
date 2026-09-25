# Learning Outcomes and Assessment Map

**Course Level:** Post-Graduate (M.Sc. Advanced AI Systems / Applied AI Engineering)  
**Session Title:** Multi-Agent Orchestration & Human-in-the-Loop Safety in Production  
**Duration:** 90 Minutes  

---

## 1. Pedagogical Alignment

| # | Topic Area | Bloom's Taxonomy Level | Measurable Learning Outcome (LO) | Formative Checkpoint (Lab Exercise) | Summative Assessment Criteria |
|---|---|---|---|---|---|
| **LO-1** | **Typed Agent Contracts** | Analyze (Level 4) | Deconstruct unstructured agent text communication and formulate strict, typed Pydantic contracts between specialized roles. | Exercise 1: Inspect request/response schemas in `contracts.py` during execution. | Produces valid schema definitions that catch and reject malformed tool inputs at compile/runtime. |
| **LO-2** | **Supervisor State Machines** | Evaluate (Level 5) | Evaluate failure modes in autonomous loops (recursion, race conditions, drift) and implement a bounded supervisor state machine. | Exercise 4: Intentionally trigger the `MAX_ITERATIONS` breaker. | Designs a state machine preventing infinite delegation cycles while retaining complete execution context. |
| **LO-3** | **Approval Gates & Side-Effects** | Create (Level 6) | Architect an asynchronous approval gate that intercepts side-effecting operations (`publish_assessment_bank`) before execution. | Exercise 2 & 3: Intercept `waiting_approval` state and execute an `edit-and-approve` mutation. | Implements a persistent, non-blocking gate supporting approve, reject, and edit actions with an immutable audit trail. |
| **LO-4** | **Automated Validation & Fallbacks** | Apply (Level 3) | Implement programmatic verification passes (distractor plausibility checks) and configure graceful degradation to plain RAG. | Exercise 4 & Stretch Challenge 1: Trigger automated distractor evaluation and fallback pathways. | Writes resilient handlers that degrade to cached baseline syllabi upon agent timeout or unrecoverable error. |