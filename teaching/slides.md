---
marp: true
theme: default
paginate: true
header: "CS-8803: Advanced Agentic Systems | Education Copilot Architecture"
footer: "Lecture 08: Multi-Agent Orchestration & Deterministic Safety in Production"
---

# Multi-Agent Orchestration & Deterministic Safety in Production
### Architectural Patterns, Boundary Gating, and Human-in-the-Loop Systems

**Target Audience:** Post-Graduate (M.Sc. / Ph.D.) Software & AI Systems Engineering  
**Duration:** 90 Minutes (Lecture + Case Analysis + Q&A)  
**System Reference:** `rag-copilot` (Clean Architecture, Python 3.12, FastAPI)

---

## Agenda & Session Map (90 Mins)

1. **Foundations & Architecture (20 min)**
   - Why autonomous agent loops fail in production
   - Clean Architecture ring boundaries & provider decoupling
   - Directed Supervisor Pattern vs. Autonomous Swarms
2. **Specialized Agent Design & Contracts (25 min)**
   - Typed Pydantic boundaries vs. string prompt-chaining
   - Deep dive: `StandardsMapper`, `CurriculumDesigner`, `ItemGenerator`
   - Case study: The "Quantum Telepathy" vector search bug
3. **Validation & Human-in-the-Loop Safety (20 min)**
   - Deterministic verification: Catching plausible-but-wrong items
   - Why "catch and flag" beats "catch and hide"
   - Asynchronous approval gates & role-segregated authorization
4. **Concurrency, Resilience & Degradation (15 min)**
   - The silent `ThreadPoolExecutor` shutdown bug
   - Max-iteration breakers, backoff, and graceful fallback to plain RAG
5. **Synthesis & Seminar Discussion (10 min)**

---

## Slide 1: The Production Trap of Autonomous Agents

### The "Demo vs. Reality" Gap
* **Demo Mindset (ReAct / AutoGPT):** Give an LLM an open loop, a list of tools, and hope it decides when to terminate.
* **Failure Modes in Production:**
  * **Non-terminating recursion:** Infinite reasoning loops consuming compute and tokens.
  * **Semantic drift:** Passing unstructured text across 3+ agents leads to cumulative hallucination.
  * **Uncontrolled side-effects:** Agents executing database writes without strict gating.

> **Key Takeaway:** Production multi-agent systems are **finite state machines with supervised transitions**, not unconstrained stochastic conversations.

*Speaker Notes:*  
Point out that the industry is moving away from purely autonomous agents toward deterministic graphs/state machines. In an education vertical where generated assessments impact student evaluation, unconstrained generation is a non-starter.

---

## Slide 2: Clean Architecture for Agentic Systems
```
┌────────────────────────────────────────────────────────┐
│  Interface (FastAPI, CLI, SSE Streaming)               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  Infrastructure (Postgres, SQLite, Gemini, TF-IDF)│  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │  Application (Supervisor, Agents, Use Cases)│  │  │
│  │  │  ┌──────────────────────────────────────┐  │  │  │
│  │  │  │  Domain (Entities, Ports, Contracts) │  │  │  │
│  │  │  └──────────────────────────────────────┘  │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
```
* **Dependency Inversion:** Inner rings (`domain/`, `application/`) **never** import outer rings (`infrastructure/`, `interface/`).
* **Port-and-Adapter Purity:** Swapping an LLM (Gemini → OpenAI) or Vector Store (Numpy → pgvector) touches exactly **one** adapter file. Zero changes to agent logic.

*Speaker Notes:*  
Show the architectural grep test from our repository: `grep -rn "fastapi\|sqlite3\|numpy" domain/ application/` returns zero lines. Emphasize why this makes testing fast, deterministic, and completely offline-capable.

---

## Slide 3: Orchestration Pattern: Directed Supervisor

Why choose a **Supervisor State Machine** over a Pipeline or Peer-to-Peer Swarm?

| Feature | Pipeline | Peer-to-Peer Swarm | Supervisor State Machine (Our Choice) |
|---|---|---|---|
| **Control Flow** | Rigid, Linear | Stochastic, Emergent | **Centralized, Deterministic** |
| **Inspection** | Run-level only | Hard to trace | **Step-by-step audit by Run ID** |
| **Circuit Breaking**| Difficult | Extremely Difficult | **Native (Max iterations, Step timeouts)** |
| **Suspension** | Cannot pause | Unpredictable | **First-class approval gate halt** |

* **The Orchestrator owns the state:** Agents do not know who called them or who comes next. They receive a contract, execute a bounded role, and yield.

*Speaker Notes:*  
Explain that peer-to-peer swarms sound elegant in academic papers but fail compliance audits. A supervisor provides an exact audit ledger where state transitions can be signed and inspected.

---

## Slide 4: Inter-Agent Communication: Typed Contracts

* **The Anti-Pattern:** Free-form string handoffs (`"Here is the outline, please write 3 questions..."`).
* **The Production Standard:** Strictly validated Pydantic contracts.

```python
class CompetencyGapReport(BaseModel):
    target_role: str
    mapped_standards: list[StandardMatch]
    unmapped_competencies: list[str]

class ModuleOutline(BaseModel):
    module_title: str
    target_competencies: list[str]
    source_chunks: list[str]
    needs_human_input: bool = False
```
* **Fail-Fast Boundaries:** If `StandardsMapper` produces malformed JSON, execution halts immediately at the boundary—preventing downstream agents from hallucinating on garbage input.

*Speaker Notes:*  
Ask the room: "What happens when an LLM omits a required citation key in a free-form string?" In our contract model, Pydantic raises a `ValidationError` at the door, triggering a localized retry rather than silent downstream poisoning.

---

## Slide 5: Agent 1: The Standards Mapper

* **Role:** Map a target role and competency list against curriculum standards.
* **Input:** Target role + caller-supplied competency list.
* **Output:** `CompetencyGapReport`.
* **Allowed Tools:** `AssessCompetencyMatchTool` (Read-only).
* **Structural Termination Condition:**  
  A single, deterministic loop over the input list:
  $$\forall c \in \text{Competencies} \implies c \in \text{Mapped} \lor c \in \text{Unmapped}$$
  No unbounded LLM iteration allowed.

*Speaker Notes:*  
Note that taxonomies are product-scale surfaces. In our scoped slice, we test the agent's ability to classify inputs strictly into binary mapped/unmapped categories with verifiable chunk citations.

---

## Slide 6: Case Study: The "Quantum Telepathy" Bug

### The Bug Discovered During Implementation
* **Initial Design:** `StandardsMapper` used raw `search_corpus(competency)` to find matching standards.
* **The Failure:** Vector & lexical search return the top-$k$ nearest neighbors **regardless of relevance**.
* **Observed Result:** Inputting a nonsense competency like `"quantum telepathy"` still returned chunks with positive RRF scores. The `"unmapped"` branch could **never** trigger!

```
[Query: "quantum telepathy"] ──> Vector Search ──> Returns Chunk #4 (Score: 0.018)
                                                        │
                      StandardsMapper asserts MATCH! ◄──┘  (BUG!)
```

*Speaker Notes:*  
This was an authentic bug caught during development. Vector databases do not understand "I don't know"—they calculate distance in high-dimensional space. An agent that relies solely on top-k retrieval is incapable of detecting gap states.

---

## Slide 7: Relevance Gating & Refusal Integration

### The Fix: Integrating Domain Refusal into Tools
* `StandardsMapper` was refactored to use `AssessCompetencyMatchTool`.
* Instead of raw retrieval, it routes through the system's **refusal classification**:
  ```python
  def execute(self, competency: str) -> MatchResult:
      answer = self.answer_use_case.execute(f"Standards for {competency}")
      if answer.is_refusal:
          return MatchResult(matched=False, reason="Out of corpus")
      return MatchResult(matched=True, citation=answer.citations[0])
  ```
* **Architecture Insight:** Agents must inherit the core retrieval engine’s refusal thresholds rather than inventing separate, uncalibrated heuristics.

*Speaker Notes:*  
Highlight how this connects back to FR-3 (Evaluation). The agent now directly inherits the evaluation harness's calibrated refusal capabilities.

---

## Slide 8: Agent 2 & Agent 3: Curriculum & Item Generation

### Agent 2: Curriculum Designer
* **Input:** `CompetencyGapReport`
* **Output:** `ModuleOutline`
* **Termination:** Exactly one module per identified gap, or explicit `needs_human_input=True` if zero gaps matched.

### Agent 3: Item Generator
* **Input:** `ModuleOutline`
* **Output:** `list[AssessmentItem]`
* **Tool Isolation:** Holds `draft_item` and `search_corpus`.  
  **Does NOT hold `submit_for_approval`!**
* **Termination:** Configured item quota reached or corpus exhausted.

*Speaker Notes:*  
Emphasize the isolation principle: Item Generator produces drafts, but the orchestrator mediates submission. If the generator is compromised by an indirect prompt injection in the source text, it lacks the credentials to publish or alter state.

---

## Slide 9: Automated Validation: Plausible-but-Wrong Items

Before any human sees an item, it passes through `application/validation.py`.

### Three Deterministic Invariants:
1. **Canonical Re-fetch:** The correct option must appear verbatim in the source chunk.  
   *Security Note:* It must be re-fetched directly from `DocumentRepository.find_chunk_by_id`, never trust the agent's in-memory copy!
2. **Distinctness:** All distractors must be non-duplicate and distinct from the correct key.
3. **Citation Integrity:** The cited chunk ID must exist in the persistent store.

> **Design Choice:** Offline deterministic checks over an "LLM-as-judge" pass. Fast, free, and regression-testable in CI.

*Speaker Notes:*  
Discuss why we do not use an LLM-as-judge for this first pass. LLMs are poor at detecting exact verbatim string containment and subtle duplicate phrasing, and they add latency/cost. Deterministic Python functions excel at this.

---

## Slide 10: Validation Philosophy: Flag vs. Hide

```
                        [ Generated Assessment Item ]
                                     │
                                     ▼
                        ┌─────────────────────────┐
                        │  Validation Pass Checks  │
                        └────────────┬────────────┘
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
             [ Passed = True ]               [ Passed = False ]
                     │                               │
                     │                        (Add Reason Notes)
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                      [ Enters Approval Store ]
                 (Human Instructor Reviews All Items)
```
* **Anti-Pattern:** Dropping failed items silently (`if not valid: discard()`).  
  *Turns "catch and flag" into "catch and hide"—hiding systematic generation errors.*
* **Production Standard:** Submit every item; flag invalid ones with descriptive error notes.

*Speaker Notes:*  
In mission-critical systems, silent drops obscure systemic degradation. If an LLM suddenly hallucinates citations due to model drift, dropping them hides the incident until the queue is empty. Flagging ensures the instructor sees the degradation immediately.

---

## Slide 11: The Approval Gate: Asynchronous Suspension

* **The Problem:** A human review can take minutes, hours, or days.
* **Naive Architecture:** Keeping an HTTP socket open or holding an execution thread in `time.sleep()`.  
  *(Crashes under connection drops, worker timeouts, or auto-scaling restarts).*
* **Production Architecture:**
  1. The Orchestrator halts execution after validation.
  2. The run state is snapshotted into `SqliteWorkflowRepository` with `status="waiting_approval"`.
  3. The background thread terminates cleanly.
  4. The review endpoint (`POST /approvals/{id}/decide`) acts as a fresh transaction.

*Speaker Notes:*  
Walk through why stateless resumption is mandatory for cloud deployments (e.g., Kubernetes pods restarting). The state must live in durable storage, not process memory.

---

## Slide 12: Access Control: Generation vs. Oversight Roles

**FR-8 Rule:** Roles are split by **generation vs. oversight**, not simple read/write.

| Endpoint | `CONTRIBUTOR` | `REVIEWER` | Architectural Justification |
|---|:---:|:---:|---|
| `POST /ingest` | ✅ | ❌ | Contributors supply corpus documents. |
| `POST /workflow/run` | ✅ | ❌ | Contributors trigger generation jobs. |
| `GET /approvals` | ❌ | ✅ | Oversight must be independent. |
| `POST /approvals/{id}/decide` | ❌ | ✅ | **Prevents self-approval of flawed items.** |
| `GET /sessions` | Own only | All sessions | Audit visibility. |

* **Server-Side Enforcement:** `decided_by` is derived exclusively from the authenticated API token context (`request.state.user`), never accepted from the request body.

*Speaker Notes:*  
Emphasize this key separation of duties: In financial and educational systems, the creator cannot be the approver. If a contributor could approve their own items, the human-in-the-loop safety guarantee is completely defeated.

---

## Slide 13: Mandatory Supervisor Controls (FR-5)

To guarantee termination and stability, the Supervisor enforces 4 guardrails:

```
           [ In-Flight Step ]
                  │
   ┌──────────────┼──────────────┬──────────────┐
   ▼              ▼              ▼              ▼
[ Max-Iter ]   [ Timeout ]   [ Retries ]   [ Degradation ]
  Limit = 6      5.0s hard     Exp. Backoff  Fallback to Plain
  Breaker        Cutoff        Jitter        RAG Extract
```

1. **Max-Iteration Breaker:** Traps infinite agent loops.
2. **Per-Step Timeout:** Kills runaway LLM calls or hanging network sockets.
3. **Retry with Backoff:** Absorbs transient rate limits (HTTP 429).
4. **Graceful Degradation:** Serves cached baseline content when downstream agents fail.

*Speaker Notes:*  
These are not theoretical controls. They are implemented in `application/orchestration/supervisor.py` and covered by end-to-end integration tests with mock failure injectors.

---

## Slide 14: Deep Dive: The Silent ThreadPoolExecutor Bug

### The Code That Looked Correct:
```python
# BROKEN IMPLEMENTATION
with ThreadPoolExecutor() as executor:
    future = executor.submit(agent.execute, payload)
    try:
        return future.result(timeout=5.0)
    except TimeoutError:
        raise StepTimeoutException("Step timed out!")
```

* **The Trap:** When Python exits the `with` block, `executor.__exit__` automatically calls `executor.shutdown(wait=True)`.
* **The Consequence:** The thread pool **silently blocks** until the hanging worker finishes, completely defeating the 5-second timeout!

*Speaker Notes:*  
Spend time here—this is a classic Python concurrency bug. Most developers write `with ThreadPoolExecutor()` assuming the timeout frees the main thread. It doesn't. Wall-clock testing proved the process froze.

---

## Slide 15: Concurrency Bug Solution: Explicit Detachment

### The Production Fix in `application/orchestration/supervisor.py`:
```python
# CORRECT IMPLEMENTATION
executor = ThreadPoolExecutor(max_workers=1)
future = executor.submit(agent.execute, payload)
try:
    result = future.result(timeout=5.0)
    executor.shutdown(wait=False)
    return result
except TimeoutError:
    # Explicitly detach without waiting for the zombie thread
    executor.shutdown(wait=False)
    raise StepTimeoutException("Step timed out!")
```
* **Lesson:** Enforce timeouts with explicit resource detachment (`wait=False`), verified by real wall-clock integration tests (`tests/test_supervisor.py`).

*Speaker Notes:*  
Point out that Python cannot forcibly kill an OS thread. Detaching with `wait=False` lets the application loop continue immediately, leaving the zombie thread to expire harmlessly in the background.

---

## Slide 16: Graceful Degradation to Plain RAG

What happens when Gemini runs out of quota or an agent times out?

```
[ Primary: GeminiLLMProvider ] ──(HTTP 429 Quota Exceeded)──┐
                                                             │
                                                             ▼
                                             [ Fallback: ExtractiveProvider ]
                                                             │
                                                             ▼
                                             [ Verbatim Top-Cited Excerpt ]
```
* **Per-Call Resolution:** Fallback providers check health on every request, not just at boot.
* **Degraded Response Contract:**
  * System continues serving traffic.
  * Emits an explicit notice to `stderr`: `[fallback] degrading to extractive`.
  * Response envelope flags `status="degraded"` for client transparency.

*Speaker Notes:*  
Refer to the system's resilience tests. A free-tier system must anticipate quota exhaustion. Crashing the service with a 500 error is unacceptable; degrading to extractive text maintains operational continuity.

---

## Slide 17: Real-Time Streaming & Cancellation (FR-6)

### Server-Sent Events (SSE) + Cancellation Tokens
* **Why SSE over WebSockets?** Single-direction event streaming (server → client) matches LLM generation with lower protocol overhead.
* **Two Cancellation Vectors:**
  1. **Transport Disconnect:** `request.is_disconnected()` monitors socket drops.
  2. **Explicit Rest API:** `POST /workflow/cancel/{run_id}` sets an in-memory `CancellationToken`.

```python
# Checked at every agent step boundary:
if cancellation_token.is_cancelled:
    yield ProgressEvent(event="run_cancelled", run_id=run_id)
    raise RunCancelledException()
```
* **Integrity:** A cancelled run is handled distinctly from an error—it **never** silently degrades to a completed response.

*Speaker Notes:*  
Explain why browser `EventSource` requires an explicit cancellation REST endpoint: web browsers offer no JavaScript hook to trigger an explicit socket teardown on an existing EventSource connection without closing the page.

---

## Slide 18: Telemetry, Evaluation, & Ground Truth (FR-3)

Continuous regression testing using offline determinism (`eval/harness.py`):

| Evaluation Metric | Baseline Result | Root Cause Analysis & Interpretation |
|---|:---:|---|
| **Retrieval Hit-Rate** | **96.3%** | High recall on standard technical queries via hybrid RRF. |
| **Groundedness** | **78.0%** | Lexical TF-IDF chunk competition; requires neural embeddings. |
| **Refusal Correctness** | **93.1%** | Out-of-corpus queries pass threshold due to RRF score scaling. |
| **Prompt Injection Safety** | **100%** | Strict prompt-chunk separation prevents instruction hijacking. |

> **Principle:** Keep honest baseline limitations documented in code and ADRs; never artificially tune constants to fake a 100% score on a small corpus.

*Speaker Notes:*  
Review the baseline table from `eval/results/report.md`. Point out the 0/2 out-of-corpus failure. Explain that threshold tuning couldn't fix it without breaking standard retrieval, proving that true resolution requires semantic embeddings.

---

## Slide 19: Architectural Checklist for Production Multi-Agent Systems

Before deploying any agentic architecture, verify these 6 gates:

- [ ] **Contract Purity:** Are all agent interfaces typed Pydantic models with no raw string passing?
- [ ] **Sandboxed Side-Effects:** Are write tools structurally excluded from agent tool sets?
- [ ] **Asynchronous Approval:** Can the system pause for human review without holding threads open?
- [ ] **Role Segregation:** Are creator roles prevented from acting as approvers?
- [ ] **Non-Blocking Timeouts:** Are worker threads detached with `wait=False` on timeout?
- [ ] **Deterministic Verification:** Are outputs validated against canonical storage before display?

*Speaker Notes:*  
Offer this checklist as a practical auditing framework students can apply to their own research and industry projects.

---

## Slide 20: Conclusion & Q&A

### Summary of Core Ideas
1. **Agents are State Machines:** Autonomy belongs inside the step; orchestration belongs to a deterministic supervisor.
2. **Safety is Structural:** Rely on code-level gates, schema boundaries, and role-based endpoints—not system prompt requests.
3. **Embrace Degradation:** Design fallback chains so that service quota exhaustion degrades to simpler baselines rather than total system outages.

---

### Open Floor for Seminar Discussion
* **Repository:** `rag-copilot`  
* **Architecture Documentation:** `docs/ADR-004` (Orchestration), `ADR-006` (Access Control)  
* **Lab Assignment:** Open `teaching/lab_sheet.md` to begin the 45-minute practical.
```