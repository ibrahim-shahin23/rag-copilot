# Five Common Trainee Misconceptions in Multi-Agent & RAG Systems

### 1. "Agents should collaborate via free-form natural language text."
* **The Error:** Passing raw text strings between agents (e.g., `"Summarize this and tell the next agent to design a quiz"`).
* **Root Problem:** Causes prompt drift, unparseable downstream keys, silent hallucinations, and breaks automated testing.
* **Correction:** Model all inter-agent boundaries with Pydantic contracts (`TargetRoleContract`, `ModuleOutlineContract`). Validate data strictly at boundary lines.

---

### 2. "Side-effects can be prevented through system prompts alone."
* **The Error:** Instructing the model: *"You are an assistant. Never update the database without asking for confirmation first."*
* **Root Problem:** Prompt injection, long context decay, and non-deterministic model completions consistently bypass text-only guardrails.
* **Correction:** Enforce tool gating in the orchestration code. Classify write tools explicitly; when a write tool is requested, pause the state machine and yield control to an approval endpoint.

---

### 3. "Human-in-the-loop means holding an open HTTP connection."
* **The Error:** Using `time.sleep()`, synchronous locks, or open long-polling sockets waiting for human input.
* **Root Problem:** Gateway timeouts (HTTP 504), connection drops, unmanageable memory leaks, and inability to horizontally scale.
* **Correction:** Use asynchronous state persistence. Snapshot the `RunState` into a datastore with status `waiting_approval` and release the thread. Resume asynchronously when the review endpoint is invoked.

---

### 4. "Fault tolerance is just a top-level `try/except` block."
* **The Error:** Wrapping the entire orchestration loop in a generic `except Exception` that returns a 500 status code or error string.
* **Root Problem:** Wastes tokens on repeated dead-end queries and obscures root causes during production incidents.
* **Correction:** Implement layered control:
  1. Per-step timeouts using `asyncio.wait_for`.
  2. Bounded retries with exponential backoff on transient network faults.
  3. Max-iteration breakers for infinite delegation cycles.
  4. Graceful degradation falling back to a deterministic, cached Plain RAG baseline.

---

### 5. "Distractor generation requires no automated post-validation."
* **The Error:** Assuming an LLM will naturally generate four choices with three plausible yet unambiguously wrong options.
* **Root Problem:** Models often output distractors that are either unintentionally correct, nonsensical, or obvious giveaways (e.g., "All of the above").
* **Correction:** Insert an automated validation pass (`validate_distractors_automated`) that evaluates distractor plausibility and dissimilarity before the item is submitted to the Lead Instructor.

### 6. "Validation failures should cause generated items to be discarded immediately."
* **The Error:** Writing `if not validation_passed: drop_item()`.
* **Root Problem:** Turning a quality gate into a silent drop mechanism turns "catch and flag" into "catch and hide." Instructors never see why items fail, and systematic pipeline bugs remain invisible.
* **The Production Fix:** As implemented in `application/validation.py`, submit **every** generated item to the approval store regardless of pass/fail status. Flag failing items with `validation_passed=False` and attach explicit `validation_notes` so the human reviewer can inspect the error.

---

### 7. "Context manager `with ThreadPoolExecutor()` enforces per-step timeouts."
* **The Error:** Wrapping agent steps in `with ThreadPoolExecutor() as executor: future.result(timeout=3.0)`.
* **Root Problem:** Python's `ThreadPoolExecutor.__exit__` automatically calls `.shutdown(wait=True)`. Even if `future.result()` raises a `TimeoutError` after 3 seconds, the program silently freezes until the hung thread finishes.
* **The Production Fix:** Avoid the context manager for timeouts. Instantiate the executor directly, catch `TimeoutError`, and explicitly invoke `executor.shutdown(wait=False)` to release control immediately.

---

### 8. "Role-based access is just a 'read vs. write' distinction."
* **The Error:** Allowing any authenticated user with write access to approve generated curriculum items.
* **Root Problem:** A content generator (`CONTRIBUTOR`) could approve their own flawed or hallucinated curriculum items, completely bypassing the human-in-the-loop safety design.
* **The Production Fix:** Split roles by **generation vs. oversight** (`CONTRIBUTOR` vs. `REVIEWER`). A `CONTRIBUTOR` can ingest, query, and initiate runs, but receives HTTP 403 on `/approvals/{id}/decide`. A `REVIEWER` inspects traces and approves items, but cannot trigger new generation runs.

---

### 9. "Raw vector search is sufficient for competency mapping."
* **The Error:** Pointing `StandardsMapper` directly at `search_corpus(competency_query)`.
* **Root Problem:** Standard vector search returns the top-k nearest chunks regardless of whether the query is relevant. A nonsensical competency like *"quantum telepathy"* still matches random documents, making the "unmapped competency" code path unreachable.
* **The Production Fix:** Route standards mapping through a relevance/refusal decision gate (`AssessCompetencyMatchTool`), evaluating cosine confidence or explicit model refusal before confirming a match.

---

### 10. "Approval endpoints can take the reviewer's identity from the request body."
* **The Error:** Accepting `{"reviewer": "lead_instructor"}` in the JSON payload of `POST /approvals/{id}/decide`.
* **Root Problem:** Any valid API key holder can impersonate senior instructors by altering the JSON payload, compromising the compliance audit trail.
* **The Production Fix:** Bind `decided_by` strictly to the authenticated identity attached to the `X-API-Key` session in the request header (`request.state.user.user_id`). Ignore user IDs passed in the body.