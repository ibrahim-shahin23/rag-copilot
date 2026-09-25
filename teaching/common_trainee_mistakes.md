---

### File 4: `teaching/common_trainee_mistakes.md`

```markdown
# Five Common Trainee Misconceptions in Multi-Agent Engineering

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