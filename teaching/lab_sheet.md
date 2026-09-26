# Hands-On Lab: Multi-Agent Curriculum Pipeline, Validation, & Access Control

**Target Repository:** `rag-copilot`  
**Lab Duration:** 45–50 Minutes  
**Prerequisites:** 
- Repository dependencies installed (`pip install -r requirements.txt`)
- Local server running (`python -m uvicorn interface.http_api:app --reload`)
- Valid test keys copied (`cp auth_users.example.json auth_users.json`)

---

## Exercise 1: Architectural Purity Check
Verify Clean Architecture enforcement before executing logic. The business logic must never import vendor SDKs.

Run:
```bash
grep -rn "sklearn\|rank_bm25\|sqlite3\|numpy\|fastapi\|starlette\|pydantic\|httpx" domain/ application/
Expected Output: No output returned (exit code 0). The core domain and application use cases depend exclusively on internal abstractions.
```
## Exercise 2: Run the Multi-Agent Pipeline as a CONTRIBUTOR
Seed the corpus and initiate the 3-agent pipeline (StandardsMapper → CurriculumDesigner → ItemGenerator).

# Ingest educational specification
```
curl -s -X POST http://127.0.0.1:8000/ingest \
  -H "X-API-Key: contributor-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"source": "spec.md", "doc_type": "md", "raw_text": "FR-2 requires hybrid retrieval with reciprocal rank fusion.", "data_dir": "data"}'
```
# Execute workflow
```
curl -s -X POST http://127.0.0.1:8000/workflow/run \
  -H "X-API-Key: contributor-demo-key" \
  -H "Content-Type: application/json" \
  -d '{
    "target_role": "RAG Engineer",
    "competencies": ["hybrid retrieval"],
    "data_dir": "data"
  }' | jq .
```
Expected Output:
Returns a JSON payload with status: "completed", a valid run_id, and items queued for approval.

## Exercise 3: Inspect Automated Validation & Flagged Items
Verify how application/validation.py handles generated assessment items.
Fetch pending items using the REVIEWER key:
```
curl -s "http://127.0.0.1:8000/approvals?data_dir=data" \
  -H "X-API-Key: reviewer-demo-key" | jq .
```
Expected Verification:
Notice that items contain validation_passed: true (or false) and validation_notes.
Note the architecture design: Failed items are flagged, not silently dropped, ensuring the human reviewer maintains final oversight.

## Exercise 4: Test Strict Access Separation (FR-8)
Demonstrate that generation and oversight roles are mutually exclusive.
Attempt to decide an approval as a CONTRIBUTOR:

EXPORT ITEM_ID="<item-id-from-exercise-3>"
```
curl -s -o /dev/null -w "%{http_code}\n" -X POST "http://127.0.0.1:8000/approvals/$ITEM_ID/decide" \
  -H "X-API-Key: contributor-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"decision": "approve", "data_dir": "data"}'
Expected Output: 403 (Forbidden). Contributors cannot approve their own generated items.
Execute approval with the REVIEWER key:
```
```
curl -s -X POST "http://127.0.0.1:8000/approvals/$ITEM_ID/decide" \
  -H "X-API-Key: reviewer-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"decision": "edit", "approved_text": "Updated answer key with explicit citation", "data_dir": "data"}' | jq .
```
Expected Output: Status returns approved, and decided_by is recorded strictly from the authenticated API token context, never trusted from the client payload.

## Exercise 5: Test Client Cancellation during Real-Time Streaming (FR-6)
Test pipeline cancellation via the explicit endpoint while an SSE stream is active.
# Start a streaming workflow
```
curl -N -X POST http://127.0.0.1:8000/workflow/stream \
  -H "X-API-Key: contributor-demo-key" \
  -H "Content-Type: application/json" \
  -d '{"target_role": "RAG Engineer", "competencies": ["hybrid retrieval"], "data_dir": "data"}' &
```
# In a separate terminal, trigger cancellation
```
curl -X POST http://127.0.0.1:8000/workflow/cancel/<run_id> \
  -H "X-API-Key: contributor-demo-key"
  ```
Expected Output:
The event stream yields event: run_cancelled and aborts prior to executing the next agent step.

## Stretch Challenges
* **Stretch Challenge 1 (Relevance Thresholding in `StandardsMapper`):**  
  In `application/agents/standards_mapper.py`, `AssessCompetencyMatchTool` uses the refusal signal of `AnswerQueryUseCase`. Introduce a dynamic threshold parameter that flags ambiguous queries as `"unmapped"` when cosine distance is below an explicit confidence margin.
* **Stretch Challenge 2 (Deterministic Distractor Dissimilarity Check):**  
  In `application/validation.py`, add a deterministic string-similarity check (e.g., Levenshtein ratio or token overlap) ensuring no distractor option is greater than 85% textually similar to the correct answer.
* **Stretch Challenge 3 (Non-Blocking Supervisor Timeout Fix):**  
  Inspect `application/orchestration/supervisor.py`. Verify why `with ThreadPoolExecutor() as executor:` causes timeouts to block on context exit. Refactor the supervisor worker to use explicit `executor.shutdown(wait=False)` and benchmark response latency under hung worker conditions.