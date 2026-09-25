# Hands-On Lab: Deterministic Multi-Agent State Machines & Safety Gates

**Target Repository:** `rag-copilot`  
**Lab Time:** 45 Minutes  
**Prerequisites:** Docker Compose running (`docker compose up --build`) or local FastAPI server at `http://localhost:8000`.

---

## Exercise 1: Trigger Workflow and Verify Approval Gate Suspension

1. Submit a curriculum and assessment generation request requiring external publication:
   ```bash
   curl -s -X POST http://localhost:8000/orchestration/runs \
     -H "Content-Type: application/json" \
     -d '{"query": "Design an assessment on Docker for Junior DevOps Engineers. Publish to assessment bank."}' | jq .

