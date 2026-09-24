# Security Architecture and Threat Matrix (`SECURITY.md`)

This document maps all security controls implemented across the RAG Copilot codebase to the **OWASP Top 10 Web Application Security Risks** and the **OWASP Top 10 for Large Language Model (LLM) Applications**.

---

## Executive Summary

The RAG Copilot system incorporates a multi-layered security model spanning transport, authentication, authorization, input validation, rate limiting, audit logging, and LLM safety. Server-side controls are enforced structurally across the FastAPI application layer ([`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py)), the relational storage adapters ([`infrastructure/relational/`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/)), and multi-agent workflow pipelines ([`application/orchestration/supervisor.py`](file:///g:/Technical%20assessment/rag-copilot/application/orchestration/supervisor.py)).

---

## 1. OWASP Web Top 10 Threat Mapping & Controls

| OWASP Web Risk | Security Control Implemented | Codebase Mapping | Defense Mechanism |
| :--- | :--- | :--- | :--- |
| **A01: Broken Access Control** | Role-Based Access Control (RBAC) & Object-Ownership Isolation | [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L90-L101)<br>[`infrastructure/auth/static_user_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/auth/static_user_repository.py#L20-L33) | Server-side `require_role(Role.CONTRIBUTOR)` and `require_role(Role.REVIEWER)` dependency enforcement; `/sessions` filters event logs by caller identity (`user.username`) unless caller is `REVIEWER`; `decided_by` in `/approvals/decide` extracts user identity strictly from authenticated token. |
| **A02: Cryptographic Failures** | Secret Management & Transport Security Guidance | [`infrastructure/config.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/config.py#L26-L30)<br>[`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L106-L114) | API keys and secrets stored via `.env` (git-ignored); fallback demo keys trigger loud startup warning; Security Headers include HSTS (`Strict-Transport-Security`) for TLS enforcement. |
| **A03: Injection** | SQL Parameterization & Input Validation Bounds | [`infrastructure/relational/sqlite_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/sqlite_repository.py#L60-L81)<br>[`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L180-L220) | Positional `?` bindings used exclusively in SQLite queries; Pydantic models enforce string length and schema limits (`raw_text`, `query`, `target_role`) preventing payload overflow & buffer injection. |
| **A04: Insecure Design** | Human-in-the-Loop Approval Gate & Non-Overlapping Roles | [`docs/ADR-006-acceess-control.md`](file:///g:/Technical%20assessment/rag-copilot/docs/ADR-006-acceess-control.md#L13-L22)<br>[`application/orchestration/supervisor.py`](file:///g:/Technical%20assessment/rag-copilot/application/orchestration/supervisor.py#L110-L135) | Separate `CONTRIBUTOR` (generation) and `REVIEWER` (oversight) permissions prevent self-approval of generated items; mandatory validation gate checks items before persistence. |
| **A05: Security Misconfiguration** | Defensive Response Headers, CORS Policy & No Unsafe Defaults | [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L56-L105) | Strict CORS allowed origins configured via environment variables; middleware sets `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, and `Content-Security-Policy`. |
| **A06: Vulnerable & Outdated Components** | CI Automated Dependency Scanning & SAST | [`.github/workflows/security-scan.yml`](file:///g:/Technical%20assessment/rag-copilot/.github/workflows/security-scan.yml) | GitHub Actions workflow automates `pip-audit` for dependency vulnerabilities, `bandit` for static Python security scanning, and `gitleaks` for secret detection. |
| **A07: Identification & Auth Failures** | Strict API Key Auth Header Resolution | [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L74-L81) | Mandatory `X-API-Key` validation on all protected endpoints returning `401 Unauthorized` for missing/invalid keys; process-level key lookup. |
| **A08: Software & Data Integrity Failures** | Content Hashing & Document Idempotency | [`domain/entities.py`](file:///g:/Technical%20assessment/rag-copilot/domain/entities.py)<br>[`infrastructure/relational/sqlite_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/sqlite_repository.py#L83-L98) | SHA-256 content hashing (`content_hash`) prevents duplicate/tampered document ingestion; chunk indices tracked with explicit position metadata. |
| **A09: Security Logging & Monitoring Failures** | Secret-Redacted Audit Logging & Session Tracking | [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L168-L185)<br>[`infrastructure/relational/session_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/session_repository.py#L30-L55) | Session events recorded in SQLite (`SessionEvent`); `sanitize_secrets()` regex strips API keys/tokens from request/response summaries before logging. |
| **A10: Server-Side Request Forgery (SSRF)** | Restricted HTTP Client Call Sites & Local Provider Scoping | [`infrastructure/llm/providers.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/llm/providers.py#L185-L195) | LLM and embedding providers use explicit endpoint configurations (`GEMMA_BASE_URL` defaulted to `http://127.0.0.1:1234`); no user-supplied arbitrary URL fetching. |

---

### Detailed Breakdowns for Web Top 10 Requirements

### 1. Broken Access Control: Object-Ownership Checks
- **Enforcement Point**: [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L90-L101) & [`interface/http_api.py:list_sessions`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L480-L495)
- **Control**: Role permissions are split between content generation (`CONTRIBUTOR`) and oversight (`REVIEWER`). When accessing historical audit logs (`GET /sessions`), a `CONTRIBUTOR` is filtered strictly to events matching `user.username`. A `REVIEWER` can inspect all user sessions for compliance monitoring.
- **Audit Tamper Resistance**: In `POST /approvals/{item_id}/decide`, the `decided_by` field is extracted directly from the verified `User` object attached to the request context. Client-supplied identity spoofing is impossible.

### 2. Cryptographic Failures: Data Protection & Encryption Standard
- **Key Storage**: Production secrets (e.g., `GEMINI_API_KEY`) are stored in environment variables loaded via `.env`, which is ignored by version control (`.gitignore`).
- **Demo Credential Safety**: The user repository ([`infrastructure/auth/static_user_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/auth/static_user_repository.py)) falls back to `auth_users.example.json` only when `auth_users.json` is missing, printing a loud warning on startup.
- **Transport Security**: Security headers include `Strict-Transport-Security: max-age=31536000; includeSubDomains` whenever HTTPS requests are served.

### 3. Injection: Parameterized Queries & Validated Uploads
- **SQL Injection Defense**: All relational database operations in [`SqliteDocumentRepository`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/sqlite_repository.py), [`SqliteWorkflowRepository`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/workflow_repository.py), and [`SqliteSessionRepository`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/session_repository.py) use parameter bindings (`?`) exclusively.
- **Validated Input Payload Bounds**: FastAPI endpoint request bodies utilize Pydantic validation rules:
  - `IngestRequest.raw_text`: constrained to `max_length=5_000_000` chars.
  - `AskRequest.query`: constrained to `max_length=10_000` chars.
  - `WorkflowRequest.target_role`: constrained to `max_length=128` chars.
  - `ApprovalDecisionRequest.edited_text`: constrained to `max_length=100_000` chars.

### 4. Rate Limiting & Abuse: Controls & Thresholds
- **Middleware Implementation**: [`interface/http_api.py:rate_limit_middleware`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L70-L100)
- **Sliding-Window Algorithm**: Requests per API Key (or client IP for unauthenticated health checks) are tracked in a 60-second window store.
- **Threshold**: Defaults to 120 requests/minute, configurable via `RATE_LIMIT_PER_MINUTE`.
- **Response Handling**: Exceeding the threshold immediately returns `HTTP 429 Too Many Requests` with standard compliance headers: `Retry-After: 60`, `X-RateLimit-Limit`, and `X-RateLimit-Remaining: 0`.

### 5. Security Misconfiguration: Headers, CORS & Defensive Defaults
- **Security Headers Middleware**: [`interface/http_api.py:add_security_headers`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L101-L114)
  - `X-Content-Type-Options: nosniff` (prevents MIME sniffing)
  - `X-Frame-Options: DENY` (prevents clickjacking)
  - `X-XSS-Protection: 1; mode=block` (enables browser cross-site scripting filter)
  - `Content-Security-Policy: default-src 'self'` (restricts resource origins)
- **CORS Configuration**: Restricts cross-origin requests to explicit origins via `CORS_ALLOWED_ORIGINS` (defaults to `http://localhost:3000,http://127.0.0.1:3000`).

### 6. CI Dependency Scanning: Automation Details
- **Workflow File**: [`.github/workflows/security-scan.yml`](file:///g:/Technical%20assessment/rag-copilot/.github/workflows/security-scan.yml)
- **Automated Triggers**: Runs on every `push` to main, every `pull_request`, and on a weekly cron schedule (`0 0 * * 1`).
- **Scanning Tools**:
  - `pip-audit`: Checks installed Python dependencies against known CVE databases.
  - `bandit`: Performs static application security testing (SAST) for common Python security flaws.
  - `gitleaks`: Scans commit history and codebase for leaked API keys, tokens, and credentials.

### 7. Audit Logging: Security Logs Excluding Secrets
- **Event Recorder**: [`interface/http_api.py:_record_session`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L178-L186)
- **Redaction Logic**: All request and response summaries pass through `sanitize_secrets()` ([`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L168-L177)) before persistence.
- **Regex Patterns**: Detects key-value pairs (e.g. `api_key=...`, `token=...`, `secret=...`, `AIzaSy...`) and replaces secret values with `***REDACTED***` prior to writing to `copilot.db`.

---

## 2. OWASP LLM Top 10 Threat Mapping & Controls

| OWASP LLM Risk | Security Control Implemented | Codebase Mapping | Defense Mechanism |
| :--- | :--- | :--- | :--- |
| **LLM01: Prompt Injection** | Structured Prompt Engineering & System/User Context Isolation | [`application/retrieve.py`](file:///g:/Technical%20assessment/rag-copilot/application/retrieve.py#L40-L75)<br>[`application/tools.py`](file:///g:/Technical%20assessment/rag-copilot/application/tools.py) | System prompts explicitly declare LLM instructions while user queries and retrieved context chunks are inserted into delimited context blocks. |
| **LLM02: Sensitive Information Disclosure** | Secret Redaction & Grounded Citation Scope | [`interface/http_api.py:sanitize_secrets`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L168-L177)<br>[`application/retrieve.py`](file:///g:/Technical%20assessment/rag-copilot/application/retrieve.py) | RAG queries restricted to indexed document corpus; answer generation bounded by strict refusal threshold; secrets sanitized in output logs. |
| **LLM03: Supply Chain Vulnerabilities** | Dependency Pinning & Automated SAST/Audit | [`requirements.txt`](file:///g:/Technical%20assessment/rag-copilot/requirements.txt)<br>[`.github/workflows/security-scan.yml`](file:///g:/Technical%20assessment/rag-copilot/.github/workflows/security-scan.yml) | Dependencies pinned in `requirements.txt`; `pip-audit` runs automatically in CI to catch vulnerable upstream packages. |
| **LLM04: Data and Model Poisoning** | Chunk Quality Filtering & Content Hash Integrity | [`application/chunking.py`](file:///g:/Technical%20assessment/rag-copilot/application/chunking.py)<br>[`infrastructure/relational/sqlite_repository.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/relational/sqlite_repository.py) | Ingested text requires content-hash deduplication; chunking pipeline discards zero-length slivers while enforcing minimum section integrity. |
| **LLM05: Improper Offloading** | Fallback Resilience Architecture | [`infrastructure/resilience/fallback_providers.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/resilience/fallback_providers.py)<br>[`infrastructure/config.py`](file:///g:/Technical%20assessment/rag-copilot/infrastructure/config.py#L80-L90) | Hosted primary providers (`GeminiLLMProvider`, `GeminiEmbeddingProvider`) automatically fall back to local secondary adapters (`ExtractiveFallbackProvider`, `TfidfEmbeddingProvider`) on API error/timeout. |
| **LLM06: Excessive Agency** | Human-in-the-Loop Gate & Restricted Agent Capabilities | [`application/orchestration/supervisor.py`](file:///g:/Technical%20assessment/rag-copilot/application/orchestration/supervisor.py)<br>[`domain/workflow_entities.py`](file:///g:/Technical%20assessment/rag-copilot/domain/workflow_entities.py) | Agent outputs (assessment items) are placed in `PENDING` approval status and cannot be finalized without explicit human `REVIEWER` intervention. |
| **LLM07: System Resource Denial of Service** | Streaming Cancellation & Request Rate Limiting | [`application/orchestration/cancellation.py`](file:///g:/Technical%20assessment/rag-copilot/application/orchestration/cancellation.py)<br>[`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L330-L375) | `CancellationToken` mechanism drains streaming tasks immediately when HTTP clients disconnect, halting wasteful background LLM execution. |
| **LLM08: Vector and Embedding Weaknesses** | Hybrid Fusion Retrieval (Dense + BM25) & Refusal Thresholds | [`application/fusion.py`](file:///g:/Technical%20assessment/rag-copilot/application/fusion.py)<br>[`application/retrieve.py`](file:///g:/Technical%20assessment/rag-copilot/application/retrieve.py) | Combines dense vector search with sparse BM25 keyword matching via Reciprocal Rank Fusion (RRF); refuses out-of-corpus queries when fusion scores fall below threshold. |
| **LLM09: Misinformation** | Mandatory Structured Citations | [`domain/entities.py:Citation`](file:///g:/Technical%20assessment/rag-copilot/domain/entities.py)<br>[`application/retrieve.py`](file:///g:/Technical%20assessment/rag-copilot/application/retrieve.py) | All generated answers MUST return granular citations (source filename, chunk ID, section heading, character position, fusion score) for verifiability. |
| **LLM10: Unbounded Consumption** | Maximum Input Bounds & Request Thresholds | [`interface/http_api.py`](file:///g:/Technical%20assessment/rag-copilot/interface/http_api.py#L180-L220) | Rigid character limits on queries (10k chars) and raw document ingestion (5M chars) prevent memory/compute exhaustion during LLM invocation. |

---

## 3. Verification & Testing

Security controls are verified via automated unit and integration test suites:
- **Security Control Tests**: [`tests/test_security_controls.py`](file:///g:/Technical%20assessment/rag-copilot/tests/test_security_controls.py) (verifies response headers, CORS, secret sanitization, rate limiting, and input validation).
- **HTTP RBAC & Authentication Tests**: [`tests/test_http_api.py`](file:///g:/Technical%20assessment/rag-copilot/tests/test_http_api.py) (verifies role separation, session history isolation, non-spoofable audit logs).
- **Fallback Resilience Tests**: [`tests/test_resilience.py`](file:///g:/Technical%20assessment/rag-copilot/tests/test_resilience.py) (verifies provider degradation without crashing).
- **CI Pipeline**: [`.github/workflows/security-scan.yml`](file:///g:/Technical%20assessment/rag-copilot/.github/workflows/security-scan.yml) (automates dependency vulnerability checks and code security analysis).
