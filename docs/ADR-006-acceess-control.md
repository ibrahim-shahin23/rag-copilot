# ADR-006: Access Control (FR-8) and HTTP Surface Completion (FR-7)

## Status
Accepted

## Context
FR-7 requires a documented HTTP API (OpenAPI), covering ingest, ask with
citations, running the workflow, acting on the approval gate, and viewing
a trace — plus persistent session history. FR-8 requires authentication
plus at least two roles with genuinely different permissions, enforced
server-side, not by hiding buttons.

## Decision: two roles, split by generation vs. oversight
**CONTRIBUTOR**: ingest, ask, run/cancel the workflow. **REVIEWER**: view
pending approvals, decide on them (approve / reject / edit-and-approve),
and inspect a run's trace. Deliberately non-overlapping rather than a
simple read/write split: a CONTRIBUTOR who could also approve their own
generated items would defeat FR-4's entire human-in-the-loop point (see
PLAN.md §4). A REVIEWER who could also ingest/ask/run would blur the
"this role is oversight, not content generation" boundary this system's
design already leans on elsewhere.

`require_role(*roles)` (`interface/http_api.py`) is a FastAPI dependency
checked *before* an endpoint body runs — the enforcement point is
server-side and structural, not a client-side UI decision to hide a
button. Verified by tests exercising both directions
(`test_reviewer_cannot_call_contributor_endpoints`,
`test_contributor_cannot_call_reviewer_endpoints`), not just one.

## Decision: static, file-backed API keys — not a full identity provider
`StaticUserRepository` (`infrastructure/auth/`) maps API keys to
`(username, role)` via a JSON file, checked via a required `X-API-Key`
header. No password hashing, no token expiry, no OAuth — genuinely out of
scope for what this slice needs to demonstrate, which is that two roles
exist with different, server-enforced permissions. `auth_users.json` is
git-ignored; `auth_users.example.json` (checked in, clearly fake demo
keys) is what the API falls back to with a loud startup warning if the
real file doesn't exist, so the API is usable out-of-the-box in a
sandbox/demo without silently running with no usable credentials — but
unmistakably not a production configuration. A real deployment needs
proper credential storage (hashed, rotated, provider-issued) — tracked in
PLAN.md's roadmap, not solved here.

## Decision: `decided_by` comes from the authenticated user, never the request body
`POST /approvals/{item_id}/decide`'s audit trail (`decided_by`) is always
the resolved `User.username` from the API key, never a client-supplied
field — a REVIEWER cannot claim a decision was made by someone else. This
was a deliberate choice, not an oversight: the request body only carries
`decision` and `edited_text`.

## Decision: session history scope by role, not by explicit query parameter
`GET /sessions` returns only the calling user's own history for
CONTRIBUTOR, but *everyone's* history for REVIEWER — consistent with
REVIEWER already being the oversight/audit role for approvals and traces
elsewhere in this file, rather than introducing a third, separate
"can view all sessions" permission bit.

## Decision: what FR-7's "minimal UI" is, here
FR-7 explicitly allows "web or CLI/TUI." `interface/cli.py` already
covers ingest, ask, running the workflow, and acting on the approval gate
— it just doesn't share the HTTP layer's auth, since it's a local process
talking directly to the same SQLite files, not a remote client. The CLI
and the HTTP API are two separate front doors onto the same application
layer (both ultimately call `IngestDocumentUseCase`, `AnswerQueryUseCase`,
`Supervisor`, etc.) — neither is a stub standing in for the other.

## Real bug found and fixed while building this
Testing `/ingest` over HTTP with a short, realistic single-sentence
document ("FR-9 requires correlation IDs.") returned `status: "failed"`
with `"Document ... produced zero valid chunks"` — not a hypothetical,
the actual response. Root cause in `application/chunking.py`: the
`min_chars` filter (meant to drop a degenerate trailing sliver left over
from windowing a *long* section) was applied unconditionally to every
candidate chunk, including a short section's *only* chunk. A document
short enough to need no windowing at all could still be entirely
discarded if its single chunk happened to be under `min_chars` (default
40) — which a completely ordinary short sentence easily is. Fixed by only
applying the length filter when a section actually produced more than one
window; a section's sole chunk is now always kept regardless of length.
Regression tests: `test_short_document_below_min_chars_still_chunks`,
`test_tiny_trailing_sliver_from_real_windowing_is_still_droppable` (the
second confirms the original degenerate-sliver-dropping behavior still
works when windowing genuinely happens).

This is the second time a "small input" edge case has surfaced a real bug
in this codebase (the first was the small-corpus BM25/refusal-threshold
findings in FR-3's evaluation) — worth noting as a pattern: this system's
test corpora and demo documents have consistently been small, and small
inputs keep finding edges that larger, more typical inputs wouldn't
exercise. Future test additions should keep deliberately including small
inputs, not just realistic-sized ones.

## Consequences / limitations, stated plainly
- No rate limiting, no token expiry, no audit log of *failed* auth
  attempts (only successful, authenticated calls get a SessionEvent) —
  all real gaps for a production deployment, tracked in PLAN.md's
  Security section (§6), not solved here.
- The `X-API-Key` header is sent in plaintext over whatever transport the
  server runs on; this slice doesn't set up TLS termination (that's a
  deployment concern, not application code, but worth stating so it isn't
  assumed handled).
- Two roles is the FR-8 minimum ("at least two"); a real curriculum
  system would likely want more granular roles (e.g. a separate
  "admin"). Not built here — adding one is a new `Role` enum value plus
  updating the relevant `require_role(...)` calls, not a redesign.