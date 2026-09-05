# ADR-005: Streaming Transport & Cancellation (FR-6)

## Status
Accepted

## Context
FR-6 requires token-level streaming (SSE or WebSocket), live agent
progress events (not a frozen spinner), and client cancellation that
actually stops server-side work.

## Decision: SSE over WebSocket
Both directions of this system's real-time traffic (answer tokens, agent
progress events) are server-to-client only — the client never needs to
push data mid-stream beyond an initial request and an optional cancel.
SSE is the simpler protocol for that shape: plain HTTP, no separate
upgrade handshake, and browsers' native `EventSource` handles reconnection
for free. WebSocket would earn its complexity if the client needed to
send data *during* the stream (e.g. adjusting parameters live); nothing
here does.

## Decision: a new StreamingLLMProvider port, not a new method on LLMProvider
Adding `stream_complete()` as a required abstract method on the existing
`LLMProvider` port would have broken every test fake across the suite
that implements only `complete()` (there are many, going back to
`tests/test_fusion.py`), for no benefit to code that never needs
streaming. `StreamingLLMProvider(LLMProvider)` is a separate ABC that
`GeminiLLMProvider` and `ExtractiveFallbackProvider` both additionally
implement — code that needs streaming depends on the narrower type; code
that doesn't is untouched. `infrastructure/config.py`'s `Wiring.llm` field
was upgraded to `StreamingLLMProvider` (a strict superset), which is why
every existing `.complete()` call site kept working unchanged when this
landed — confirmed by the full test suite passing without modification
to any non-streaming test.

## Decision: run_streaming() is the one implementation; run() wraps it
`Supervisor.run()` used to contain the entire pipeline. It's now a thin
loop that drains `run_streaming()` and fetches the final `Run` from the
repository. This was a real refactor with real regression risk — pinned
down by running the full existing `test_supervisor.py` suite unchanged
immediately after, before writing a single new streaming test. There is
now exactly one pipeline implementation; a streamed and non-streamed
version can't quietly drift apart from each other over time.

## Decision: cancellation is a step-boundary check, not step-level preemption
`CancellationToken.is_cancelled()` is checked at the top of
`_run_step_streaming()`, before a step starts — never mid-step. This is
consistent with, and inherits, the same limitation ADR-004 already
documented for per-step timeouts: Python cannot forcibly kill a running
thread, so genuine mid-step preemption isn't available without
process-level isolation. What IS real and tested: once cancellation is
observed, the orchestrator will not start the *next* step. For the actual
tool calls this system makes — corpus search, deterministic/LLM item
drafting, a local DB write — steps are short, so cancellation lands
within roughly one step's duration in practice, not instantly. Stated
plainly rather than oversold.

Cancellation is handled as its own exception path (`RunCancelled`),
deliberately separate from the generic-failure path that triggers
graceful degradation. A cancelled run must never silently become a
degraded-but-still-answered run — that would substitute an answer the
client never asked for in place of the stop they did ask for. Tested
directly: `test_cancellation_takes_precedence_over_degradation`.

## Decision: two independent cancellation mechanisms at the HTTP layer
1. **Connection-close detection** (`await request.is_disconnected()`,
   polled between forwarded events) — how a browser's `EventSource.close()`
   or an aborted `fetch()` naturally cancels a stream, with zero extra
   client-side code.
2. **An explicit `POST /workflow/cancel/{run_id}`** — necessary because
   browser `EventSource` gives client code no hook to trigger a
   disconnect on demand (it doesn't expose an abort-with-signal API the
   way `fetch` does); a real "Cancel" button in a UI built on
   `EventSource` needs a separate request to signal intent. Both call the
   same `CancellationToken.cancel()` the Supervisor already checks.

Once disconnection is detected, the endpoint keeps draining
`run_streaming()` internally (still calling `next()` via the `for` loop)
without forwarding further SSE lines — this lets the Supervisor observe
the cancellation and persist a clean `CANCELLED` status, rather than
leaving an ambiguous "was this run just abandoned or genuinely stopped?"
gap in the audit trail.

## Testing note: transport-level disconnect timing isn't reliably testable here
An early attempt to test real mid-stream disconnection via FastAPI's
`TestClient` (reading a couple of SSE lines, then closing the response
early) didn't produce a meaningful test: this pipeline runs the tiny demo
corpus fast enough that the server-side generator had already finished
before there was a real window to close the connection — at that point
the test would be measuring httpx/Starlette's ASGI transport timing, not
this code. Instead, `test_client_disconnect_stops_forwarding_events_and_cancels_the_run`
calls the endpoint function directly with a fake `Request` whose
`is_disconnected()` flips to `True` after a controlled number of calls —
deterministic, and it exercises the actual branch (cancel the token, stop
forwarding, keep draining, persist `CANCELLED`) rather than hoping for
favorable timing. The registry-based explicit-cancel endpoint is tested
directly against the registry, since the deeper claim — that a cancelled
token actually stops the Supervisor mid-pipeline — is already proven at
the application layer in `test_supervisor.py`.

## Consequences / limitations, stated plainly
- `/ask/stream`'s disconnect handling stops *forwarding* tokens to a gone
  client but can't recall a hosted LLM call already dispatched — same
  "can't preempt what's already in flight" limitation as everything else
  in this codebase that touches threads or network calls.
- `FallbackStreamingLLMProvider` (infrastructure/resilience/) restarts
  entirely from the secondary provider if the primary fails partway
  through a stream, rather than resuming — a client sees primary's
  partial output followed by secondary's full output from the start. In
  practice this matters less than it sounds, since the common failure
  modes (missing key, 429, connection refused) all fail on the very first
  chunk.
- The run_id -> CancellationToken registry in `interface/http_api.py` is
  in-process and unsynchronized — fine for this single-process demo
  transport, not for a multi-worker production deployment. Tracked in
  PLAN.md's roadmap alongside the rest of FR-7's surface work.
- `GeminiLLMProvider.stream_complete()` is written against Gemini's
  documented `alt=sse` streaming response shape but has not been
  exercised against the live API in this sandbox (no network egress to
  `generativelanguage.googleapis.com`, no configured key) — same
  documented-but-untested status as the rest of this file's Gemini calls.

## Alternatives considered
- **WebSocket** — rejected for this traffic shape (see above); would be
  reconsidered if a future feature needs the client to push mid-stream.
- **A message queue / pub-sub for progress events** (e.g. Redis) instead
  of a direct generator — more infrastructure than a single-process demo
  needs; the right choice once FR-7's multi-worker deployment actually
  exists, not before.