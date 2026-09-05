"""
Cancellation support for FR-6. A thin, thread-safe wrapper around
threading.Event: a client-facing layer (the HTTP SSE endpoint, or a CLI
Ctrl-C handler) calls .cancel() when the client disconnects or asks to
stop; Supervisor.run_streaming() checks .is_cancelled() at each step
boundary and raises RunCancelled instead of starting the next step.

Honest scope: this stops the NEXT step from starting. It does not, and
cannot, preempt a step already in flight — Python cannot forcibly kill a
running thread (same limitation documented in ADR-004 for per-step
timeouts). For the tool calls this system actually makes (corpus search,
deterministic/LLM item drafting, a local DB write), steps are short, so in
practice cancellation lands within one step's duration, not instantly —
that's a real, stated limitation, not a claim of hard real-time preemption.
"""
from __future__ import annotations

import threading


class RunCancelled(Exception):
    """Raised internally when a cancellation is observed at a step
    boundary. Handled distinctly from other exceptions: a cancellation
    is an intentional client action, not a failure, so it must NOT trigger
    graceful degradation to plain RAG — that would ignore what the client
    asked for."""


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()