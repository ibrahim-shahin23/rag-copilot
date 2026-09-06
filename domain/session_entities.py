"""
Domain entities for FR-7's "persistent session history" requirement.
Every authenticated HTTP operation records one of these — what was called,
by whom, with what, and what came back — durably, not just logged to
stdout where it vanishes on restart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SessionEvent:
    id: str
    username: str
    role: str
    endpoint: str
    request_summary: str
    response_summary: str
    timestamp: datetime = field(default_factory=_now)

    @staticmethod
    def new(username: str, role: str, endpoint: str, request_summary: str, response_summary: str) -> "SessionEvent":
        return SessionEvent(
            id=str(uuid.uuid4()),
            username=username,
            role=role,
            endpoint=endpoint,
            request_summary=request_summary[:500],
            response_summary=response_summary[:500],
        )