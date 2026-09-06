"""Port for FR-7's persistent session history."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from domain.session_entities import SessionEvent


class SessionRepository(ABC):
    @abstractmethod
    def save_event(self, event: SessionEvent) -> None:
        ...

    @abstractmethod
    def list_events(self, username: Optional[str] = None) -> list[SessionEvent]:
        """All events, most recent first, optionally filtered to one
        user's own history. `username=None` returns every user's history
        — callers (interface/http_api.py) restrict that to the REVIEWER
        role only; this port itself doesn't enforce access control, that's
        FR-8's job at the transport layer."""
        ...