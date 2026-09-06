"""SQLite-backed adapter for SessionRepository (FR-7)."""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from domain.session_entities import SessionEvent
from domain.session_ports import SessionRepository

_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_events (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    request_summary TEXT NOT NULL,
    response_summary TEXT NOT NULL,
    timestamp TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session_events_username ON session_events(username);
"""


class SqliteSessionRepository(SessionRepository):
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def save_event(self, event: SessionEvent) -> None:
        self._conn.execute(
            """
            INSERT INTO session_events
            (id, username, role, endpoint, request_summary, response_summary, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id, event.username, event.role, event.endpoint,
                event.request_summary, event.response_summary, event.timestamp.isoformat(),
            ),
        )
        self._conn.commit()

    def list_events(self, username: Optional[str] = None) -> list[SessionEvent]:
        if username is not None:
            rows = self._conn.execute(
                "SELECT id, username, role, endpoint, request_summary, response_summary, timestamp "
                "FROM session_events WHERE username = ? ORDER BY timestamp DESC",
                (username,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, username, role, endpoint, request_summary, response_summary, timestamp "
                "FROM session_events ORDER BY timestamp DESC"
            ).fetchall()
        return [
            SessionEvent(
                id=r[0], username=r[1], role=r[2], endpoint=r[3],
                request_summary=r[4], response_summary=r[5],
                timestamp=datetime.fromisoformat(r[6]),
            )
            for r in rows
        ]