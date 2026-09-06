import tempfile
import shutil

import pytest

from domain.session_entities import SessionEvent
from infrastructure.relational.session_repository import SqliteSessionRepository


@pytest.fixture()
def repo():
    d = tempfile.mkdtemp()
    r = SqliteSessionRepository(f"{d}/sessions.db")
    yield r
    shutil.rmtree(d, ignore_errors=True)


def test_saved_event_is_retrievable(repo):
    event = SessionEvent.new(
        username="alice", role="contributor", endpoint="POST /ask",
        request_summary="query=hello", response_summary="refused=False",
    )
    repo.save_event(event)

    events = repo.list_events()
    assert len(events) == 1
    assert events[0].id == event.id
    assert events[0].username == "alice"


def test_list_events_filters_by_username(repo):
    repo.save_event(SessionEvent.new("alice", "contributor", "POST /ask", "q1", "r1"))
    repo.save_event(SessionEvent.new("bob", "reviewer", "GET /approvals", "q2", "r2"))
    repo.save_event(SessionEvent.new("alice", "contributor", "POST /ingest", "q3", "r3"))

    alice_events = repo.list_events(username="alice")
    assert len(alice_events) == 2
    assert all(e.username == "alice" for e in alice_events)

    all_events = repo.list_events()
    assert len(all_events) == 3


def test_list_events_most_recent_first(repo):
    import time
    repo.save_event(SessionEvent.new("alice", "contributor", "e1", "r", "r"))
    time.sleep(0.01)
    repo.save_event(SessionEvent.new("alice", "contributor", "e2", "r", "r"))

    events = repo.list_events(username="alice")
    assert events[0].endpoint == "e2"
    assert events[1].endpoint == "e1"


def test_long_summaries_are_truncated_not_rejected():
    event = SessionEvent.new(
        username="alice", role="contributor", endpoint="POST /ingest",
        request_summary="x" * 10000, response_summary="y" * 10000,
    )
    assert len(event.request_summary) == 500
    assert len(event.response_summary) == 500