"""
Domain entities for FR-8 (access control). Same rule as everywhere else:
plain dataclasses/enums, zero dependency on any web framework, auth
library, or SDK — the acceptance test applies here too.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Role(str, Enum):
    """Two roles with genuinely different permissions (FR-8), not just a
    label — see domain/auth_ports.py's docstring and
    interface/http_api.py for what's actually enforced server-side.

    CONTRIBUTOR: can ingest documents, ask questions, and run/cancel the
    multi-agent workflow. Cannot decide on approval-gate items — a
    contributor approving their own generated items would defeat the
    point of a human-in-the-loop gate.

    REVIEWER: can view pending approvals, decide on them (approve /
    reject / edit-and-approve), and inspect a run's trace. Cannot ingest,
    ask, or start/cancel a workflow run — a reviewer's job is oversight,
    not content generation.
    """

    CONTRIBUTOR = "contributor"
    REVIEWER = "reviewer"


@dataclass(frozen=True)
class User:
    username: str
    role: Role