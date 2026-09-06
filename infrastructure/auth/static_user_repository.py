"""
Static user repository (FR-8) — API keys mapped to (username, role) via a
JSON file, not a real identity provider. This is a deliberately minimal
auth mechanism appropriate for this slice's scope (see
docs/ADR-006-access-control.md): real password hashing, token expiry,
OAuth, etc. are out of scope here — what's in scope, and genuinely
enforced, is that two roles exist with different, server-side-checked
permissions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from domain.auth_entities import Role, User
from domain.auth_ports import UserRepository


class StaticUserRepository(UserRepository):
    def __init__(self, users_by_api_key: dict[str, User]) -> None:
        self._users = users_by_api_key

    @staticmethod
    def from_file(path: Path) -> "StaticUserRepository":
        data = json.loads(path.read_text(encoding="utf-8"))
        users: dict[str, User] = {}
        for api_key, entry in data.items():
            users[api_key] = User(username=entry["username"], role=Role(entry["role"]))
        return StaticUserRepository(users)

    def find_by_api_key(self, api_key: str) -> Optional[User]:
        return self._users.get(api_key)