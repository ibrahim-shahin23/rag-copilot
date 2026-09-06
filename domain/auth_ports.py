"""Port for FR-8's authentication lookup."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from domain.auth_entities import User


class UserRepository(ABC):
    @abstractmethod
    def find_by_api_key(self, api_key: str) -> Optional[User]:
        ...