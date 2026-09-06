import json

import pytest

from domain.auth_entities import Role
from infrastructure.auth.static_user_repository import StaticUserRepository


def test_from_file_loads_users_with_correct_roles(tmp_path):
    path = tmp_path / "users.json"
    path.write_text(json.dumps({
        "key-a": {"username": "alice", "role": "contributor"},
        "key-b": {"username": "bob", "role": "reviewer"},
    }))
    repo = StaticUserRepository.from_file(path)

    alice = repo.find_by_api_key("key-a")
    assert alice.username == "alice"
    assert alice.role == Role.CONTRIBUTOR

    bob = repo.find_by_api_key("key-b")
    assert bob.role == Role.REVIEWER


def test_unknown_api_key_returns_none(tmp_path):
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"key-a": {"username": "alice", "role": "contributor"}}))
    repo = StaticUserRepository.from_file(path)
    assert repo.find_by_api_key("does-not-exist") is None


def test_invalid_role_in_file_raises_clear_error(tmp_path):
    path = tmp_path / "users.json"
    path.write_text(json.dumps({"key-a": {"username": "alice", "role": "not-a-real-role"}}))
    with pytest.raises(ValueError):
        StaticUserRepository.from_file(path)