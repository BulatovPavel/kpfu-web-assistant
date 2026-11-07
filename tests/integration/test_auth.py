import sqlite3
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.api.main import app
from backend.db.models import Base
from backend.db.session import get_db
import backend.api.main as main_module


@pytest.fixture
def api_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, Path]]:
    database_path = tmp_path / "auth-test.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database_path}")
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def create_test_tables() -> None:
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def override_get_db() -> Iterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    monkeypatch.setattr(main_module, "create_tables", create_test_tables)
    monkeypatch.setattr(main_module, "dispose_engine", engine.dispose)
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            yield client, database_path
    finally:
        app.dependency_overrides.pop(get_db, None)


def register_user(
    client: TestClient,
    *,
    email: str = "alice@example.com",
    username: str = "alice",
    password: str = "MyPass123",
):
    return client.post(
        "/auth/register",
        json={"email": email, "username": username, "password": password},
    )


def set_user_active(database_path: Path, email: str, is_active: bool) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE users SET is_active = ? WHERE email = ?",
            (is_active, email),
        )


def delete_user(database_path: Path, email: str) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM users WHERE email = ?", (email,))


def test_register_returns_token_and_default_user_role(api_client) -> None:
    client, _ = api_client

    response = register_user(client)

    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["role"] == "user"
    assert body["email"] == "alice@example.com"
    assert body["username"] == "alice"
    assert isinstance(body["user_id"], int)
    assert "hashed_password" not in body


@pytest.mark.parametrize(
    ("email", "username"),
    [
        ("alice@example.com", "different-name"),
        ("different@example.com", "alice"),
    ],
)
def test_register_rejects_duplicate_email_or_username(
    api_client,
    email: str,
    username: str,
) -> None:
    client, _ = api_client
    assert register_user(client).status_code == 201

    response = register_user(client, email=email, username=username)

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("bob@example.com", "short"),
        ("bob@example.com", "MyPassword"),
        ("bob@example.com", "mypassword1"),
        ("not-an-email", "MyPass123"),
    ],
)
def test_register_rejects_invalid_input(
    api_client,
    email: str,
    password: str,
) -> None:
    client, _ = api_client

    response = register_user(
        client,
        email=email,
        username="bob",
        password=password,
    )

    assert response.status_code == 422


def test_login_with_correct_credentials_returns_token(api_client) -> None:
    client, _ = api_client
    registered = register_user(client)
    assert registered.status_code == 201

    # JWT timestamps have second-level precision; ensure login creates a new token.
    time.sleep(1.1)
    response = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "MyPass123"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["role"] == "user"
    assert body["access_token"] != registered.json()["access_token"]


@pytest.mark.parametrize(
    ("email", "password"),
    [
        ("alice@example.com", "WrongPass123"),
        ("unknown@example.com", "MyPass123"),
    ],
)
def test_login_rejects_invalid_credentials(
    api_client,
    email: str,
    password: str,
) -> None:
    client, _ = api_client
    assert register_user(client).status_code == 201

    response = client.post(
        "/auth/login",
        json={"email": email, "password": password},
    )

    assert response.status_code == 401


def test_login_rejects_inactive_user(api_client) -> None:
    client, database_path = api_client
    assert register_user(client).status_code == 201
    set_user_active(database_path, "alice@example.com", False)

    response = client.post(
        "/auth/login",
        json={"email": "alice@example.com", "password": "MyPass123"},
    )

    assert response.status_code == 403


def test_me_without_token_returns_401(api_client) -> None:
    client, _ = api_client

    response = client.get("/auth/me")

    assert response.status_code == 401


def test_me_with_valid_token_returns_public_profile(api_client) -> None:
    client, _ = api_client
    registered = register_user(client)
    token = registered.json()["access_token"]

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    profile = response.json()
    assert {
        "id",
        "email",
        "username",
        "role",
        "is_active",
        "created_at",
    } <= profile.keys()
    assert profile["email"] == "alice@example.com"
    assert profile["username"] == "alice"
    assert profile["role"] == "user"
    assert profile["is_active"] is True
    assert "hashed_password" not in profile
    assert "full_name" not in profile


@pytest.mark.parametrize("user_state", ["deleted", "inactive"])
def test_me_returns_404_for_deleted_or_inactive_user(
    api_client,
    user_state: str,
) -> None:
    client, database_path = api_client
    registered = register_user(client)
    token = registered.json()["access_token"]

    if user_state == "deleted":
        delete_user(database_path, "alice@example.com")
    else:
        set_user_active(database_path, "alice@example.com", False)

    response = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404