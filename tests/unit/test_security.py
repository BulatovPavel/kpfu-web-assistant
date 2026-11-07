import pytest
import time
from datetime import timedelta

import jwt
from fastapi import HTTPException

from backend.utils import security
from backend.utils.security import get_password_hash, verify_password, create_access_token, decode_token


def test_password_hash_has_bcrypt_prefix() -> None:
    hashed = get_password_hash("MyPass123")
    assert hashed.startswith("$2b$")


def test_password_hash_does_not_contain_plain_password() -> None:
    password = "MyPass123"
    hashed = get_password_hash(password)
    assert password not in hashed


def test_verify_password_accepts_correct_password() -> None:
    password = "MyPass123"
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True


def test_verify_password_rejects_wrong_password() -> None:
    hashed = get_password_hash("MyPass123")
    assert verify_password("WrongPass123", hashed) is False


def test_verify_password_rejects_unknown_password_hash() -> None:
    hashed = get_password_hash("AnotherPass123")
    assert verify_password("MyPass123", hashed) is False


def test_verify_password_returns_false_for_malformed_hash() -> None:
    assert verify_password("MyPass123", "not-a-valid-bcrypt-hash") is False


def test_get_password_hash_rejects_73_ascii_bytes() -> None:
    with pytest.raises(ValueError):
        get_password_hash("a" * 73)


def test_get_password_hash_accepts_72_ascii_bytes() -> None:
    password = "a" * 72
    hashed = get_password_hash(password)
    assert hashed.startswith("$2b$")
    assert verify_password(password, hashed) is True


def test_get_password_hash_accepts_36_cyrillic_characters() -> None:
    password = "я" * 36  # 72 bytes in UTF-8
    hashed = get_password_hash(password)
    assert hashed.startswith("$2b$")
    assert verify_password(password, hashed) is True


def test_get_password_hash_rejects_37_cyrillic_characters() -> None:
    password = "я" * 37  # 74 bytes in UTF-8
    with pytest.raises(ValueError):
        get_password_hash(password)


def test_hashes_of_same_password_are_different() -> None:
    password = "MyPass123"
    assert get_password_hash(password) != get_password_hash(password)


def test_access_token_has_three_jwt_segments() -> None:
    token = create_access_token({"sub": "x", "role": "user"})
    assert token.count(".") == 2


def test_decode_token_returns_claims_and_timestamps() -> None:
    token = create_access_token({"sub": "x", "role": "user"})

    payload = decode_token(token)

    assert payload["sub"] == "x"
    assert payload["role"] == "user"
    assert isinstance(payload["iat"], int)
    assert isinstance(payload["exp"], int)


def test_decode_token_rejects_expired_token() -> None:
    token = create_access_token(
        {"sub": "x", "role": "user"},
        expires_delta=timedelta(seconds=1),
    )
    time.sleep(2)

    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)

    assert exc_info.value.status_code == 401


def test_create_access_token_rejects_unsupported_claim_type() -> None:
    with pytest.raises(TypeError):
        create_access_token({"sub": ["list"]})


def test_decode_token_rejects_invalid_string() -> None:
    with pytest.raises(HTTPException) as exc_info:
        decode_token("not.a.jwt")

    assert exc_info.value.status_code == 401


def test_decode_token_rejects_token_signed_with_another_secret() -> None:
    token = jwt.encode(
        {"sub": "x", "role": "user"},
        "a-different-secret",
        algorithm=security.ALGORITHM,
    )

    with pytest.raises(HTTPException) as exc_info:
        decode_token(token)

    assert exc_info.value.status_code == 401