from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from gods_watching.auth.passwords import (
    PasswordPolicyError,
    SecretFileError,
    hash_password,
    parse_password,
    read_password_file,
    verify_password,
)
from gods_watching.auth.policy import (
    MutationOriginError,
    canonical_client_ip,
    is_same_origin,
    require_same_origin,
)
from gods_watching.auth.throttle import IpLoginThrottle
from gods_watching.auth.types import SessionToken

if TYPE_CHECKING:
    from pathlib import Path


def test_parse_password_accepts_only_scope_lengths() -> None:
    # Given
    shortest = "p" * 12
    longest = "p" * 128

    # When
    parsed_shortest = parse_password(shortest)
    parsed_longest = parse_password(longest)

    # Then
    assert str(parsed_shortest) == shortest
    assert str(parsed_longest) == longest


@pytest.mark.parametrize("password", ["p" * 11, "p" * 129])
def test_parse_password_rejects_outside_scope_lengths(password: str) -> None:
    # When / Then
    with pytest.raises(PasswordPolicyError):
        _ = parse_password(password)


def test_forwarded_ip_is_trusted_only_from_configured_gateway() -> None:
    # Given
    forwarded = "203.0.113.9, 10.0.0.1"

    # When
    trusted = canonical_client_ip("10.0.0.2", forwarded, "10.0.0.2")
    untrusted = canonical_client_ip("10.0.0.9", forwarded, "10.0.0.2")

    # Then
    assert trusted == "203.0.113.9"
    assert untrusted == "10.0.0.9"


def test_forwarded_ip_uses_first_nonempty_client_element_and_canonicalizes_it() -> None:
    # Given
    forwarded = " , 2001:0db8:0000:0000:0000:0000:0000:0007, 192.0.2.9"

    # When
    client_ip = canonical_client_ip("10.0.0.2", forwarded, "10.0.0.2")

    # Then
    assert client_ip == "2001:db8::7"


def test_origin_policy_requires_exact_same_origin() -> None:
    # Then
    assert is_same_origin("https://console.example.test", "https://console.example.test")
    assert not is_same_origin("https://attacker.example.test", "https://console.example.test")
    assert not is_same_origin(None, "https://console.example.test")


def test_login_throttle_blocks_sixth_failure_for_five_minutes() -> None:
    # Given
    throttle = IpLoginThrottle()
    started = datetime(2026, 9, 7, 12, tzinfo=UTC)

    # When
    outcomes = [
        throttle.record_failure("203.0.113.9", started + timedelta(seconds=i)) for i in range(5)
    ]
    blocked = throttle.record_failure("203.0.113.9", started + timedelta(seconds=5))

    # Then
    assert outcomes == [False, False, False, False, False]
    assert blocked is True
    assert throttle.retry_after("203.0.113.9", started + timedelta(seconds=5)) == 300


def test_argon2id_hash_verifies_without_secret_in_token_repr() -> None:
    # Given
    raw = "correct horse battery staple"
    password = parse_password(raw)
    token = SessionToken.issue()

    # When
    encoded_hash = hash_password(password)

    # Then
    assert encoded_hash.startswith("$argon2id$")
    assert verify_password(password, encoded_hash)
    assert not verify_password(parse_password("incorrect horse battery"), encoded_hash)
    assert raw not in repr(token)
    assert raw not in str(token)


def test_secret_file_requires0600_and_accepts_one_trailing_newline(tmp_path: Path) -> None:
    # Given
    path = tmp_path / "operator.secret"
    _ = path.write_text("correct horse battery staple\n", encoding="utf-8")
    path.chmod(0o600)

    # When
    parsed = read_password_file(path)

    # Then
    assert str(parsed) == "correct horse battery staple"
    path.chmod(0o640)
    with pytest.raises(SecretFileError):
        _ = read_password_file(path)


def test_same_origin_helper_rejects_missing_and_cross_origin_mutations() -> None:
    # When / Then
    with pytest.raises(MutationOriginError):
        require_same_origin(None, "https://console.example.test")
    with pytest.raises(MutationOriginError):
        require_same_origin("https://attacker.example.test", "https://console.example.test")
