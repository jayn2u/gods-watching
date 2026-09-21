"""Argon2id password policy, hashing, and secret-file boundaries."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final, NewType, override

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

if TYPE_CHECKING:
    from pathlib import Path

Password = NewType("Password", str)

DEFAULT_OPERATOR_USERNAME: Final = "admin"

_PASSWORD_MIN_LENGTH: Final = 4
_PASSWORD_MAX_LENGTH: Final = 128
_USERNAME_MAX_LENGTH: Final = 64
_SECRET_FILE_MODE: Final = 0o600
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65_536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
)


@dataclass(frozen=True, slots=True)
class PasswordPolicyError(Exception):
    """Report password input that violates the operator credential policy."""

    reason: PasswordPolicyReason

    @override
    def __str__(self) -> str:
        """Describe a password policy failure without exposing its value."""
        return f"password rejected: {self.reason.value}"


class PasswordPolicyReason(StrEnum):
    """Classify a password policy rejection without carrying the password."""

    LENGTH = "length must be between 4 and 128 characters"
    LINE_BREAK = "line breaks are not allowed"


@dataclass(frozen=True, slots=True)
class SecretFileError(Exception):
    """Report a missing, unreadable, or unsafe password file."""

    path: Path
    reason: SecretFileReason

    @override
    def __str__(self) -> str:
        """Describe a secret-file failure without exposing file contents."""
        return f"password file {self.path}: {self.reason.value}"


class SecretFileReason(StrEnum):
    """Classify a secret-file failure without carrying file contents."""

    MISSING = "file is missing"
    METADATA = "file metadata is unavailable"
    MODE = "file mode must be 0600"
    ENCODING = "file is not readable UTF-8"
    POLICY = "password does not meet policy"


def parse_password(raw: str) -> Password:
    """Parse an operator password at the trust boundary."""
    length = len(raw)
    if not _PASSWORD_MIN_LENGTH <= length <= _PASSWORD_MAX_LENGTH:
        raise PasswordPolicyError(PasswordPolicyReason.LENGTH)
    if "\r" in raw or "\n" in raw:
        raise PasswordPolicyError(PasswordPolicyReason.LINE_BREAK)
    return Password(raw)


def normalize_username(raw: str) -> str:
    """Normalize an operator identifier for case-insensitive comparison."""
    candidate = raw.strip()
    if not candidate or len(candidate) > _USERNAME_MAX_LENGTH:
        return ""
    return candidate.casefold()


def verify_username(candidate: str, expected: str) -> bool:
    """Compare an operator identifier against the configured one in constant time."""
    configured = normalize_username(expected)
    supplied = normalize_username(candidate)
    if not configured or not supplied:
        return False
    return secrets.compare_digest(supplied, configured)


def hash_password(password: Password) -> str:
    """Hash a validated password with Argon2id and a random salt."""
    return _PASSWORD_HASHER.hash(str(password))


def verify_password(password: Password, encoded_hash: str) -> bool:
    """Verify a password against an Argon2id hash without raising mismatch errors."""
    try:
        return _PASSWORD_HASHER.verify(encoded_hash, str(password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def generate_password() -> Password:
    """Generate a valid random initial operator password."""
    return parse_password(secrets.token_urlsafe(32))


def read_password_file(path: Path) -> Password:
    """Read a UTF-8 password file that is exactly mode 0600."""
    try:
        mode = path.stat().st_mode & 0o777
    except FileNotFoundError as error:
        raise SecretFileError(path, SecretFileReason.MISSING) from error
    except OSError as error:
        raise SecretFileError(path, SecretFileReason.METADATA) from error
    if mode != _SECRET_FILE_MODE:
        raise SecretFileError(path, SecretFileReason.MODE)
    try:
        raw = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SecretFileError(path, SecretFileReason.ENCODING) from error
    raw = raw.removesuffix("\r\n").removesuffix("\n")
    try:
        return parse_password(raw)
    except PasswordPolicyError as error:
        raise SecretFileError(path, SecretFileReason.POLICY) from error
