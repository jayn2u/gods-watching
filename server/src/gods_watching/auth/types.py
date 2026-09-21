"""Typed values and narrow protocols for the authentication service."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, NewType, Protocol, override

from .passwords import DEFAULT_OPERATOR_USERNAME

if TYPE_CHECKING:
    from datetime import datetime

    from gods_watching.contracts.identifiers import LoginSessionId

ClientIp = NewType("ClientIp", str)


class Clock(Protocol):
    """Provide the UTC instant used by expiry and throttle decisions."""

    def now(self) -> datetime:
        """Return an aware UTC timestamp."""
        ...


@dataclass(frozen=True, slots=True)
class ClockError(Exception):
    """Report an invalid timestamp from a configured authentication clock."""

    @override
    def __str__(self) -> str:
        """Return a stable clock error without exposing runtime values."""
        return "authentication clock must return an aware timestamp"


class SessionRevocationReason(StrEnum):
    """Explain why a persisted login session became unusable."""

    EVICTED = "evicted"
    LOGOUT = "logout"
    CREDENTIAL_REPLACED = "credential_replaced"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class SessionRevocation:
    """Identify one session whose live resources must be closed."""

    session_id: LoginSessionId
    reason: SessionRevocationReason
    revoked_at: datetime


class SessionRevocationHook(Protocol):
    """Receive awaited revocations for the later media integration."""

    async def __call__(self, event: SessionRevocation) -> None:
        """Close resources owned by one revoked application session."""
        ...


@dataclass(frozen=True, slots=True)
class NoopRevocationHook:
    """Provide the safe default until the media manager is wired in 12c."""

    async def __call__(self, event: SessionRevocation) -> None:
        """Acknowledge a revocation without owning media resources."""
        _ = event


@dataclass(frozen=True, slots=True, repr=False)
class SessionToken:
    """Hold a browser token while keeping accidental representations redacted."""

    _raw: str

    @classmethod
    def issue(cls) -> SessionToken:
        """Generate a 256-bit URL-safe opaque token."""
        return cls(secrets.token_urlsafe(32))

    @classmethod
    def from_raw(cls, raw: str) -> SessionToken:
        """Wrap a cookie value without exposing it in service errors or reprs."""
        return cls(raw)

    @property
    def raw(self) -> str:
        """Return the value for the HTTP-only cookie boundary."""
        return self._raw

    def digest(self) -> bytes:
        """Return the only token representation stored by the database."""
        return hashlib.sha256(self._raw.encode("utf-8")).digest()

    @override
    def __repr__(self) -> str:
        """Keep token values out of tracebacks, logs, and test failure output."""
        return "SessionToken(<redacted>)"

    @override
    def __str__(self) -> str:
        """Keep implicit string formatting redacted."""
        return "<redacted>"


@dataclass(frozen=True, slots=True, repr=False)
class LoginAttempt:
    """Carry login input while redacting the password from representations."""

    password: str
    peer_ip: str
    forwarded_for: str | None = None
    trusted_gateway: str | None = None
    username: str = DEFAULT_OPERATOR_USERNAME

    @override
    def __repr__(self) -> str:
        """Keep a login password out of tracebacks and test output."""
        return "LoginAttempt(<redacted>)"


@dataclass(frozen=True, slots=True)
class SessionView:
    """Expose session state without its browser token."""

    session_id: LoginSessionId
    created_at: datetime
    last_activity_at: datetime
    idle_expires_at: datetime
    absolute_expires_at: datetime


@dataclass(frozen=True, slots=True)
class LoginAccepted:
    """Return a new session token and its non-secret expiry state."""

    token: SessionToken
    session: SessionView
    evicted: tuple[SessionRevocation, ...]


@dataclass(frozen=True, slots=True)
class InvalidCredentials:
    """Represent a failed login without revealing which check failed."""


@dataclass(frozen=True, slots=True)
class LoginThrottled:
    """Represent a temporary client-IP lockout."""

    retry_after_seconds: int


type LoginResult = LoginAccepted | InvalidCredentials | LoginThrottled


@dataclass(frozen=True, slots=True)
class Authenticated:
    """Return the session that authenticated a request."""

    session: SessionView


class AuthenticationFailureReason(StrEnum):
    """Classify an unusable session for the HTTP boundary."""

    UNKNOWN = "unknown"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass(frozen=True, slots=True)
class AuthenticationFailure:
    """Represent an authentication failure without echoing a token."""

    reason: AuthenticationFailureReason


type AuthenticationResult = Authenticated | AuthenticationFailure


@dataclass(frozen=True, slots=True)
class PasswordReplacement:
    """Report whether a password changed and which sessions were revoked."""

    changed: bool
    revoked: tuple[SessionRevocation, ...]
