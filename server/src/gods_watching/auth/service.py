"""Transactional single-operator authentication and session lifecycle service."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import anyio

from gods_watching.storage.models import LoginSession

from .passwords import (
    DEFAULT_OPERATOR_USERNAME,
    PasswordPolicyError,
    hash_password,
    parse_password,
    read_password_file,
    verify_password,
    verify_username,
)
from .policy import canonical_client_ip
from .repository import (
    CredentialStore,
    SqlAlchemyCredentialStore,
    TransactionProvider,
    lock_auth_namespace,
)
from .session_ops import (
    active_sessions,
    expired_sessions,
    find_session,
    revocation_event,
    revoke_all,
    session_view,
)
from .throttle import IpLoginThrottle
from .types import (
    Authenticated,
    AuthenticationFailure,
    AuthenticationFailureReason,
    AuthenticationResult,
    Clock,
    ClockError,
    InvalidCredentials,
    LoginAccepted,
    LoginAttempt,
    LoginResult,
    LoginThrottled,
    NoopRevocationHook,
    PasswordReplacement,
    SessionRevocation,
    SessionRevocationHook,
    SessionRevocationReason,
    SessionToken,
)

IDLE_TIMEOUT: Final = timedelta(minutes=30)
ABSOLUTE_TIMEOUT: Final = timedelta(hours=8)
CLEANUP_INTERVAL_SECONDS: Final = 5.0


@dataclass(frozen=True, slots=True)
class UtcClock:
    """Provide aware UTC timestamps to production auth operations."""

    def now(self) -> datetime:
        """Return the current UTC instant."""
        return datetime.now(UTC)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ClockError
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class AuthService:
    """Own password verification, DB sessions, expiry, and revocation events."""

    transactions: TransactionProvider
    operator_username: str = DEFAULT_OPERATOR_USERNAME
    credentials: CredentialStore = field(default_factory=SqlAlchemyCredentialStore)
    clock: Clock = field(default_factory=UtcClock)
    revocation_hook: SessionRevocationHook = field(default_factory=NoopRevocationHook)
    throttle: IpLoginThrottle = field(default_factory=IpLoginThrottle)

    async def initialize_password(self, raw_password: str) -> PasswordReplacement:
        """Set the initial credential only when the singleton row is absent."""
        password = parse_password(raw_password)
        async with self.transactions.transaction() as session:
            await lock_auth_namespace(session)
            existing = await self.credentials.read_password_hash(session, lock=True)
            if existing is not None:
                return PasswordReplacement(changed=False, revoked=())
            await self.credentials.write_password_hash(
                session,
                hash_password(password),
                updated_at=self._now(),
            )
        return PasswordReplacement(changed=True, revoked=())

    async def replace_password(self, raw_password: str) -> PasswordReplacement:
        """Atomically replace the password and revoke every existing session."""
        password = parse_password(raw_password)
        revocations = ()
        async with self.transactions.transaction() as session:
            await lock_auth_namespace(session)
            previous_hash = await self.credentials.read_password_hash(session, lock=True)
            if previous_hash is not None and verify_password(password, previous_hash):
                return PasswordReplacement(changed=False, revoked=())
            now = self._now()
            await self.credentials.write_password_hash(
                session,
                hash_password(password),
                updated_at=now,
            )
            revocations = await revoke_all(
                session,
                reason=SessionRevocationReason.CREDENTIAL_REPLACED,
                now=now,
            )
        await self._emit(revocations)
        return PasswordReplacement(changed=True, revoked=revocations)

    async def sync_password(self, raw_password: str) -> PasswordReplacement:
        """Make the configured password authoritative, creating or replacing the credential."""
        return await self.replace_password(raw_password)

    async def replace_password_file(self, path: str | Path) -> PasswordReplacement:
        """Read a mode-0600 password file before opening the replacement transaction."""
        return await self.replace_password(read_password_file(Path(path)))

    async def login(self, attempt: LoginAttempt) -> LoginResult:
        """Verify credentials, create a hashed-token session, and evict the oldest fourth."""
        client_ip = canonical_client_ip(
            attempt.peer_ip,
            attempt.forwarded_for,
            attempt.trusted_gateway,
        )
        now = self._now()
        if self.throttle.is_blocked(client_ip, now):
            return LoginThrottled(self.throttle.retry_after(client_ip, now))
        try:
            password = parse_password(attempt.password)
        except PasswordPolicyError:
            return self._failed_login(client_ip, now)
        # Evaluated here, applied after the hash check so a wrong identifier still pays
        # the Argon2 cost and cannot be distinguished by response time.
        username_ok = verify_username(attempt.username, self.operator_username)

        revocations: tuple[SessionRevocation, ...] = ()
        accepted: LoginAccepted | None = None
        async with self.transactions.transaction() as session:
            await lock_auth_namespace(session)
            encoded_hash = await self.credentials.read_password_hash(session, lock=True)
            password_ok = encoded_hash is not None and verify_password(password, encoded_hash)
            if not password_ok or not username_ok:
                accepted = None
            else:
                now = self._now()
                active = await active_sessions(session, now)
                evicted_rows = active[:-3]
                revocations = tuple(
                    revocation_event(row, SessionRevocationReason.EVICTED, now)
                    for row in evicted_rows
                )
                for row in evicted_rows:
                    row.revoked_at = now
                token = SessionToken.issue()
                row = LoginSession(
                    token_hash=token.digest(),
                    created_at=now,
                    last_activity_at=now,
                    idle_expires_at=min(now + IDLE_TIMEOUT, now + ABSOLUTE_TIMEOUT),
                    absolute_expires_at=now + ABSOLUTE_TIMEOUT,
                )
                session.add(row)
                await session.flush()
                accepted = LoginAccepted(token, session_view(row), revocations)
        if accepted is None:
            return self._failed_login(client_ip, now)
        self.throttle.record_success(client_ip)
        await self._emit(revocations)
        return accepted

    async def authenticate(
        self,
        token: SessionToken,
        *,
        user_action: bool = False,
    ) -> AuthenticationResult:
        """Authenticate one token and refresh idle expiry only for meaningful actions."""
        revocations: tuple[SessionRevocation, ...] = ()
        result: AuthenticationResult
        async with self.transactions.transaction() as session:
            row = await find_session(session, token, lock=True)
            if row is None:
                result = AuthenticationFailure(AuthenticationFailureReason.UNKNOWN)
            elif row.revoked_at is not None:
                result = AuthenticationFailure(AuthenticationFailureReason.REVOKED)
            else:
                now = self._now()
                if now >= row.absolute_expires_at or now >= row.idle_expires_at:
                    row.revoked_at = now
                    revocations = (revocation_event(row, SessionRevocationReason.EXPIRED, now),)
                    result = AuthenticationFailure(AuthenticationFailureReason.EXPIRED)
                else:
                    if user_action:
                        row.last_activity_at = now
                        row.idle_expires_at = min(now + IDLE_TIMEOUT, row.absolute_expires_at)
                    result = Authenticated(session_view(row))
        await self._emit(revocations)
        return result

    async def logout(self, token: SessionToken) -> tuple[SessionRevocation, ...]:
        """Revoke only the session represented by the supplied token."""
        revocations: tuple[SessionRevocation, ...] = ()
        async with self.transactions.transaction() as session:
            row = await find_session(session, token, lock=True)
            if row is not None and row.revoked_at is None:
                now = self._now()
                row.revoked_at = now
                revocations = (revocation_event(row, SessionRevocationReason.LOGOUT, now),)
        await self._emit(revocations)
        return revocations

    async def cleanup_expired(self) -> tuple[SessionRevocation, ...]:
        """Revoke idle or absolute-expired sessions and await their close hooks."""
        revocations = ()
        async with self.transactions.transaction() as session:
            await lock_auth_namespace(session)
            now = self._now()
            rows = await expired_sessions(session, now)
            for row in rows:
                row.revoked_at = now
            revocations = tuple(
                revocation_event(row, SessionRevocationReason.EXPIRED, now) for row in rows
            )
        await self._emit(revocations)
        return revocations

    async def cleanup_loop(self) -> None:
        """Run expiry cleanup every five seconds until its task scope is cancelled."""
        while True:
            _ = await self.cleanup_expired()
            await anyio.sleep(CLEANUP_INTERVAL_SECONDS)

    def _now(self) -> datetime:
        return _utc(self.clock.now())

    def _failed_login(self, client_ip: str, now: datetime) -> LoginResult:
        if self.throttle.record_failure(client_ip, now):
            return LoginThrottled(self.throttle.retry_after(client_ip, now))
        return InvalidCredentials()

    async def _emit(self, events: tuple[SessionRevocation, ...]) -> None:
        for event in events:
            await self.revocation_hook(event)
