from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final

import anyio
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from gods_watching.auth import (
    Authenticated,
    AuthenticationFailure,
    AuthenticationFailureReason,
    AuthService,
    InvalidCredentials,
    LoginAccepted,
    LoginAttempt,
    LoginThrottled,
    NoopRevocationHook,
    PasswordReplacement,
    SecretFileError,
    SessionRevocation,
    SessionRevocationReason,
    SqlAlchemyCredentialStore,
)
from gods_watching.storage import Database, LoginSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from gods_watching.auth import CredentialStore

_AUTH_INPUT_A: Final = "correct horse battery staple"
_AUTH_INPUT_B: Final = "new correct horse battery"
_START: Final = datetime(2026, 9, 7, 12, tzinfo=UTC)


class ControlledClock:
    def __init__(self, current: datetime) -> None:
        self.current: datetime = current

    def now(self) -> datetime:
        return self.current

    def advance(self, **delta: float) -> None:
        self.current += timedelta(**delta)


class RecordingHook:
    def __init__(self) -> None:
        self.events: list[SessionRevocation] = []

    async def __call__(self, event: SessionRevocation) -> None:
        self.events.append(event)


class InjectedCredentialFailureError(RuntimeError):
    pass


class FailingCredentialStore:
    def __init__(self) -> None:
        self.delegate: SqlAlchemyCredentialStore = SqlAlchemyCredentialStore()

    async def read_password_hash(self, session: AsyncSession, *, lock: bool) -> str | None:
        return await self.delegate.read_password_hash(session, lock=lock)

    async def write_password_hash(
        self,
        session: AsyncSession,
        encoded_hash: str,
        *,
        updated_at: datetime,
    ) -> None:
        await self.delegate.write_password_hash(session, encoded_hash, updated_at=updated_at)
        raise InjectedCredentialFailureError


@pytest.fixture
async def auth_database(engine: AsyncEngine) -> AsyncIterator[Database]:
    # Given
    async with engine.begin() as connection:
        _ = await connection.execute(text("DELETE FROM operator_credentials"))
        _ = await connection.execute(text("DELETE FROM sessions"))
    database = Database(engine, async_sessionmaker(engine, expire_on_commit=False))
    yield database
    async with engine.begin() as connection:
        _ = await connection.execute(text("DELETE FROM operator_credentials"))
        _ = await connection.execute(text("DELETE FROM sessions"))


def _service(
    database: Database,
    clock: ControlledClock,
    hook: RecordingHook | None = None,
    credentials: CredentialStore | None = None,
) -> AuthService:
    return AuthService(
        transactions=database,
        clock=clock,
        revocation_hook=hook or NoopRevocationHook(),
        credentials=credentials or SqlAlchemyCredentialStore(),
    )


async def _login(service: AuthService, password: str = _AUTH_INPUT_A) -> LoginAccepted:
    result = await service.login(LoginAttempt(password=password, peer_ip="203.0.113.9"))
    assert isinstance(result, LoginAccepted)
    return result


async def _active_count(database: Database) -> int:
    async with database.session_factory() as session:
        result = await session.scalar(
            select(func.count()).select_from(LoginSession).where(LoginSession.revoked_at.is_(None))
        )
        assert result is not None
        return result


@pytest.mark.anyio
async def test_db_login_stores_argon2id_hash_and_opaque_token(auth_database: Database) -> None:
    # Given
    clock = ControlledClock(_START)
    service = _service(auth_database, clock)

    # When
    initialized = await service.initialize_password(_AUTH_INPUT_A)
    accepted = await _login(service)

    # Then
    assert initialized.changed is True
    assert accepted.token.raw not in repr(accepted)
    assert accepted.token.raw not in str(accepted.token)
    assert accepted.token.digest() != accepted.token.raw.encode()
    async with auth_database.session_factory() as session:
        encoded_hash = await SqlAlchemyCredentialStore().read_password_hash(session, lock=False)
        assert encoded_hash is not None
        assert encoded_hash.startswith("$argon2id$")
        stored = await session.scalar(select(LoginSession))
        assert stored is not None
        assert stored.token_hash == accepted.token.digest()
        assert accepted.token.raw.encode() not in stored.token_hash


@pytest.mark.anyio
async def test_fifth_login_evicts_oldest_and_awaits_close_hook(auth_database: Database) -> None:
    # Given
    clock = ControlledClock(_START)
    hook = RecordingHook()
    service = _service(auth_database, clock, hook)
    _ = await service.initialize_password(_AUTH_INPUT_A)
    first = await _login(service)
    for _ in range(3):
        clock.advance(seconds=1)
        _ = await _login(service)

    # When
    clock.advance(seconds=1)
    fifth = await _login(service)

    # Then
    assert len(fifth.evicted) == 1
    assert fifth.evicted[0].session_id == first.session.session_id
    assert fifth.evicted[0].reason is SessionRevocationReason.EVICTED
    assert hook.events == list(fifth.evicted)
    evicted_result = await service.authenticate(first.token)
    assert isinstance(evicted_result, AuthenticationFailure)
    assert evicted_result.reason is AuthenticationFailureReason.REVOKED
    assert await _active_count(auth_database) == 4


@pytest.mark.anyio
async def test_activity_refreshes_idle_only_and_cleanup_emits_expiry(
    auth_database: Database,
) -> None:
    # Given
    clock = ControlledClock(_START)
    hook = RecordingHook()
    service = _service(auth_database, clock, hook)
    _ = await service.initialize_password(_AUTH_INPUT_A)
    accepted = await _login(service)
    clock.advance(minutes=29)

    # When
    passive = await service.authenticate(accepted.token)
    active = await service.authenticate(accepted.token, user_action=True)

    # Then
    assert isinstance(passive, Authenticated)
    assert passive.session.last_activity_at == _START
    assert isinstance(active, Authenticated)
    assert active.session.last_activity_at == _START + timedelta(minutes=29)
    clock.advance(minutes=30)
    expired = await service.cleanup_expired()
    assert len(expired) == 1
    assert expired[0].reason is SessionRevocationReason.EXPIRED
    assert hook.events == list(expired)
    absolute = await _login(service)
    clock.advance(hours=8)
    absolute_expired = await service.cleanup_expired()
    assert len(absolute_expired) == 1
    assert absolute_expired[0].session_id == absolute.session.session_id
    assert absolute_expired[0].reason is SessionRevocationReason.EXPIRED
    assert hook.events == list(expired) + list(absolute_expired)


@pytest.mark.anyio
async def test_logout_is_scoped_and_password_replacement_revokes_all(
    auth_database: Database,
) -> None:
    # Given
    clock = ControlledClock(_START)
    hook = RecordingHook()
    service = _service(auth_database, clock, hook)
    _ = await service.initialize_password(_AUTH_INPUT_A)
    first = await _login(service)
    second = await _login(service)

    # When
    logged_out = await service.logout(first.token)
    replacement = await service.replace_password(_AUTH_INPUT_B)

    # Then
    assert len(logged_out) == 1
    assert logged_out[0].reason is SessionRevocationReason.LOGOUT
    assert len(replacement.revoked) == 1
    assert replacement.revoked[0].session_id == second.session.session_id
    same = await service.replace_password(_AUTH_INPUT_B)
    assert same == PasswordReplacement(changed=False, revoked=())
    old_login = await service.login(LoginAttempt(_AUTH_INPUT_A, "203.0.113.10"))
    new_login = await service.login(LoginAttempt(_AUTH_INPUT_B, "203.0.113.11"))
    assert isinstance(old_login, InvalidCredentials)
    assert isinstance(new_login, LoginAccepted)


@pytest.mark.anyio
async def test_secret_file_and_transaction_failure_preserve_credential(
    auth_database: Database,
    tmp_path: Path,
) -> None:
    # Given
    clock = ControlledClock(_START)
    service = _service(auth_database, clock)
    _ = await service.initialize_password(_AUTH_INPUT_A)
    missing = tmp_path / "missing.secret"

    # When / Then
    with pytest.raises(SecretFileError):
        _ = await service.replace_password_file(missing)
    wrong_mode = tmp_path / "wrong.secret"
    _ = wrong_mode.write_text(_AUTH_INPUT_B, encoding="utf-8")
    wrong_mode.chmod(0o644)
    with pytest.raises(SecretFileError):
        _ = await service.replace_password_file(wrong_mode)
    failing = _service(auth_database, clock, credentials=FailingCredentialStore())
    with pytest.raises(InjectedCredentialFailureError):
        _ = await failing.replace_password(_AUTH_INPUT_B)
    old_login = await service.login(LoginAttempt(_AUTH_INPUT_A, "203.0.113.12"))
    new_login = await service.login(LoginAttempt(_AUTH_INPUT_B, "203.0.113.13"))
    assert isinstance(old_login, LoginAccepted)
    assert isinstance(new_login, InvalidCredentials)


@pytest.mark.anyio
async def test_concurrent_logins_are_serialized_at_four_active_sessions(
    auth_database: Database,
) -> None:
    # Given
    clock = ControlledClock(_START)
    hook = RecordingHook()
    service = _service(auth_database, clock, hook)
    _ = await service.initialize_password(_AUTH_INPUT_A)
    results: list[LoginAccepted | InvalidCredentials | LoginThrottled] = []

    async def login_once() -> None:
        result = await service.login(LoginAttempt(_AUTH_INPUT_A, "203.0.113.20"))
        assert isinstance(result, (LoginAccepted, InvalidCredentials, LoginThrottled))
        results.append(result)

    # When
    async with anyio.create_task_group() as task_group:
        for _ in range(5):
            task_group.start_soon(login_once)

    # Then
    assert len(results) == 5
    assert sum(isinstance(result, LoginAccepted) for result in results) == 5
    assert len(hook.events) == 1
    assert await _active_count(auth_database) == 4
