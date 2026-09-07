"""Database adapters used by the authentication service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, override

from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table, select, text

if TYPE_CHECKING:
    from contextlib import AbstractAsyncContextManager
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

_AUTH_NAMESPACE_LOCK: int = 8_013_271_991
_UPSERT_INSERT = "INSERT INTO operator_credentials (singleton, password_hash, updated_at)"
_UPSERT_VALUES = "VALUES (TRUE, :password_hash, :updated_at)"
_UPSERT_CONFLICT = "ON CONFLICT (singleton) DO UPDATE SET"
_UPSERT_HASH = "password_hash = EXCLUDED.password_hash,"
_UPSERT_UPDATED = "updated_at = EXCLUDED.updated_at"
_AUTH_METADATA = MetaData()
_OPERATOR_CREDENTIALS = Table(
    "operator_credentials",
    _AUTH_METADATA,
    Column("singleton", Boolean, primary_key=True),
    Column("password_hash", String, nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
)


class TransactionProvider(Protocol):
    """Supply a caller-independent transaction for one auth operation."""

    def transaction(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Return a context manager that commits or rolls back atomically."""
        ...


class CredentialStore(Protocol):
    """Store exactly one Argon2id operator credential row."""

    async def read_password_hash(
        self,
        session: AsyncSession,
        *,
        lock: bool,
    ) -> str | None:
        """Read the hash, optionally locking the singleton row."""
        ...

    async def write_password_hash(
        self,
        session: AsyncSession,
        encoded_hash: str,
        *,
        updated_at: datetime,
    ) -> None:
        """Insert or replace the hash in the caller's transaction."""
        ...


@dataclass(frozen=True, slots=True)
class CredentialStorageError(Exception):
    """Report an invalid or unavailable credential row without secret data."""

    @override
    def __str__(self) -> str:
        """Return a stable storage error."""
        return "operator credential storage is invalid"


@dataclass(frozen=True, slots=True)
class SqlAlchemyCredentialStore:
    """Use the migration-owned operator_credentials singleton table."""

    async def read_password_hash(
        self,
        session: AsyncSession,
        *,
        lock: bool,
    ) -> str | None:
        """Read one hash without materializing credential plaintext."""
        statement = select(_OPERATOR_CREDENTIALS.c.password_hash).where(
            _OPERATOR_CREDENTIALS.c.singleton.is_(True)
        )
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    async def write_password_hash(
        self,
        session: AsyncSession,
        encoded_hash: str,
        *,
        updated_at: datetime,
    ) -> None:
        """Upsert one hash without exposing it in logs or errors."""
        statement = (
            f"{_UPSERT_INSERT} {_UPSERT_VALUES} {_UPSERT_CONFLICT} {_UPSERT_HASH} {_UPSERT_UPDATED}"
        )
        _ = await session.execute(
            text(statement),
            {"password_hash": encoded_hash, "updated_at": updated_at},
        )


async def lock_auth_namespace(session: AsyncSession) -> None:
    """Serialize credential replacement and session-cap decisions per database."""
    _ = await session.execute(
        text("SELECT pg_advisory_xact_lock(:lock_key)"),
        {"lock_key": _AUTH_NAMESPACE_LOCK},
    )
