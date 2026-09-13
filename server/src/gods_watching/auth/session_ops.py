"""SQLAlchemy session queries shared by authentication lifecycle operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import or_, select

from gods_watching.contracts.identifiers import LoginSessionId
from gods_watching.storage.models import LoginSession

from .types import SessionRevocation, SessionRevocationReason, SessionToken, SessionView

if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


def session_view(row: LoginSession) -> SessionView:
    """Convert a persisted row into a token-free session view."""
    return SessionView(
        session_id=LoginSessionId(row.id),
        created_at=row.created_at,
        last_activity_at=row.last_activity_at,
        idle_expires_at=row.idle_expires_at,
        absolute_expires_at=row.absolute_expires_at,
    )


def revocation_event(
    row: LoginSession,
    reason: SessionRevocationReason,
    now: datetime,
) -> SessionRevocation:
    """Convert a revoked row into the narrow media close event."""
    return SessionRevocation(
        session_id=LoginSessionId(row.id),
        reason=reason,
        revoked_at=now,
    )


async def find_session(
    session: AsyncSession,
    token: SessionToken,
    *,
    lock: bool,
) -> LoginSession | None:
    """Find a token by digest, optionally locking its row for mutation."""
    statement = select(LoginSession).where(LoginSession.token_hash == token.digest())
    if lock:
        statement = statement.with_for_update()
    return await session.scalar(statement)


async def active_sessions(session: AsyncSession, now: datetime) -> list[LoginSession]:
    """Lock and return sessions that have not reached either expiry limit."""
    statement = (
        select(LoginSession)
        .where(
            LoginSession.revoked_at.is_(None),
            LoginSession.idle_expires_at > now,
            LoginSession.absolute_expires_at > now,
        )
        .order_by(LoginSession.created_at, LoginSession.id)
        .with_for_update()
    )
    return list((await session.scalars(statement)).all())


async def expired_sessions(session: AsyncSession, now: datetime) -> list[LoginSession]:
    """Lock and return sessions that reached idle or absolute expiry."""
    statement = (
        select(LoginSession)
        .where(
            LoginSession.revoked_at.is_(None),
            or_(
                LoginSession.idle_expires_at <= now,
                LoginSession.absolute_expires_at <= now,
            ),
        )
        .with_for_update()
    )
    return list((await session.scalars(statement)).all())


async def revoke_all(
    session: AsyncSession,
    *,
    reason: SessionRevocationReason,
    now: datetime,
) -> tuple[SessionRevocation, ...]:
    """Revoke all non-revoked sessions and return their close events."""
    statement = (
        select(LoginSession)
        .where(LoginSession.revoked_at.is_(None))
        .order_by(LoginSession.created_at, LoginSession.id)
        .with_for_update()
    )
    rows = list((await session.scalars(statement)).all())
    for row in rows:
        row.revoked_at = now
    return tuple(revocation_event(row, reason, now) for row in rows)
