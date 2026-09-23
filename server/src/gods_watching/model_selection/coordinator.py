"""PostgreSQL advisory-lock coordination for search and model transitions."""

# The lock boundary intentionally raises stable typed messages at several
# short-lived failure points; keeping those diagnostics local avoids widening
# the public exception hierarchy solely for linting.
# ruff: noqa: TRY003, EM101, TC006

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anyio
from sqlalchemy import text

from gods_watching.model_selection.models import MaintenanceModeError, ModelSelectionConflictError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

    from gods_watching.storage.database import Database

_MODEL_SWITCH_LOCK_KEY = 8_739_211
_PIPELINE_OWNER_LOCK_KEY = 8_739_212


@dataclass(frozen=True, slots=True)
class SearchLockState:
    """Identity snapshot captured while the caller's DB transaction owns the lock."""

    model_id: str
    model_revision: str
    dimension: int


class TransitionCoordinator:
    """Coordinate all model transitions and search requests through PostgreSQL."""

    def __init__(self, database: Database | None = None) -> None:
        """Bind the durable database or an explicit test-only local lock."""
        self.database: Database | None
        self._local_transition_lock: asyncio.Lock
        self._local_worker_lock: asyncio.Lock
        self.database = database
        self._local_transition_lock = asyncio.Lock()
        self._local_worker_lock = asyncio.Lock()

    @property
    def switch_lock_key(self) -> int:
        """Expose the stable lock key for diagnostics and integration tests."""
        return _MODEL_SWITCH_LOCK_KEY

    @property
    def worker_lock_key(self) -> int:
        """Expose the process-ownership lock key."""
        return _PIPELINE_OWNER_LOCK_KEY

    @asynccontextmanager
    async def search_lock(
        self,
        session: AsyncSession,
        *,
        identity: SearchLockState | None = None,
    ) -> AsyncIterator[SearchLockState | None]:
        """Try the transaction advisory lock before any retrieval I/O.

        The caller's transaction owns the lock.  It therefore spans text/image
        inference and the result SQL, allowing an in-flight search to finish
        before a transition takes the exclusive session lock.
        """
        if self.database is None:
            if self._local_transition_lock.locked():
                raise MaintenanceModeError("model transition is in progress")
            yield identity
            return
        result = cast(
            object,
            await session.scalar(
                text("SELECT pg_try_advisory_xact_lock_shared(:lock_key)"),
                {"lock_key": _MODEL_SWITCH_LOCK_KEY},
            ),
        )
        if not bool(result):
            raise MaintenanceModeError("model transition is in progress")
        # The lock itself is the race barrier.  Reading the durable job after
        # acquiring it lets callers also gate a queued job before staging starts.
        queued = cast(
            object,
            await session.scalar(
                text(
                    """
                    SELECT EXISTS (
                        SELECT 1 FROM model_transition_jobs
                        WHERE phase IN (
                            'queued', 'preparing', 'reindexing',
                            'activating', 'rolling_back'
                        )
                    )
                    """
                )
            ),
        )
        if bool(queued):
            raise MaintenanceModeError("model transition is in progress")
        yield identity

    @asynccontextmanager
    async def transition_lock(self) -> AsyncIterator[AsyncConnection | None]:
        """Hold a dedicated session advisory lock for every staging commit."""
        if self.database is None:
            if self._local_transition_lock.locked():
                raise ModelSelectionConflictError("model transition is already owned")
            _ = await self._local_transition_lock.acquire()
            try:
                yield None
            finally:
                self._local_transition_lock.release()
            return
        async with self.database.engine.connect() as connection:
            transaction = await connection.begin()
            acquired = False
            try:
                # A normal session-level lock waits for every in-flight shared
                # search transaction to finish before the switch proceeds.
                _ = await connection.execute(
                    text("SELECT pg_advisory_lock(:lock_key)"),
                    {"lock_key": _MODEL_SWITCH_LOCK_KEY},
                )
                acquired = True
                yield connection
            finally:
                with anyio.CancelScope(shield=True):
                    if acquired:
                        try:
                            try:
                                _ = cast(
                                    object,
                                    await connection.scalar(
                                        text("SELECT pg_advisory_unlock(:lock_key)"),
                                        {"lock_key": _MODEL_SWITCH_LOCK_KEY},
                                    ),
                                )
                            except BaseException:
                                await connection.invalidate()
                                raise
                        finally:
                            await transaction.commit()
                    elif transaction.is_active:
                        await transaction.rollback()

    @asynccontextmanager
    async def worker_ownership(self) -> AsyncIterator[AsyncConnection | None]:
        """Serialize worker runtime owners across process restarts."""
        if self.database is None:
            if self._local_worker_lock.locked():
                raise ModelSelectionConflictError("pipeline worker ownership is already held")
            _ = await self._local_worker_lock.acquire()
            try:
                yield None
            finally:
                self._local_worker_lock.release()
            return
        async with self.database.engine.connect() as connection:
            transaction = await connection.begin()
            acquired = False
            try:
                _ = await connection.execute(
                    text("SELECT pg_advisory_lock(:lock_key)"),
                    {"lock_key": _PIPELINE_OWNER_LOCK_KEY},
                )
                acquired = True
                yield connection
            finally:
                with anyio.CancelScope(shield=True):
                    if acquired:
                        try:
                            try:
                                _ = cast(
                                    object,
                                    await connection.scalar(
                                        text("SELECT pg_advisory_unlock(:lock_key)"),
                                        {"lock_key": _PIPELINE_OWNER_LOCK_KEY},
                                    ),
                                )
                            except BaseException:
                                await connection.invalidate()
                                raise
                        finally:
                            await transaction.commit()
                    elif transaction.is_active:
                        await transaction.rollback()


__all__ = ["SearchLockState", "TransitionCoordinator"]
