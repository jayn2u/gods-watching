"""Structured auth expiry and media ownership lifecycle for the FastAPI app."""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING

import anyio
from sqlalchemy import select

from gods_watching.auth import CLEANUP_INTERVAL_SECONDS
from gods_watching.media import MediaSessionId
from gods_watching.storage import LoginSession

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from fastapi import FastAPI

    from .dependencies import ApiDependencies


def build_lifespan(
    dependencies: ApiDependencies,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create a lifespan callback bound to explicitly prepared production services."""

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        async with dependencies.clip_lifecycle:
            startup_revocations = await dependencies.auth.cleanup_expired()
            startup_closed = await reconcile_revoked_sessions(dependencies)
            del startup_revocations, startup_closed
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(dependencies.auth.cleanup_loop)
                task_group.start_soon(_reconcile_loop, dependencies)
                if dependencies.training is not None:
                    task_group.start_soon(dependencies.training.warm_dataset)
                try:
                    yield
                finally:
                    task_group.cancel_scope.cancel()
            _ = await dependencies.whep.close_all()
            await dependencies.database.close()

    return _lifespan


async def reconcile_revoked_sessions(dependencies: ApiDependencies) -> int:
    """Close WHEP resources for DB revocations committed by another process."""
    async with dependencies.database.session_factory() as session:
        statement = select(LoginSession.id).where(LoginSession.revoked_at.is_not(None))
        session_ids = tuple((await session.scalars(statement)).all())
    closed = 0
    for session_id in session_ids:
        closed += await dependencies.whep.close_session(MediaSessionId(str(session_id)))
    return closed


async def _reconcile_loop(dependencies: ApiDependencies) -> None:
    while True:
        await anyio.sleep(CLEANUP_INTERVAL_SECONDS)
        _ = await reconcile_revoked_sessions(dependencies)
