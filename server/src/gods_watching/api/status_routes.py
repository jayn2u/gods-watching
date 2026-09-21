"""Authenticated operational status route."""

from fastapi import APIRouter, Depends

from gods_watching.contracts.status import StatusResponse
from gods_watching.status import read_status

from .camera_routes import SessionDependencyFactory, TransactionProvider


def build_status_router(
    *,
    database: TransactionProvider,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build the passive status endpoint against durable worker telemetry."""
    passive = require_session(user_action=False)
    router = APIRouter(prefix="/api/status", tags=["status"])

    async def get_status() -> StatusResponse:
        async with database.transaction() as session:
            return await read_status(session)

    router.add_api_route("", get_status, methods=["GET"], dependencies=[Depends(passive)])
    return router


__all__ = ["build_status_router"]
