"""Authenticated latest detector overlay route."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - FastAPI resolves this route annotation
from typing import TYPE_CHECKING, Protocol
from uuid import UUID  # noqa: TC003 - FastAPI resolves this path annotation

from fastapi import APIRouter, Depends, HTTPException, Response, status

from gods_watching.contracts.live import LiveDetectionResponse
from gods_watching.live_detections import (
    LiveDetectionCameraNotFoundError,
    LiveDetectionRepository,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from .camera_routes import SessionDependencyFactory, TransactionProvider


class LiveDetectionReader(Protocol):
    """Read one latest overlay through the caller-owned transaction."""

    async def read(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        now: datetime | None = None,
    ) -> LiveDetectionResponse:
        """Return current-generation boxes or the bounded empty snapshot."""
        ...


def build_live_detection_router(
    *,
    database: TransactionProvider,
    require_session: SessionDependencyFactory,
    reader: LiveDetectionReader | None = None,
) -> APIRouter:
    """Build the passive authenticated latest-detection endpoint."""
    passive = require_session(user_action=False)
    snapshot_reader = reader or LiveDetectionRepository()
    router = APIRouter(prefix="/api/live/{camera_id}", tags=["live"])

    async def get_detections(camera_id: UUID, response: Response) -> LiveDetectionResponse:
        response.headers["Cache-Control"] = "no-store"
        try:
            async with database.transaction() as session:
                return await snapshot_reader.read(session, camera_id)
        except LiveDetectionCameraNotFoundError as error:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "camera_not_found", "message": "camera was not found"},
            ) from error

    router.add_api_route(
        "/detections",
        get_detections,
        methods=["GET"],
        response_model=LiveDetectionResponse,
        dependencies=[Depends(passive)],
    )
    return router


__all__ = ["LiveDetectionReader", "build_live_detection_router"]
