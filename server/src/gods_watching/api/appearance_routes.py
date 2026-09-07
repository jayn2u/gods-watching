"""Authenticated appearance detail and current crop HTTP routes."""

from hashlib import sha256
from typing import NoReturn, Protocol, final
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.appearances import AppearanceResponse
from gods_watching.search import (
    CropChangedDuringReadError,
    CropPayload,
    CropUnavailableError,
)

from .camera_routes import SessionDependency, SessionDependencyFactory, TransactionProvider


class AppearanceLookupProvider(Protocol):
    """Provide current-pointer detail and crop reads for route composition."""

    async def get_detail(
        self,
        session: AsyncSession,
        appearance_id: UUID,
    ) -> AppearanceResponse:
        """Return one current visible appearance detail row."""
        ...

    async def get_crop(self, session: AsyncSession, appearance_id: UUID) -> CropPayload:
        """Return bytes revalidated against the current representative pointer."""
        ...


@final
class _AppearanceHandlers:
    """Keep passive authentication and pointer reads explicit for appearance routes."""

    def __init__(
        self,
        database: TransactionProvider,
        lookup: AppearanceLookupProvider,
        require_session: SessionDependencyFactory,
    ) -> None:
        self._database = database
        self._lookup = lookup
        self._passive = require_session(user_action=False)

    @property
    def passive_dependency(self) -> SessionDependency:
        """Return the dependency that authenticates without refreshing idle activity."""
        return self._passive

    async def detail(self, appearance_id: UUID) -> AppearanceResponse:
        """Return durable metadata for one current visible appearance."""
        try:
            async with self._database.transaction() as session:
                return await self._lookup.get_detail(session, appearance_id)
        except CropUnavailableError:
            _raise_api_error(404, "appearance_not_found", "appearance was not found")
        except CropChangedDuringReadError:
            _raise_api_error(
                409,
                "appearance_changed",
                "appearance changed during read; retry the request",
            )

    async def crop(self, appearance_id: UUID) -> Response:
        """Return the current JPEG only after the service's version revalidation."""
        try:
            async with self._database.transaction() as session:
                payload = await self._lookup.get_crop(session, appearance_id)
        except CropUnavailableError:
            _raise_api_error(404, "crop_unavailable", "appearance crop is unavailable")
        except CropChangedDuringReadError:
            _raise_api_error(
                409,
                "appearance_changed",
                "appearance changed during read; retry the request",
            )
        etag_material = (
            f"{payload.appearance.appearance_id}:{payload.representative_version}".encode()
        )
        etag = sha256(etag_material + payload.payload).hexdigest()
        return Response(
            content=payload.payload,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "private, no-store, max-age=0",
                "ETag": f'"{etag}"',
                "Vary": "Cookie",
                "X-Representative-Version": str(payload.representative_version),
            },
        )


def build_appearance_router(
    *,
    database: TransactionProvider,
    lookup: AppearanceLookupProvider,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build authenticated detail and crop routes around a prepared lookup service."""
    handlers = _AppearanceHandlers(database, lookup, require_session)
    router = APIRouter(prefix="/api/appearances", tags=["appearances"])
    router.add_api_route(
        "/{appearance_id}",
        handlers.detail,
        methods=["GET"],
        response_model=AppearanceResponse,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "/{appearance_id}/crop",
        handlers.crop,
        methods=["GET"],
        response_class=Response,
        dependencies=[Depends(handlers.passive_dependency)],
    )
    return router


def _raise_api_error(status_code: int, code: str, message: str) -> NoReturn:
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = ["AppearanceLookupProvider", "build_appearance_router"]
