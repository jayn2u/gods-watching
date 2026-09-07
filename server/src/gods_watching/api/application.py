"""FastAPI application factory for the prepared operator services."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, FastAPI, Request, Response
from sqlalchemy import select

from gods_watching.media import build_whep_router
from gods_watching.storage import Camera

from .camera_routes import AuthenticatedRequest as CameraAuthenticatedRequest
from .camera_routes import build_camera_router
from .lifespan import build_lifespan
from .session_routes import build_session_router
from .sessionguard import AuthenticatedRequest, require_session
from .settings_routes import build_settings_router
from .whep_auth import SessionWhepAuthorizer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gods_watching.contracts.identifiers import CameraId

    from .dependencies import ApiDependencies


def create_app(dependencies: ApiDependencies) -> FastAPI:
    """Compose a real API from externally prepared database, auth, and media services."""
    wired = dependencies.with_whep_revocation()
    app = FastAPI(
        title="Gods Watching API",
        lifespan=build_lifespan(wired),
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    app.include_router(_build_api_router(wired))
    return app


def build_api_router(dependencies: ApiDependencies) -> APIRouter:
    """Build the authenticated router from explicitly prepared services."""
    return _build_api_router(dependencies.with_whep_revocation())


def _build_api_router(dependencies: ApiDependencies) -> APIRouter:
    router = APIRouter()
    router.include_router(build_session_router(dependencies.auth, dependencies.config))
    require_camera_session = _camera_session_factory(dependencies)
    router.include_router(
        build_camera_router(
            database=dependencies.database,
            cameras=dependencies.cameras,
            runtime=dependencies.camera_runtime,
            require_session=require_camera_session,
        )
    )
    router.include_router(
        build_settings_router(
            database=dependencies.database,
            settings=dependencies.settings,
            require_session=require_camera_session,
        )
    )
    router.include_router(
        build_whep_router(
            dependencies.whep,
            SessionWhepAuthorizer(dependencies.auth, dependencies.config),
            camera_access=_camera_access(dependencies),
        )
    )
    return router


def _camera_session_factory(
    dependencies: ApiDependencies,
) -> Callable[..., Callable[..., Awaitable[CameraAuthenticatedRequest]]]:
    def _factory(*, user_action: bool) -> Callable[..., Awaitable[CameraAuthenticatedRequest]]:
        guard = require_session(
            dependencies.auth,
            dependencies.config,
            user_action=user_action,
            mutation=user_action,
        )

        async def _dependency(
            request: Request,
            response: Response,
        ) -> CameraAuthenticatedRequest:
            context: AuthenticatedRequest = await guard(request, response)
            return CameraAuthenticatedRequest(session_id=str(context.session.session_id))

        return _dependency

    return _factory


def _camera_access(dependencies: ApiDependencies) -> Callable[[CameraId], Awaitable[bool]]:
    async def _check(camera_id: CameraId) -> bool:
        async with dependencies.database.session_factory() as session:
            statement = select(Camera.id).where(
                Camera.id == camera_id,
                Camera.deleted_at.is_(None),
            )
            return await session.scalar(statement) is not None

    return _check
