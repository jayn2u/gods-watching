"""Authenticated camera registration, source tests, and lifecycle routes."""

from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Annotated, Final, NoReturn, Protocol, final
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from pydantic import AnyUrl
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.cameras import (
    CameraDeletedError,
    CameraNameConflictError,
    CameraRecordNotFoundError,
    CameraServiceError,
    ParsedRtspSource,
    ProbeFailureCode,
    RtspProbeError,
    StaleCameraVersionError,
    parse_rtsp_source,
)
from gods_watching.cameras.lifecycle import (
    CameraActivationReason,
    CameraActivationRequest,
    CameraLifecyclePlan,
)
from gods_watching.cameras.service import CameraMutation
from gods_watching.contracts.cameras import (
    CameraCreateRequest,
    CameraPatchRequest,
    CameraResponse,
    CameraTestRequest,
    CameraTestResponse,
)
from gods_watching.contracts.identifiers import CameraId
from gods_watching.storage import Camera

from .camera_runtime import (
    BoundCameraRuntimePort,
    CameraRuntimePort,
    RuntimeEffectError,
)

_ETAG_QUOTE_MIN_LENGTH: Final = 2


@dataclass(frozen=True, slots=True)
class AuthenticatedRequest:
    """Token-free marker returned by the shared application session guard."""

    session_id: str


type SessionDependency = Callable[..., Awaitable[AuthenticatedRequest]]


class SessionDependencyFactory(Protocol):
    """Build the shared passive or action-refreshing auth dependency."""

    def __call__(self, *, user_action: bool) -> SessionDependency:
        """Return a FastAPI dependency for one route activity policy."""
        ...


class TransactionProvider(Protocol):
    """Provide caller-owned asynchronous database transactions."""

    def transaction(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Return a transaction context that commits on success."""
        ...


class CameraRepositoryProvider(Protocol):
    """Expose trusted source and threshold state for a versioned update."""

    async def get(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        for_update: bool = False,
    ) -> Camera:
        """Load one camera row, optionally locking it for an update."""
        ...

    async def source(self, session: AsyncSession, camera: Camera) -> str:
        """Decrypt one source only inside the trusted service boundary."""
        ...


class CameraServiceProvider(Protocol):
    """Describe the durable camera service consumed by HTTP composition."""

    @property
    def repository(self) -> CameraRepositoryProvider:
        """Expose the trusted repository needed for source comparisons."""
        ...

    async def list(self, session: AsyncSession) -> tuple[CameraResponse, ...]:
        """List non-deleted cameras without source credentials."""
        ...

    async def create(
        self,
        session: AsyncSession,
        request: CameraCreateRequest,
    ) -> CameraMutation:
        """Persist one camera and return its detector lifecycle plan."""
        ...

    async def update(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        request: CameraPatchRequest,
        *,
        expected_version: int,
    ) -> CameraMutation:
        """Persist one optimistic versioned patch."""
        ...

    async def delete(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        *,
        expected_version: int,
    ) -> CameraMutation:
        """Soft-delete one camera and return its cancellation plan."""
        ...

    async def test_source(self, source_url: str) -> CameraTestResponse:
        """Probe one operator source inside the fixed bounded probe."""
        ...


@dataclass(frozen=True, slots=True)
class CameraRouterDependencies:
    """Compose the camera router from durable services and shared auth."""

    database: TransactionProvider
    cameras: CameraServiceProvider
    runtime: CameraRuntimePort
    require_session: SessionDependencyFactory


@dataclass(frozen=True, slots=True)
class _CameraBeforeUpdate:
    """State read while the camera row is locked before applying a patch."""

    source: str
    threshold: float


@final
class _CameraHandlers:
    """FastAPI handlers that keep auth and transaction boundaries explicit."""

    def __init__(self, dependencies: CameraRouterDependencies) -> None:
        self._database: TransactionProvider = dependencies.database
        self._cameras: CameraServiceProvider = dependencies.cameras
        self._runtime: CameraRuntimePort = dependencies.runtime
        self._passive: SessionDependency = dependencies.require_session(user_action=False)
        self._mutation: SessionDependency = dependencies.require_session(user_action=True)

    @property
    def passive_dependency(self) -> SessionDependency:
        return self._passive

    @property
    def mutation_dependency(self) -> SessionDependency:
        return self._mutation

    async def list_cameras(self) -> tuple[CameraResponse, ...]:
        """Return camera metadata through the passive auth dependency."""
        async with self._database.transaction() as session:
            return await self._cameras.list(session)

    async def create_camera(
        self,
        payload: CameraCreateRequest,
        response: Response,
    ) -> CameraResponse:
        """Commit a camera before activating its live path and detector."""
        try:
            async with self._database.transaction() as session:
                created = await self._cameras.create(session, payload)
        except (CameraNameConflictError, IntegrityError) as error:
            _raise_api_error(
                status.HTTP_409_CONFLICT,
                "camera_name_conflict",
                "camera name is already in use",
                error,
            )
        except RtspProbeError as error:
            _raise_probe_error(error)
        response.status_code = status.HTTP_201_CREATED
        response.headers["ETag"] = _etag(created.camera.version)
        try:
            await _apply_create(self._runtime, created, payload.source_url)
        except RuntimeEffectError as error:
            _raise_runtime_error(error)
        return created.camera

    async def test_camera(
        self,
        payload: CameraTestRequest,
    ) -> CameraTestResponse:
        """Probe one source without persisting or exposing its credentials."""
        try:
            return await self._cameras.test_source(str(payload.source_url))
        except RtspProbeError as error:
            _raise_probe_error(error)

    async def update_camera(
        self,
        camera_id: UUID,
        payload: CameraPatchRequest,
        response: Response,
        if_match: Annotated[str | None, Header(alias="If-Match")] = None,
        x_camera_version: Annotated[int | None, Header(alias="X-Camera-Version")] = None,
    ) -> CameraResponse:
        """Apply a versioned patch, then reconcile its post-commit effects."""
        expected_version = _expected_version(if_match, x_camera_version)
        before: _CameraBeforeUpdate | None = None
        try:
            async with self._database.transaction() as session:
                before = await _read_before_update(session, self._cameras, camera_id)
                updated = await self._cameras.update(
                    session,
                    CameraId(camera_id),
                    payload,
                    expected_version=expected_version,
                )
        except StaleCameraVersionError as error:
            _raise_api_error(
                status.HTTP_409_CONFLICT,
                "stale_camera_version",
                "camera version is stale",
                error,
            )
        except (CameraRecordNotFoundError, CameraDeletedError) as error:
            _raise_api_error(
                status.HTTP_404_NOT_FOUND,
                "camera_not_found",
                "camera was not found",
                error,
            )
        except CameraNameConflictError as error:
            _raise_api_error(
                status.HTTP_409_CONFLICT,
                "camera_name_conflict",
                "camera name is already in use",
                error,
            )
        except (CameraServiceError, IntegrityError) as error:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_camera_update",
                "camera update is invalid",
                error,
            )
        except RtspProbeError as error:
            _raise_probe_error(error)
        response.headers["ETag"] = _etag(updated.camera.version)
        source_changed = _source_changed(before, payload, updated, expected_version)
        threshold_changed = _threshold_changed(before, payload, updated, expected_version)
        try:
            await _apply_update(
                self._runtime,
                updated,
                payload,
                source_changed=source_changed,
                threshold_changed=threshold_changed,
            )
        except RuntimeEffectError as error:
            _raise_runtime_error(error)
        return updated.camera

    async def delete_camera(
        self,
        camera_id: UUID,
        response: Response,
        if_match: Annotated[str | None, Header(alias="If-Match")] = None,
        x_camera_version: Annotated[int | None, Header(alias="X-Camera-Version")] = None,
    ) -> None:
        """Commit a tombstone before closing detector, media, and WHEP resources."""
        expected_version = _expected_version(if_match, x_camera_version)
        try:
            async with self._database.transaction() as session:
                deleted = await self._cameras.delete(
                    session,
                    CameraId(camera_id),
                    expected_version=expected_version,
                )
        except StaleCameraVersionError as error:
            _raise_api_error(
                status.HTTP_409_CONFLICT,
                "stale_camera_version",
                "camera version is stale",
                error,
            )
        except (CameraRecordNotFoundError, CameraDeletedError) as error:
            _raise_api_error(
                status.HTTP_404_NOT_FOUND,
                "camera_not_found",
                "camera was not found",
                error,
            )
        except CameraServiceError as error:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_camera_delete",
                "camera deletion is invalid",
                error,
            )
        response.status_code = status.HTTP_204_NO_CONTENT
        response.headers["ETag"] = _etag(deleted.camera.version)
        try:
            await _apply_delete(self._runtime, deleted)
        except RuntimeEffectError as error:
            _raise_runtime_error(error)


def build_camera_router(
    *,
    database: TransactionProvider,
    cameras: CameraServiceProvider,
    runtime: CameraRuntimePort,
    require_session: SessionDependencyFactory,
) -> APIRouter:
    """Build the authenticated camera router with no implicit dependencies."""
    handlers = _CameraHandlers(
        CameraRouterDependencies(
            database=database,
            cameras=cameras,
            runtime=runtime,
            require_session=require_session,
        )
    )
    router = APIRouter(prefix="/api/cameras", tags=["cameras"])
    router.add_api_route(
        "",
        handlers.list_cameras,
        methods=["GET"],
        response_model=tuple[CameraResponse, ...],
        dependencies=[Depends(handlers.passive_dependency)],
    )
    router.add_api_route(
        "",
        handlers.create_camera,
        methods=["POST"],
        response_model=CameraResponse,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/test",
        handlers.test_camera,
        methods=["POST"],
        response_model=CameraTestResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/{camera_id}",
        handlers.update_camera,
        methods=["PATCH"],
        response_model=CameraResponse,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    router.add_api_route(
        "/{camera_id}",
        handlers.delete_camera,
        methods=["DELETE"],
        status_code=status.HTTP_204_NO_CONTENT,
        response_model=None,
        dependencies=[Depends(handlers.mutation_dependency)],
    )
    return router


async def _read_before_update(
    session: AsyncSession,
    cameras: CameraServiceProvider,
    camera_id: UUID,
) -> _CameraBeforeUpdate | None:
    """Read old source and threshold when the concrete service exposes its repository."""
    try:
        repository = cameras.repository
    except AttributeError:
        return None
    current = await repository.get(session, camera_id, for_update=True)
    return _CameraBeforeUpdate(
        source=await repository.source(session, current),
        threshold=current.detection_threshold,
    )


def _source_changed(
    before: _CameraBeforeUpdate | None,
    payload: CameraPatchRequest,
    mutation: CameraMutation,
    expected_version: int,
) -> bool:
    if payload.source_url is None:
        return False
    if before is not None:
        return str(payload.source_url) != before.source
    return mutation.camera.version > expected_version


def _threshold_changed(
    before: _CameraBeforeUpdate | None,
    payload: CameraPatchRequest,
    mutation: CameraMutation,
    expected_version: int,
) -> bool:
    if payload.detection_threshold is None:
        return False
    if before is not None:
        return payload.detection_threshold != before.threshold
    return mutation.camera.version > expected_version


async def _apply_create(
    runtime: CameraRuntimePort,
    mutation: CameraMutation,
    source_url: AnyUrl,
) -> None:
    """Publish the committed fence before live media and detector activation."""
    camera = mutation.camera
    await runtime.mark_committed(camera.camera_id, camera.version)
    if mutation.lifecycle.activation is not None:
        await _activate_bound_source(runtime, mutation.lifecycle.activation)
    else:
        _ = await runtime.activate_source(
            camera.camera_id,
            camera.version,
            parse_rtsp_source(source_url),
        )
    await _dispatch_plan(runtime, mutation.lifecycle)


async def _apply_update(
    runtime: CameraRuntimePort,
    mutation: CameraMutation,
    payload: CameraPatchRequest,
    *,
    source_changed: bool,
    threshold_changed: bool,
) -> None:
    """Apply cancellation, source replacement, threshold handoff, and activation."""
    camera = mutation.camera
    if camera.version == 1 and mutation.lifecycle.activation is None:
        return
    await runtime.mark_committed(camera.camera_id, camera.version)
    plan = mutation.lifecycle
    if plan.cancellation is not None:
        await runtime.cancel(plan.cancellation)
    if source_changed and payload.source_url is not None:
        if plan.activation is not None:
            await _activate_bound_source(runtime, plan.activation)
        else:
            source = _activation_source(plan, payload.source_url)
            _ = await runtime.activate_source(camera.camera_id, camera.version, source)
    if threshold_changed and plan.activation is None and payload.detection_threshold is not None:
        _ = await runtime.apply_threshold(
            camera.camera_id,
            camera.version,
            payload.detection_threshold,
        )
    if plan.activation is not None:
        if plan.activation.reason is CameraActivationReason.DETECTION_ENABLED:
            await _activate_bound_source(runtime, plan.activation)
        await runtime.activate(plan.activation)


async def _apply_delete(runtime: CameraRuntimePort, mutation: CameraMutation) -> None:
    """Fence the tombstone, then close every camera-scoped external resource."""
    camera = mutation.camera
    await runtime.mark_committed(camera.camera_id, camera.version, deleted=True)
    if mutation.lifecycle.cancellation is not None:
        await runtime.cancel(mutation.lifecycle.cancellation)
    await runtime.close_camera(camera.camera_id)


async def _dispatch_plan(runtime: CameraRuntimePort, plan: CameraLifecyclePlan) -> None:
    """Dispatch an explicit detector lifecycle plan in cancellation order."""
    if plan.cancellation is not None:
        await runtime.cancel(plan.cancellation)
    if plan.activation is not None:
        await runtime.activate(plan.activation)


async def _activate_bound_source(
    runtime: CameraRuntimePort,
    request: CameraActivationRequest,
) -> None:
    if isinstance(runtime, BoundCameraRuntimePort):
        _ = await runtime.activate_bound_source(request)
        return
    _ = await runtime.activate_source(request.camera_id, request.version, request.source)


def _activation_source(plan: CameraLifecyclePlan, source_url: AnyUrl) -> ParsedRtspSource:
    """Prefer the service's parsed source and parse disabled-detection edits once."""
    if plan.activation is not None and plan.activation.reason is CameraActivationReason.SOURCE_EDIT:
        return plan.activation.source
    return parse_rtsp_source(source_url)


def _expected_version(if_match: str | None, header_version: int | None) -> int:
    """Parse one optimistic version header and reject ambiguous/missing values."""
    if if_match is not None and header_version is not None:
        _raise_api_error(
            status.HTTP_400_BAD_REQUEST,
            "ambiguous_camera_version",
            "send one camera version header",
        )
    if header_version is not None:
        if header_version < 1:
            _raise_api_error(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                "invalid_camera_version",
                "camera version must be positive",
            )
        return header_version
    if if_match is None:
        _raise_api_error(
            status.HTTP_428_PRECONDITION_REQUIRED,
            "camera_version_required",
            "If-Match or X-Camera-Version is required",
        )
    candidate = if_match.strip()
    if candidate.startswith("W/"):
        candidate = candidate[2:].strip()
    if len(candidate) >= _ETAG_QUOTE_MIN_LENGTH and candidate[0] == candidate[-1] == '"':
        candidate = candidate[1:-1]
    try:
        version = int(candidate)
    except ValueError as error:
        _raise_api_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_camera_version",
            "camera version must be an integer",
            error,
        )
    if version < 1:
        _raise_api_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "invalid_camera_version",
            "camera version must be positive",
        )
    return version


def _etag(version: int) -> str:
    return f'W/"{version}"'


def _raise_probe_error(error: RtspProbeError) -> NoReturn:
    """Map bounded probe failures to stable sanitized API errors."""
    messages = {
        ProbeFailureCode.INVALID_SOURCE: "RTSP source is invalid",
        ProbeFailureCode.AUTHENTICATION_FAILED: "RTSP source authentication failed",
        ProbeFailureCode.UNREACHABLE: "RTSP source could not be reached",
        ProbeFailureCode.UNSUPPORTED_CODEC: "RTSP source must provide H.264 video",
        ProbeFailureCode.DECODE_FAILED: "RTSP source did not decode a video frame",
        ProbeFailureCode.TIMEOUT: "RTSP source probe timed out",
        ProbeFailureCode.PROCESS_UNAVAILABLE: "FFmpeg is unavailable for RTSP probing",
    }
    _raise_api_error(
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        error.code.value,
        messages[error.code],
        error,
    )


def _raise_runtime_error(error: RuntimeEffectError) -> NoReturn:
    """Report committed but unreconciled external effects explicitly."""
    _raise_api_error(
        status.HTTP_503_SERVICE_UNAVAILABLE,
        "camera_effect_pending",
        "camera state is committed but its external effect is pending; retry reconciliation",
        error,
    )


def _raise_api_error(
    status_code: int,
    code: str,
    message: str,
    cause: BaseException | None = None,
) -> NoReturn:
    """Raise a stable error without copying credentials or database internals."""
    del cause
    raise HTTPException(status_code=status_code, detail={"code": code, "message": message})


__all__ = [
    "AuthenticatedRequest",
    "CameraRouterDependencies",
    "SessionDependency",
    "SessionDependencyFactory",
    "build_camera_router",
]
