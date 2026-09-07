"""Durable camera configuration mutations and lifecycle plans."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.cameras import (
    CameraCreateRequest,
    CameraPatchRequest,
    CameraResponse,
    CameraTestResponse,
)
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.storage import Camera, CameraSession

from .lifecycle import (
    CameraActivationReason,
    CameraActivationRequest,
    CameraCancellationReason,
    CameraCancellationRequest,
    CameraGenerationId,
    CameraLifecyclePlan,
    CameraLifecyclePort,
)
from .repository import CameraRepository, StaleCameraVersionError
from .source_probe import ParsedRtspSource, ProbeResult, RtspSourceProbe, parse_rtsp_source


class SourceProbePort(Protocol):
    """Probe one already-parsed trusted source without returning media bytes."""

    async def probe(self, source: ParsedRtspSource) -> ProbeResult:
        """Decode one frame and return sanitized stream metadata."""
        ...


@dataclass(frozen=True, slots=True)
class CameraMutation:
    """Return durable camera state plus explicit work for the later runtime layer."""

    camera: CameraResponse
    lifecycle: CameraLifecyclePlan


@dataclass(frozen=True, slots=True)
class _LifecycleUpdate:
    camera: Camera
    old_version: int
    old_enabled: bool
    new_enabled: bool
    source_changed: bool
    source: ParsedRtspSource | None
    old_source: str


@dataclass(frozen=True, slots=True)
class CameraService:
    """Persist camera settings while leaving media activation to an injected capability."""

    repository: CameraRepository
    source_probe: SourceProbePort | None = None

    async def create(
        self,
        session: AsyncSession,
        request: CameraCreateRequest,
    ) -> CameraMutation:
        """Create a camera and, when enabled, allocate its first generation."""
        source = parse_rtsp_source(request.source_url)
        await self._probe_if_configured(source)
        camera = await self.repository.create(session, request)
        activation: CameraActivationRequest | None = None
        if request.detection_enabled:
            camera_session = await self.repository.start_session(
                session,
                camera.id,
                cause=CameraActivationReason.CREATED.value,
            )
            activation = _activation(camera, camera_session, source, CameraActivationReason.CREATED)
        await session.flush()
        return CameraMutation(_camera_response(camera), CameraLifecyclePlan(activation, None))

    async def update(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        request: CameraPatchRequest,
        *,
        expected_version: int,
    ) -> CameraMutation:
        """Apply one version-fenced patch and return ordered runtime work."""
        source = parse_rtsp_source(request.source_url) if request.source_url is not None else None
        if source is not None:
            await self._probe_if_configured(source)
        camera = await self.repository.get(session, camera_id, for_update=True)
        if camera.version != expected_version:
            raise StaleCameraVersionError(expected=expected_version, actual=camera.version)
        if request.name is not None and request.name != camera.name:
            await self.repository.ensure_name_available(
                session,
                request.name,
                exclude_camera_id=camera.id,
            )
        old_enabled = camera.detection_enabled
        old_threshold = camera.detection_threshold
        old_source = await self.repository.source(session, camera)
        source_changed = source is not None and source.url != old_source
        name_changed = request.name is not None and request.name != camera.name
        new_enabled = (
            request.detection_enabled if request.detection_enabled is not None else old_enabled
        )
        camera.name = request.name if request.name is not None else camera.name
        if source_changed and source is not None:
            camera.source_ciphertext = self.repository.storage.credential_cipher.encrypt(source.url)
            camera.source_host = source.host
            camera.source_port = source.port
        camera.detection_enabled = new_enabled
        if request.detection_threshold is not None:
            camera.detection_threshold = request.detection_threshold
        threshold_changed = (
            request.detection_threshold is not None and request.detection_threshold != old_threshold
        )
        changed = source_changed or old_enabled != new_enabled or name_changed or threshold_changed
        if not changed:
            return CameraMutation(_camera_response(camera), CameraLifecyclePlan(None, None))
        old_version = camera.version
        camera.version += 1
        lifecycle = await self._update_lifecycle(
            session,
            _LifecycleUpdate(
                camera=camera,
                old_version=old_version,
                old_enabled=old_enabled,
                new_enabled=new_enabled,
                source_changed=source_changed,
                source=source,
                old_source=old_source,
            ),
        )
        await session.flush()
        return CameraMutation(
            _camera_response(camera),
            lifecycle,
        )

    async def delete(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        *,
        expected_version: int,
    ) -> CameraMutation:
        """Soft-delete a camera, retaining its label and appearance history."""
        camera = await self.repository.get(session, camera_id, for_update=True)
        if camera.version != expected_version:
            raise StaleCameraVersionError(expected=expected_version, actual=camera.version)
        old_version = camera.version
        active = await self.repository.active_session(session, camera.id, for_update=True)
        cancellation: CameraCancellationRequest | None = None
        if active is not None:
            await self.repository.end_session(session, active)
            if camera.detection_enabled:
                cancellation = _cancellation(
                    camera,
                    active,
                    old_version,
                    CameraCancellationReason.DELETED,
                )
        camera.deleted_at = datetime.now(UTC)
        camera.detection_enabled = False
        camera.version += 1
        await session.flush()
        return CameraMutation(
            _camera_response(camera),
            CameraLifecyclePlan(activation=None, cancellation=cancellation),
        )

    async def get(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        *,
        include_deleted: bool = False,
    ) -> CameraResponse:
        """Read one camera without ever decrypting its source for the caller."""
        camera = await self.repository.get(session, camera_id, include_deleted=include_deleted)
        return _camera_response(camera)

    async def list(
        self,
        session: AsyncSession,
        *,
        include_deleted: bool = False,
    ) -> tuple[CameraResponse, ...]:
        """Read stable camera rows without source credentials."""
        return tuple(
            _camera_response(camera)
            for camera in await self.repository.list(session, include_deleted=include_deleted)
        )

    async def test_source(self, source_url: str) -> CameraTestResponse:
        """Parse and probe an operator-supplied source with the fixed deadline."""
        source = parse_rtsp_source(source_url)
        probe = self.source_probe or RtspSourceProbe()
        return (await probe.probe(source)).to_response()

    async def dispatch(
        self,
        plan: CameraLifecyclePlan,
        runtime: CameraLifecyclePort,
    ) -> None:
        """Run explicit cancellation then activation callbacks supplied by 12c/18."""
        if plan.cancellation is not None:
            await runtime.cancel(plan.cancellation)
        if plan.activation is not None:
            await runtime.activate(plan.activation)

    async def _probe_if_configured(self, source: ParsedRtspSource) -> None:
        if self.source_probe is not None:
            _ = await self.source_probe.probe(source)

    async def _update_lifecycle(
        self,
        session: AsyncSession,
        update: _LifecycleUpdate,
    ) -> CameraLifecyclePlan:
        camera = update.camera
        old_enabled = update.old_enabled
        new_enabled = update.new_enabled
        source_changed = update.source_changed
        if not source_changed and old_enabled == new_enabled:
            return CameraLifecyclePlan(activation=None, cancellation=None)
        active = await self.repository.active_session(session, camera.id, for_update=True)
        cancellation = self._cancel_old_generation(
            camera=camera,
            active=active,
            old_version=update.old_version,
            old_enabled=old_enabled,
            source_changed=source_changed,
        )
        if active is not None:
            await self.repository.end_session(session, active)
        if not new_enabled or (not source_changed and old_enabled):
            return CameraLifecyclePlan(activation=None, cancellation=cancellation)
        activation_source = update.source or parse_rtsp_source(update.old_source)
        reason = (
            CameraActivationReason.SOURCE_EDIT
            if source_changed
            else CameraActivationReason.DETECTION_ENABLED
        )
        new_session = await self.repository.start_session(
            session,
            camera.id,
            cause=reason.value,
        )
        activation = _activation(camera, new_session, activation_source, reason)
        return CameraLifecyclePlan(activation=activation, cancellation=cancellation)

    @staticmethod
    def _cancel_old_generation(
        *,
        camera: Camera,
        active: CameraSession | None,
        old_version: int,
        old_enabled: bool,
        source_changed: bool,
    ) -> CameraCancellationRequest | None:
        if active is None or not old_enabled:
            return None
        reason = (
            CameraCancellationReason.SOURCE_EDIT
            if source_changed
            else CameraCancellationReason.DETECTION_DISABLED
        )
        return _cancellation(camera, active, old_version, reason)


def _camera_response(camera: Camera) -> CameraResponse:
    return CameraResponse(
        camera_id=CameraId(camera.id),
        name=camera.name,
        source_host=camera.source_host,
        source_port=camera.source_port,
        detection_enabled=camera.detection_enabled,
        detection_threshold=camera.detection_threshold,
        version=camera.version,
        deleted_at=camera.deleted_at,
    )


def _activation(
    camera: Camera,
    camera_session: CameraSession,
    source: ParsedRtspSource,
    reason: CameraActivationReason,
) -> CameraActivationRequest:
    return CameraActivationRequest(
        camera_id=CameraId(camera.id),
        version=camera.version,
        session_id=CameraSessionId(camera_session.id),
        generation_id=CameraGenerationId(camera_session.generation_id),
        source=source,
        detection_threshold=camera.detection_threshold,
        reason=reason,
    )


def _cancellation(
    camera: Camera,
    camera_session: CameraSession,
    version: int,
    reason: CameraCancellationReason,
) -> CameraCancellationRequest:
    return CameraCancellationRequest(
        camera_id=CameraId(camera.id),
        version=version,
        session_id=CameraSessionId(camera_session.id),
        generation_id=CameraGenerationId(camera_session.generation_id),
        reason=reason,
    )
