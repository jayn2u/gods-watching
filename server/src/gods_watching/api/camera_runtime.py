"""Version-fenced camera effects for media and detector integrations."""

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from enum import StrEnum
from inspect import Parameter, signature
from typing import Protocol, final, override, runtime_checkable

import anyio

from gods_watching.cameras.lifecycle import (
    CameraActivationRequest,
    CameraCancellationReason,
    CameraCancellationRequest,
)
from gods_watching.cameras.source_probe import ParsedRtspSource
from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.pipeline import GenerationBinding, PipelineGenerationEvent
from gods_watching.media.models import GenerationEventKind, SourceGenerationId


class RuntimeEffectError(RuntimeError):
    """Base class for a post-commit camera effect that could not be reconciled."""


@final
class RuntimeUnavailableError(RuntimeEffectError):
    """Report an integration capability that has not been supplied."""

    capability: str

    def __init__(self, capability: str) -> None:
        """Create an error for a missing integration capability."""
        self.capability = capability
        super().__init__(capability)

    @override
    def __str__(self) -> str:
        return f"camera runtime capability unavailable: {self.capability}"


@final
class StaleRuntimeEffectError(RuntimeEffectError):
    """Report an effect that is older than the committed camera fence."""

    camera_id: CameraId
    expected: int
    actual: int

    def __init__(self, camera_id: CameraId, expected: int, actual: int) -> None:
        """Create an error for an effect behind the committed fence."""
        self.camera_id = camera_id
        self.expected = expected
        self.actual = actual
        super().__init__(camera_id, expected, actual)

    @override
    def __str__(self) -> str:
        return "camera runtime effect is stale"


class ThresholdHandoff(StrEnum):
    """Describe whether a detector threshold reached an available worker."""

    APPLIED = "applied"
    DEFERRED = "deferred"


class CameraGenerationEventSink(Protocol):
    """Receive generation events only after the API owner has bound their identity."""

    async def publish(self, event: PipelineGenerationEvent) -> None:
        """Deliver one complete camera/session/source-generation event."""
        ...


@runtime_checkable
class CameraGenerationOwner(Protocol):
    """Own one reconnect transaction and return its complete replacement binding."""

    async def reconnect(
        self,
        previous: GenerationBinding,
    ) -> GenerationBinding:
        """End the previous database session and return a fresh binding."""
        ...


class CameraReconnectFactory(Protocol):
    """Allocate a committed reconnect request before media assigns its ID."""

    async def __call__(
        self,
        previous: GenerationBinding,
    ) -> CameraActivationRequest | GenerationBinding:
        """Return a committed activation request or a completed replacement binding."""
        ...


class CameraMediaPort(Protocol):
    """Provide authenticated MediaMTX source-path effects."""

    async def activate(
        self,
        camera_id: CameraId,
        source_url: str,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId:
        """Create or replace a camera's live source path."""
        ...

    async def deactivate(
        self,
        camera_id: CameraId,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        """Remove a camera's live source path."""
        ...


class CameraDetectorPort(Protocol):
    """Provide worker lifecycle effects without embedding inference in the API."""

    async def activate(self, request: CameraActivationRequest) -> None:
        """Start one versioned detector generation."""
        ...

    async def cancel(self, request: CameraCancellationRequest) -> None:
        """Stop one versioned detector generation."""
        ...

    async def apply_threshold(self, camera_id: CameraId, threshold: float) -> None:
        """Apply a threshold to subsequent detector work."""
        ...


class CameraRuntimePort(Protocol):
    """Typed camera capability consumed by authenticated HTTP composition."""

    async def activate(self, request: CameraActivationRequest) -> None:
        """Activate one committed detector lifecycle plan."""
        ...

    async def cancel(self, request: CameraCancellationRequest) -> None:
        """Cancel one committed detector lifecycle plan."""
        ...

    async def close_camera(self, camera_id: CameraId) -> None:
        """Close live media and resources associated with one deleted camera."""
        ...

    async def mark_committed(
        self,
        camera_id: CameraId,
        version: int,
        *,
        deleted: bool = False,
    ) -> None:
        """Advance the fence before starting post-commit network effects."""
        ...

    async def activate_source(
        self,
        camera_id: CameraId,
        version: int,
        source: ParsedRtspSource,
    ) -> SourceGenerationId | None:
        """Activate live media for one committed source generation."""
        ...

    async def apply_threshold(
        self,
        camera_id: CameraId,
        version: int,
        threshold: float,
    ) -> ThresholdHandoff:
        """Deliver or retain a threshold for future detector work."""
        ...


@runtime_checkable
class BoundCameraRuntimePort(Protocol):
    """Optional runtime capability for binding media to a DB generation."""

    async def activate_bound_source(
        self,
        request: CameraActivationRequest,
    ) -> SourceGenerationId | None:
        """Activate media and publish a START event carrying the complete binding."""
        ...


@final
class CommittedStateDetector(CameraDetectorPort):
    """Accept detector effects that the separate pipeline worker reads from the commit.

    The pipeline worker polls committed camera sessions, versions, and thresholds, so the
    API has no in-process detector to call; these effects are complete once committed.
    """

    @override
    async def activate(self, request: CameraActivationRequest) -> None:
        """Leave activation to the worker's next reconcile of the committed session."""
        del request

    @override
    async def cancel(self, request: CameraCancellationRequest) -> None:
        """Leave cancellation to the worker's next reconcile of the ended session."""
        del request

    @override
    async def apply_threshold(self, camera_id: CameraId, threshold: float) -> None:
        """Leave the threshold to the worker, which rebinds on the committed version."""
        del camera_id, threshold


class _CameraClosePort(Protocol):
    async def __call__(self, camera_id: CameraId) -> None: ...


@final
class _CameraState:
    version: int
    deleted: bool
    threshold: float | None
    binding: GenerationBinding | None
    source_url: str | None

    def __init__(self, version: int, *, deleted: bool = False) -> None:
        self.version = version
        self.deleted = deleted
        self.threshold = None
        self.binding = None
        self.source_url = None


@final
class CameraRuntime(CameraRuntimePort):
    """Serialize camera effects and reject stale callbacks by committed version."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        detector: CameraDetectorPort | None,
        media: CameraMediaPort | None,
        close_resources: _CameraClosePort | None = None,
        close_detector: _CameraClosePort | None = None,
        generation_sink: CameraGenerationEventSink | None = None,
        generation_owner: CameraReconnectFactory | CameraGenerationOwner | None = None,
    ) -> None:
        """Bind optional detector, media, generation, and resource-close capabilities."""
        self._detector: CameraDetectorPort | None = detector
        self._media: CameraMediaPort | None = media
        self._close_resources: _CameraClosePort | None = close_resources
        self._close_detector: _CameraClosePort | None = close_detector
        self._generation_sink: CameraGenerationEventSink | None = generation_sink
        self._generation_owner: CameraReconnectFactory | CameraGenerationOwner | None = (
            generation_owner
        )
        self._states: dict[CameraId, _CameraState] = {}
        self._locks: dict[CameraId, anyio.Lock] = {}

    @override
    async def mark_committed(
        self,
        camera_id: CameraId,
        version: int,
        *,
        deleted: bool = False,
    ) -> None:
        """Publish a database commit to the effect fence before network work."""
        async with self._lock_for(camera_id):
            state = self._states.get(camera_id)
            if state is not None and version < state.version:
                raise StaleRuntimeEffectError(camera_id, version, state.version)
            if state is not None and state.deleted and not deleted and version <= state.version:
                raise StaleRuntimeEffectError(camera_id, version, state.version)
            if state is None:
                self._states[camera_id] = _CameraState(version, deleted=deleted)
            else:
                state.version = version
                state.deleted = deleted
                if state.binding is not None and not deleted:
                    state.binding = replace(state.binding, camera_version=version)

    @override
    async def activate_source(
        self,
        camera_id: CameraId | CameraActivationRequest,
        version: int | None = None,
        source: ParsedRtspSource | None = None,
        *,
        publish_started: bool = True,
        publish_events: bool | None = None,
    ) -> SourceGenerationId | None:
        """Replace live media, optionally binding the returned source generation.

        The request form is the authoritative path for detector work.  The
        three-argument form remains available to callers that only need live
        media, including cameras whose detection is disabled.
        """
        if isinstance(camera_id, CameraActivationRequest):
            request: CameraActivationRequest | None = camera_id
            actual_camera_id: CameraId = request.camera_id
            actual_version: int | None = request.version
            actual_source: ParsedRtspSource | None = request.source
        else:
            request = None
            actual_camera_id = camera_id
            actual_version = version
            actual_source = source
        if publish_events is not None and request is not None:
            publish_started = publish_events
        async with self._lock_for(actual_camera_id):
            if actual_version is None or actual_source is None:
                detail = "version and source are required for an unbound activation"
                raise TypeError(detail)
            state = self._require_current(actual_camera_id, actual_version)
            if state.deleted:
                raise StaleRuntimeEffectError(actual_camera_id, actual_version, state.version)
            if self._media is None:
                capability = "media"
                raise RuntimeUnavailableError(capability)
            if state.binding is not None:
                await self._end_binding(state, kind=GenerationEventKind.ENDED)
            state.source_url = actual_source.url
            should_publish_legacy = request is None and self._generation_sink is None
            if publish_events is not None:
                should_publish_legacy = publish_events
            source_generation_id = await _activate_media(
                self._media,
                actual_camera_id,
                actual_source.url,
                publish_events=should_publish_legacy,
            )
            if request is None:
                return source_generation_id
            binding = GenerationBinding(
                camera_id=request.camera_id,
                camera_session_id=request.session_id,
                db_generation_id=request.generation_id,
                source_generation_id=source_generation_id,
                camera_version=request.version,
            )
            state.binding = binding
            if publish_started:
                await self._publish_generation(binding, GenerationEventKind.STARTED)
            return source_generation_id

    async def activate_bound_source(
        self,
        request: CameraActivationRequest,
    ) -> SourceGenerationId | None:
        """Activate one DB generation and publish its complete START event."""
        return await self.activate_source(request)

    @override
    async def activate(self, request: CameraActivationRequest) -> None:
        """Start detector work only for the current committed generation."""
        async with self._lock_for(request.camera_id):
            state = self._require_current(request.camera_id, request.version)
            if state.deleted:
                raise StaleRuntimeEffectError(request.camera_id, request.version, state.version)
            if self._detector is None:
                capability = "detector"
                raise RuntimeUnavailableError(capability)
            try:
                await self._detector.activate(request)
            except RuntimeEffectError:
                raise
            except Exception as error:
                detail = "detector activation failed"
                raise RuntimeEffectError(detail) from error

    @override
    async def cancel(self, request: CameraCancellationRequest) -> None:
        """Cancel detector work unless a newer commit has already superseded it."""
        async with self._lock_for(request.camera_id):
            if self._detector is not None:
                try:
                    await self._detector.cancel(request)
                except RuntimeEffectError:
                    raise
                except Exception as error:
                    detail = "detector cancellation failed"
                    raise RuntimeEffectError(detail) from error
            state = self._states.get(request.camera_id)
            if state is None:
                return
            binding = state.binding
            if binding is None or not _same_database_generation(binding, request):
                return
            await self._end_binding(state, kind=GenerationEventKind.ENDED)
            match request.reason:
                case CameraCancellationReason.SOURCE_EDIT:
                    deactivate_media = True
                case CameraCancellationReason.DETECTION_DISABLED | CameraCancellationReason.DELETED:
                    deactivate_media = False
            if deactivate_media and self._media is not None:
                _ = await _deactivate_media(self._media, request.camera_id, publish_events=False)

    @override
    async def apply_threshold(
        self,
        camera_id: CameraId,
        version: int,
        threshold: float,
    ) -> ThresholdHandoff:
        """Apply a threshold now or retain it until the detector worker exists."""
        async with self._lock_for(camera_id):
            state = self._require_current(camera_id, version)
            if state.deleted:
                raise StaleRuntimeEffectError(camera_id, version, state.version)
            state.threshold = threshold
            if self._detector is None:
                return ThresholdHandoff.DEFERRED
            try:
                await self._detector.apply_threshold(camera_id, threshold)
            except RuntimeEffectError:
                raise
            except Exception as error:
                detail = "detector threshold handoff failed"
                raise RuntimeEffectError(detail) from error
            return ThresholdHandoff.APPLIED

    @override
    async def close_camera(self, camera_id: CameraId) -> None:
        """Close detector, live media, and WHEP resources for one camera."""
        async with self._lock_for(camera_id):
            state = self._states.get(camera_id)
            if state is not None:
                state.deleted = True
                if state.binding is not None:
                    await self._end_binding(state, kind=GenerationEventKind.ENDED)
            await _close_effect(self._close_detector, camera_id, "detector close failed")
            if self._media is not None:
                _ = await _deactivate_media(self._media, camera_id, publish_events=False)
            await _close_effect(
                self._close_resources,
                camera_id,
                "camera resource close failed",
            )

    async def reconnect(self, previous: GenerationBinding) -> GenerationBinding:
        """End one binding and return a fresh DB/media/source binding after EOF."""
        async with self._lock_for(previous.camera_id):
            state = self._states.get(previous.camera_id)
            if (
                state is None
                or state.binding is None
                or not _same_generation_identity(state.binding, previous)
                or previous.camera_version > state.version
            ):
                actual = state.version if state is not None else previous.camera_version
                raise StaleRuntimeEffectError(previous.camera_id, previous.camera_version, actual)
            if state.source_url is None:
                detail = "camera source is unavailable for reconnect"
                raise RuntimeEffectError(detail)
            current = state.binding
            await self._publish_generation(current, GenerationEventKind.SOURCE_LOST)
            await self._publish_generation(current, GenerationEventKind.ENDED)
            state.binding = None
            owner = self._generation_owner
            if owner is None:
                capability = "generation-owner"
                raise RuntimeUnavailableError(capability)
            replacement = await _request_reconnect(owner, previous)
            if isinstance(replacement, GenerationBinding):
                _validate_replacement(previous, replacement)
                state.binding = replacement
                state.version = replacement.camera_version
                await self._publish_generation(replacement, GenerationEventKind.STARTED)
                return replacement
            _validate_replacement_request(previous, replacement)
            state.source_url = replacement.source.url
            source_generation_id = await _activate_media(
                self._media,
                replacement.camera_id,
                replacement.source.url,
                publish_events=False,
            )
            binding = GenerationBinding(
                camera_id=replacement.camera_id,
                camera_session_id=replacement.session_id,
                db_generation_id=replacement.generation_id,
                source_generation_id=source_generation_id,
                camera_version=replacement.version,
            )
            state.version = replacement.version
            state.binding = binding
            await self._publish_generation(binding, GenerationEventKind.STARTED)
            return binding

    async def source_lost(self, camera_id: CameraId) -> GenerationBinding | None:
        """End a source binding after an outage without allocating a replacement."""
        async with self._lock_for(camera_id):
            state = self._states.get(camera_id)
            if state is None or state.binding is None:
                return None
            binding = state.binding
            await self._publish_generation(binding, GenerationEventKind.SOURCE_LOST)
            await self._publish_generation(binding, GenerationEventKind.ENDED)
            state.binding = None
            if self._media is not None:
                _ = await _deactivate_media(self._media, camera_id, publish_events=False)
            return binding

    async def active_binding(self, camera_id: CameraId) -> GenerationBinding | None:
        """Return the current complete binding without allocating or publishing work."""
        async with self._lock_for(camera_id):
            state = self._states.get(camera_id)
            return state.binding if state is not None else None

    async def _end_binding(
        self,
        state: _CameraState,
        *,
        kind: GenerationEventKind,
    ) -> None:
        binding = state.binding
        if binding is None:
            return
        state.binding = None
        await self._publish_generation(binding, kind)

    async def _publish_generation(
        self,
        binding: GenerationBinding,
        kind: GenerationEventKind,
    ) -> None:
        sink = self._generation_sink
        if sink is None:
            return
        try:
            await sink.publish(
                PipelineGenerationEvent(
                    binding=binding,
                    kind=kind,
                    occurred_utc=datetime.now(UTC),
                )
            )
        except RuntimeEffectError:
            raise
        except Exception as error:
            detail = "generation event publication failed"
            raise RuntimeEffectError(detail) from error

    def _lock_for(self, camera_id: CameraId) -> anyio.Lock:
        return self._locks.setdefault(camera_id, anyio.Lock())

    def _require_current(self, camera_id: CameraId, version: int) -> _CameraState:
        state = self._states.get(camera_id)
        if state is None:
            state = _CameraState(version)
            self._states[camera_id] = state
        elif version < state.version:
            raise StaleRuntimeEffectError(camera_id, version, state.version)
        return state


__all__ = [
    "BoundCameraRuntimePort",
    "CameraDetectorPort",
    "CameraGenerationEventSink",
    "CameraGenerationOwner",
    "CameraMediaPort",
    "CameraReconnectFactory",
    "CameraRuntime",
    "CameraRuntimePort",
    "CommittedStateDetector",
    "RuntimeEffectError",
    "RuntimeUnavailableError",
    "StaleRuntimeEffectError",
    "ThresholdHandoff",
]


async def _close_effect(
    effect: Callable[[CameraId], Awaitable[None]] | None,
    camera_id: CameraId,
    detail: str,
) -> None:
    """Run one optional close callback and classify external failures."""
    if effect is None:
        return
    try:
        await effect(camera_id)
    except RuntimeEffectError:
        raise
    except Exception as error:
        raise RuntimeEffectError(detail) from error


async def _activate_media(
    media: CameraMediaPort | None,
    camera_id: CameraId,
    source_url: str,
    *,
    publish_events: bool,
) -> SourceGenerationId:
    if media is None:
        capability = "media"
        raise RuntimeUnavailableError(capability)
    activate = media.activate
    try:
        if _accepts_keyword(activate, "publish_events"):
            return await activate(camera_id, source_url, publish_events=publish_events)
        return await activate(camera_id, source_url)
    except RuntimeEffectError:
        raise
    except Exception as error:
        detail = "media activation failed"
        raise RuntimeEffectError(detail) from error


async def _deactivate_media(
    media: CameraMediaPort,
    camera_id: CameraId,
    *,
    publish_events: bool,
) -> SourceGenerationId | None:
    deactivate = media.deactivate
    try:
        if _accepts_keyword(deactivate, "publish_events"):
            return await deactivate(camera_id, publish_events=publish_events)
        return await deactivate(camera_id)
    except RuntimeEffectError:
        raise
    except Exception as error:
        detail = "media deactivation failed"
        raise RuntimeEffectError(detail) from error


def _accepts_keyword(method: Callable[..., object], name: str) -> bool:
    parameters = signature(method).parameters.values()
    return any(
        parameter.name == name or parameter.kind is Parameter.VAR_KEYWORD
        for parameter in parameters
    )


async def _request_reconnect(
    owner: CameraReconnectFactory | CameraGenerationOwner,
    previous: GenerationBinding,
) -> CameraActivationRequest | GenerationBinding:
    if isinstance(owner, CameraGenerationOwner):
        return await owner.reconnect(previous)
    return await owner(previous)


def _same_database_generation(
    binding: GenerationBinding,
    request: CameraCancellationRequest,
) -> bool:
    return (
        binding.camera_id == request.camera_id
        and binding.camera_session_id == request.session_id
        and binding.db_generation_id == request.generation_id
    )


def _same_generation_identity(first: GenerationBinding, second: GenerationBinding) -> bool:
    return (
        first.camera_id == second.camera_id
        and first.camera_session_id == second.camera_session_id
        and first.db_generation_id == second.db_generation_id
        and first.source_generation_id == second.source_generation_id
    )


def _validate_replacement(previous: GenerationBinding, replacement: GenerationBinding) -> None:
    if replacement.camera_id != previous.camera_id:
        detail = "generation owner changed camera identity"
        raise RuntimeEffectError(detail)
    if replacement.camera_version < previous.camera_version:
        raise StaleRuntimeEffectError(
            replacement.camera_id,
            replacement.camera_version,
            previous.camera_version,
        )
    if replacement.camera_session_id == previous.camera_session_id:
        detail = "generation owner reused the camera session"
        raise RuntimeEffectError(detail)
    if replacement.db_generation_id == previous.db_generation_id:
        detail = "generation owner reused the database generation"
        raise RuntimeEffectError(detail)
    if replacement.source_generation_id == previous.source_generation_id:
        detail = "generation owner reused the source generation"
        raise RuntimeEffectError(detail)


def _validate_replacement_request(
    previous: GenerationBinding,
    replacement: CameraActivationRequest,
) -> None:
    if replacement.camera_id != previous.camera_id:
        detail = "generation owner changed camera identity"
        raise RuntimeEffectError(detail)
    if replacement.version < previous.camera_version:
        raise StaleRuntimeEffectError(
            replacement.camera_id,
            replacement.version,
            previous.camera_version,
        )
    if replacement.session_id == previous.camera_session_id:
        detail = "generation owner reused the camera session"
        raise RuntimeEffectError(detail)
    if replacement.generation_id == previous.db_generation_id:
        detail = "generation owner reused the database generation"
        raise RuntimeEffectError(detail)
