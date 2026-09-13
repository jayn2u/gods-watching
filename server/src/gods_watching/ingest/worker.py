"""Bounded camera ingest, detector sampling, tracking, and pipeline handoff."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING, Protocol, override

import anyio

from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.pipeline import CropCandidate, GenerationBinding, PipelineHandoff
from gods_watching.inference.detector import Detection, DetectorRequest, DetectorResult
from gods_watching.media.models import SourceGenerationId
from gods_watching.tracking import (
    ByteTrackScope,
    DetectionFrame,
    DetectorInputReference,
    LifecycleKind,
    ResetReason,
    TrackingScope,
    TrackLifecycle,
)

from .crop import CropExtractionError, bounded_crop_dimensions, extract_rgb_crop
from .decoder import (
    DecoderConfiguration,
    DecoderStatusSink,
    GenerationReconnect,
    PyAvRtspDecoder,
)
from .models import DecodedFrame, IngestStats, IngestStatsSnapshot, LatestFrameSlot
from .scheduler import FairRoundRobinScheduler

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

_MIN_THRESHOLD = 0.1
_MAX_THRESHOLD = 0.95
_MAX_RETAINED_FRAMES = 256


class DetectorPort(Protocol):
    """Describe the detector operation required by one camera worker."""

    async def detect(self, request: DetectorRequest) -> DetectorResult:
        """Return person detections for one sampled encoded frame."""
        ...


PipelineHandoffConsumer = Callable[[PipelineHandoff], Awaitable[None]]
FrameClock = Callable[[], float]
UtcClock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class WorkerConfigurationError(ValueError):
    """Describe an invalid bounded worker setting."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class StaleGenerationError(RuntimeError):
    """Describe a frame or lifecycle action from an old source generation."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class IngestWorkerConfiguration:
    """Carry authorized source identity and bounded worker timing settings."""

    generation: GenerationBinding
    source_url: str
    threshold: float = 0.5
    detector_deadline_seconds: float = 2.0
    monotonic_clock: FrameClock = monotonic
    utc_clock: UtcClock | None = None
    generation_reconnect: GenerationReconnect | None = None
    detection_enabled: bool = True


class IngestWorker:
    """Own one camera's latest-frame slot and camera/session-scoped tracker."""

    def __init__(
        self,
        config: IngestWorkerConfiguration,
        detector: DetectorPort,
        handoff_consumer: PipelineHandoffConsumer,
        decoder: PyAvRtspDecoder | None = None,
    ) -> None:
        """Create a worker whose source and generation are already authorized."""
        if (
            not isfinite(config.threshold)
            or not _MIN_THRESHOLD <= config.threshold <= _MAX_THRESHOLD
        ):
            raise WorkerConfigurationError(detail="threshold must be between 0.1 and 0.95")
        if (
            not isfinite(config.detector_deadline_seconds)
            or config.detector_deadline_seconds <= 0.0
        ):
            raise WorkerConfigurationError(detail="detector deadline must be positive and finite")
        self._generation: GenerationBinding = config.generation
        self._source_url: str = config.source_url
        self._threshold: float = config.threshold
        self._detection_enabled: bool = config.detection_enabled
        self._detector: DetectorPort = detector
        self._handoff_consumer: PipelineHandoffConsumer = handoff_consumer
        self._detector_deadline_seconds: float = config.detector_deadline_seconds
        self._monotonic_clock: FrameClock = config.monotonic_clock
        self._utc_clock: UtcClock = config.utc_clock or (lambda: datetime.now(UTC))
        self._generation_reconnect: GenerationReconnect | None = config.generation_reconnect
        self._slot: LatestFrameSlot = LatestFrameSlot()
        self._detector_lock: anyio.Lock = anyio.Lock()
        self._stats: IngestStats = IngestStats()
        self._tracking: ByteTrackScope = self._new_tracking_scope(config.threshold)
        self._frames_by_reference: dict[DetectorInputReference, DecodedFrame] = {}
        self._last_dimensions: tuple[int, int] | None = None
        self._closed: bool = False
        self._decoder: PyAvRtspDecoder = decoder or PyAvRtspDecoder(
            DecoderConfiguration(
                source_url=config.source_url,
                source_generation_id=config.generation.source_generation_id,
                slot=self._slot,
                frame_sink=self._record_decoded,
                generation_binding=config.generation,
                generation_reconnect=self._reconnect_generation,
            )
        )

    @property
    def camera_id(self) -> CameraId:
        """Return the camera namespace owned by this worker."""
        return self._generation.camera_id

    @property
    def generation(self) -> GenerationBinding:
        """Return the immutable generation fence for this worker."""
        return self._generation

    @property
    def slot(self) -> LatestFrameSlot:
        """Return the one bounded latest-frame slot."""
        return self._slot

    @property
    def stats(self) -> IngestStatsSnapshot:
        """Return current truthful ingest counters and freshness values."""
        return self._stats.snapshot(
            now_monotonic=self._monotonic_clock(),
            dropped_frames=self._slot.dropped_frames,
        )

    @property
    def tracking(self) -> ByteTrackScope:
        """Return the camera-local tracker for inspection and status reporting."""
        return self._tracking

    @property
    def threshold(self) -> float:
        """Return the threshold used by subsequent detector requests."""
        return self._threshold

    @property
    def detection_enabled(self) -> bool:
        """Return whether this worker currently sends frames to the detector."""
        return self._detection_enabled

    def receive(self, frame: DecodedFrame) -> None:
        """Publish one decoded frame after checking its source-generation fence."""
        if self._closed:
            raise StaleGenerationError(detail="frame arrived after worker close")
        self._ensure_generation(frame.source_generation_id)
        self._record_decoded(frame)
        self._slot.put(frame)

    async def sample_once(self) -> tuple[PipelineHandoff, ...]:
        """Sample the newest frame and deliver ordered lifecycle handoffs."""
        self._stats.record_dispatch()
        frame = self._take_dispatch_frame()
        if frame is None:
            return ()
        generation = self._generation
        tracking = self._tracking
        self._stats.record_sampled()
        self._remember_frame(frame)
        detector_requests_before = self._stats.detector_requests
        result = await self._request_detection(frame, generation=generation, tracking=tracking)
        self._stats.record_dispatch_outcome(
            "detector_requested"
            if self._stats.detector_requests > detector_requests_before
            else "generation_fenced"
        )
        if result is None:
            return ()
        if self._closed or self._generation != generation or self._tracking is not tracking:
            self._stats.record_error("detector result crossed the active generation")
            return ()
        result = self._crop_eligible_result(frame, result)
        detector_result_monotonic = max(self._monotonic_clock(), frame.ingress_monotonic)
        detection_frame = DetectionFrame(
            camera_id=generation.camera_id,
            session_id=generation.camera_session_id,
            source_generation_id=generation.source_generation_id,
            ingress_utc=frame.ingress_utc,
            ingress_monotonic=frame.ingress_monotonic,
            detector_result_monotonic=detector_result_monotonic,
            source_width=frame.width,
            source_height=frame.height,
            detector_input_reference=frame.reference,
            detections=result.detections,
        )
        try:
            lifecycles = tracking.feed(detection_frame)
        except (RuntimeError, ValueError):
            self._stats.record_error("tracking update failed")
            return ()
        return await self._deliver(lifecycles, generation=generation)

    def _take_dispatch_frame(self) -> DecodedFrame | None:
        if self._closed:
            self._stats.record_dispatch_outcome("closed")
            return None
        frame = self._slot.take_latest()
        if frame is None:
            self._stats.record_dispatch_outcome("no_frame")
            return None
        self._last_dimensions = (frame.width, frame.height)
        if not self._detection_enabled:
            self._stats.record_dispatch_outcome("detection_disabled")
            return None
        return frame

    def _crop_eligible_result(self, frame: DecodedFrame, result: DetectorResult) -> DetectorResult:
        eligible = tuple(
            detection for detection in result.detections if self._is_crop_eligible(frame, detection)
        )
        self._stats.record_detection_counts(
            total=len(result.detections),
            invalid=len(result.detections) - len(eligible),
        )
        return DetectorResult(detections=eligible)

    @staticmethod
    def _is_crop_eligible(frame: DecodedFrame, detection: Detection) -> bool:
        values = detection.xyxy
        if detection.class_id != 0:
            return False
        confidence = detection.confidence
        try:
            confidence_is_valid = isfinite(confidence) and 0.0 <= confidence <= 1.0
        except (TypeError, ValueError):
            return False
        if not confidence_is_valid:
            return False
        try:
            _ = bounded_crop_dimensions(
                frame_width=frame.width,
                frame_height=frame.height,
                bounding_box=values,
            )
        except (CropExtractionError, TypeError, ValueError):
            return False
        return True

    async def _request_detection(
        self,
        frame: DecodedFrame,
        *,
        generation: GenerationBinding,
        tracking: ByteTrackScope,
    ) -> DetectorResult | None:
        if self._closed or self._generation != generation or self._tracking is not tracking:
            self._stats.record_error("detector request crossed the active generation")
            return None
        async with self._detector_lock:
            if self._closed or self._generation != generation or self._tracking is not tracking:
                self._stats.record_error("detector request crossed the active generation")
                return None
            self._stats.record_detector_request()
            try:
                with anyio.fail_after(self._detector_deadline_seconds):
                    result = await self._detector.detect(
                        DetectorRequest(
                            encoded_image=frame.encoded_image,
                            confidence=self._threshold,
                        )
                    )
            except TimeoutError:
                self._stats.record_error("detector request deadline exceeded")
                return None
            except (RuntimeError, ValueError):
                self._stats.record_error("detector request failed")
                return None
            self._stats.record_detector_result()
            return result

    async def run_decoder(self, *, status_sink: DecoderStatusSink | None = None) -> None:
        """Drain the RTSP decoder until cancellation or explicit close."""
        await self._decoder.run_forever(status_sink=status_sink or self._on_source_status)

    async def set_threshold(self, threshold: float) -> tuple[PipelineHandoff, ...]:
        """Fence active tracks before applying a changed detector threshold."""
        if self._closed:
            return ()
        if not isfinite(threshold) or not _MIN_THRESHOLD <= threshold <= _MAX_THRESHOLD:
            raise WorkerConfigurationError(detail="threshold must be between 0.1 and 0.95")
        if threshold == self._threshold:
            return ()
        events = await self._reset_tracking(ResetReason.THRESHOLD_CHANGED)
        self._threshold = threshold
        self._tracking = self._new_tracking_scope(threshold)
        return events

    async def set_detection_enabled(self, enabled: bool) -> tuple[PipelineHandoff, ...]:
        """Fence active tracks before toggling detector work for this camera."""
        if self._closed or enabled == self._detection_enabled:
            return ()
        reason = ResetReason.DETECTION_ENABLED if enabled else ResetReason.DETECTION_DISABLED
        events = await self._reset_tracking(reason)
        self._detection_enabled = enabled
        self._tracking = self._new_tracking_scope(self._threshold)
        return events

    async def close(
        self,
        *,
        reason: ResetReason = ResetReason.SOURCE_GENERATION_CHANGED,
    ) -> tuple[PipelineHandoff, ...]:
        """Stop decode, clear bounded frame state, and emit terminal track events."""
        if self._closed:
            return ()
        events = await self._reset_tracking(reason)
        self._closed = True
        self._decoder.stop()
        self._slot.clear()
        self._frames_by_reference.clear()
        return events

    async def _reset_tracking(self, reason: ResetReason) -> tuple[PipelineHandoff, ...]:
        generation = self._generation
        tracking = self._tracking
        if tracking.active_track_count == 0 or self._last_dimensions is None:
            return ()
        now = self._monotonic_clock()
        boundary = DetectionFrame(
            camera_id=generation.camera_id,
            session_id=generation.camera_session_id,
            source_generation_id=generation.source_generation_id,
            ingress_utc=self._utc_clock(),
            ingress_monotonic=now,
            detector_result_monotonic=now,
            source_width=self._last_dimensions[0],
            source_height=self._last_dimensions[1],
            detector_input_reference=DetectorInputReference("worker-boundary"),
            detections=(),
        )
        lifecycles = tracking.reset(reason=reason, boundary=boundary)
        return await self._deliver(lifecycles, generation=generation)

    async def _deliver(
        self,
        lifecycles: tuple[TrackLifecycle, ...],
        *,
        generation: GenerationBinding,
    ) -> tuple[PipelineHandoff, ...]:
        handoffs: list[PipelineHandoff] = []
        for lifecycle in lifecycles:
            candidate = self._candidate_for(lifecycle)
            if lifecycle.kind is LifecycleKind.START and candidate is None:
                self._stats.record_error("eligible track has no bounded crop")
                continue
            handoff = PipelineHandoff(
                generation=generation,
                lifecycle=lifecycle,
                candidate=candidate,
            )
            handoffs.append(handoff)
            try:
                await self._handoff_consumer(handoff)
            except (RuntimeError, ValueError):
                self._stats.record_error("pipeline handoff failed")
        return tuple(handoffs)

    def _candidate_for(self, lifecycle: TrackLifecycle) -> CropCandidate | None:
        if lifecycle.kind is LifecycleKind.END:
            return None
        observation = (
            lifecycle.first_candidate
            if lifecycle.kind is LifecycleKind.START
            else lifecycle.observation
        )
        source_frame = self._frames_by_reference.get(observation.detector_input_reference)
        if source_frame is None:
            return None
        try:
            crop = extract_rgb_crop(frame=source_frame, observation=observation)
        except CropExtractionError:
            return None
        return CropCandidate(
            track_key=lifecycle.track_key,
            observation=observation,
            crop=crop,
            source_width=source_frame.width,
            source_height=source_frame.height,
        )

    def _record_decoded(self, frame: DecodedFrame) -> None:
        if self._closed:
            return
        self._ensure_generation(frame.source_generation_id)
        self._last_dimensions = (frame.width, frame.height)
        self._stats.record_decoded(frame.ingress_monotonic)

    async def _on_source_status(self, detail: str) -> None:
        self._stats.record_source_error(detail)

    async def _reconnect_generation(self, previous: GenerationBinding) -> GenerationBinding:
        owner = self._generation_reconnect
        if owner is None:
            return previous
        if previous != self._generation:
            raise StaleGenerationError(detail="reconnect crossed the active generation")
        _ = await self._reset_tracking(ResetReason.SOURCE_GENERATION_CHANGED)
        replacement = await owner(previous)
        if replacement.camera_id != previous.camera_id:
            raise StaleGenerationError(detail="reconnect changed the camera identity")
        if replacement.source_generation_id == previous.source_generation_id:
            raise StaleGenerationError(detail="reconnect reused the source generation")
        if replacement.camera_session_id == previous.camera_session_id:
            raise StaleGenerationError(detail="reconnect reused the camera session")
        if replacement.camera_version < previous.camera_version:
            raise StaleGenerationError(detail="reconnect moved the camera version backwards")
        self._generation = replacement
        self._tracking = self._new_tracking_scope(self._threshold)
        self._frames_by_reference.clear()
        self._last_dimensions = None
        self._stats.reset_freshness()
        return replacement

    def _remember_frame(self, frame: DecodedFrame) -> None:
        self._frames_by_reference[frame.reference] = frame
        while len(self._frames_by_reference) > _MAX_RETAINED_FRAMES:
            oldest = next(iter(self._frames_by_reference))
            del self._frames_by_reference[oldest]

    def _new_tracking_scope(self, threshold: float) -> ByteTrackScope:
        return ByteTrackScope(
            scope=TrackingScope(
                camera_id=self._generation.camera_id,
                session_id=self._generation.camera_session_id,
                source_generation_id=self._generation.source_generation_id,
                detection_enabled=self._detection_enabled,
            ),
            threshold=threshold,
        )

    def _ensure_generation(self, source_generation_id: SourceGenerationId) -> None:
        if source_generation_id != self._generation.source_generation_id:
            raise StaleGenerationError(detail="frame crossed the active source generation")


class IngestCoordinator:
    """Run all active camera decoders with one fair sampling scheduler."""

    def __init__(self, *, scheduler: FairRoundRobinScheduler | None = None) -> None:
        """Create an empty coordinator with a five-fps fair scheduler by default."""
        self._scheduler: FairRoundRobinScheduler = scheduler or FairRoundRobinScheduler()
        self._workers: dict[CameraId, IngestWorker] = {}
        self._failed: dict[CameraId, None] = {}
        self._decoder_group: TaskGroup | None = None

    @property
    def workers(self) -> tuple[IngestWorker, ...]:
        """Return a stable snapshot of active camera workers."""
        return tuple(self._workers.values())

    @property
    def failed_cameras(self) -> tuple[CameraId, ...]:
        """Return registered cameras whose decoder stopped with an error, in failure order."""
        return tuple(self._failed)

    @property
    def peak_in_flight_count(self) -> int:
        """Return the observed global detector-request high-water mark."""
        return self._scheduler.peak_in_flight_count

    def add(self, worker: IngestWorker) -> None:
        """Register one worker, starting its decoder if the run loop is already active."""
        if worker.camera_id in self._workers:
            raise WorkerConfigurationError(detail="camera worker is already registered")
        self._workers[worker.camera_id] = worker
        _ = self._failed.pop(worker.camera_id, None)
        self._scheduler.register(worker.camera_id)
        if self._decoder_group is not None:
            self._decoder_group.start_soon(self._run_decoder, worker)

    async def remove(self, camera_id: CameraId) -> tuple[PipelineHandoff, ...]:
        """Stop and unregister one camera worker, if present."""
        _ = self._failed.pop(camera_id, None)
        worker = self._workers.pop(camera_id, None)
        if worker is None:
            return ()
        self._scheduler.unregister(camera_id)
        return await worker.close(reason=ResetReason.SOURCE_GENERATION_CHANGED)

    async def _run_decoder(self, worker: IngestWorker) -> None:
        # One camera's decode or reconnect failure must not cancel every other camera's
        # decoder, so the failure is recorded for status and replacement instead.
        try:
            await worker.run_decoder()
        except Exception:  # noqa: BLE001 - per-camera isolation boundary
            if self._workers.get(worker.camera_id) is worker:
                self._failed[worker.camera_id] = None

    async def run(self, *, stop_event: anyio.Event) -> None:
        """Drain decoders and dispatch one bounded detector sample per fair turn."""
        try:
            async with anyio.create_task_group() as task_group:
                for worker in self._workers.values():
                    task_group.start_soon(self._run_decoder, worker)
                self._decoder_group = task_group
                try:
                    await self._scheduler.run(self._dispatch, stop_event=stop_event)
                finally:
                    self._decoder_group = None
                    task_group.cancel_scope.cancel()
        finally:
            for worker in tuple(self._workers.values()):
                _ = await worker.close(reason=ResetReason.CLOSED)

    async def _dispatch(self, camera_id: CameraId) -> None:
        worker = self._workers.get(camera_id)
        if worker is not None:
            _ = await worker.sample_once()


__all__ = [
    "DetectorPort",
    "IngestCoordinator",
    "IngestWorker",
    "PipelineHandoffConsumer",
    "StaleGenerationError",
    "WorkerConfigurationError",
]
