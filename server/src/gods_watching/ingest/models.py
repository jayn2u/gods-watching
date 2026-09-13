"""Typed bounded state shared by the RTSP decoder and detector scheduler."""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from math import isfinite
from threading import Lock
from typing import Final, override

from gods_watching.media.models import SourceGenerationId
from gods_watching.tracking.models import DetectorInputReference

_RGB_CHANNELS: Final = 3
_FPS_WINDOW_SECONDS: Final = 10.0
_MIN_RATE_SAMPLES: Final = 2


@dataclass(frozen=True, slots=True)
class DecoderInputError(ValueError):
    """Describe malformed decoded-frame data at the ingest boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """Carry one decoded source frame and its server-assigned ingress clocks."""

    source_generation_id: SourceGenerationId
    ingress_utc: datetime
    ingress_monotonic: float
    reference: DetectorInputReference
    width: int
    height: int
    rgb_bytes: bytes
    encoded_image: bytes

    def __post_init__(self) -> None:
        """Reject a frame that cannot be safely sampled or cropped."""
        if self.ingress_utc.tzinfo is None or self.ingress_utc.utcoffset() is None:
            raise DecoderInputError(detail="ingress_utc must include a timezone")
        if not isfinite(self.ingress_monotonic) or self.ingress_monotonic < 0:
            raise DecoderInputError(detail="ingress_monotonic must be finite and non-negative")
        if self.width <= 0 or self.height <= 0:
            raise DecoderInputError(detail="decoded frame dimensions must be positive")
        expected_size = self.width * self.height * _RGB_CHANNELS
        if len(self.rgb_bytes) != expected_size:
            raise DecoderInputError(detail="decoded RGB bytes do not match dimensions")
        if not self.encoded_image:
            raise DecoderInputError(detail="encoded detector image cannot be empty")
        if not str(self.reference):
            raise DecoderInputError(detail="decoded frame reference cannot be empty")
        object.__setattr__(self, "ingress_utc", self.ingress_utc.astimezone(UTC))


@dataclass(slots=True)
class LatestFrameSlot:
    """Own one mutable latest-frame slot; replacing a frame is the backpressure policy."""

    _lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _frame: DecodedFrame | None = field(default=None, init=False, repr=False)
    _dropped_frames: int = field(default=0, init=False, repr=False)
    _peak_occupancy: int = field(default=0, init=False, repr=False)

    def put(self, frame: DecodedFrame) -> None:
        """Replace the current frame and count an overwrite when one was pending."""
        with self._lock:
            if self._frame is not None:
                self._dropped_frames += 1
            self._frame = frame
            self._peak_occupancy = 1

    def take_latest(self) -> DecodedFrame | None:
        """Take and clear the newest frame without retaining a decoder backlog."""
        with self._lock:
            frame = self._frame
            self._frame = None
            return frame

    def clear(self) -> None:
        """Release the one pending frame during cancellation or generation reset."""
        with self._lock:
            self._frame = None

    @property
    def dropped_frames(self) -> int:
        """Return the number of decoded frames overwritten before sampling."""
        with self._lock:
            return self._dropped_frames

    @property
    def occupancy(self) -> int:
        """Return the current number of retained frames."""
        with self._lock:
            return int(self._frame is not None)

    @property
    def peak_occupancy(self) -> int:
        """Return the largest observed number of retained frames."""
        with self._lock:
            return self._peak_occupancy


@dataclass(frozen=True, slots=True)
class IngestStatsSnapshot:
    """Expose truthful bounded ingest counters and freshness values."""

    frames_decoded: int
    frames_sampled: int
    detector_requests: int
    detector_results: int
    detector_detections: int
    invalid_detections: int
    eligible_detections: int
    dropped_frames: int
    pending_detector_requests: int
    actual_framerate: float
    frame_age_seconds: float | None
    sanitized_errors: int
    last_sanitized_error: str | None
    identity_switch_observations: int
    stale_generation_results: int = 0
    scheduler_dispatches: int = 0
    dispatch_detector_requested: int = 0
    dispatch_no_frame: int = 0
    dispatch_detection_disabled: int = 0
    dispatch_generation_fenced: int = 0
    dispatch_closed: int = 0
    dispatch_outcomes: int = 0
    peak_detector_requests: int = 0


@dataclass(slots=True)
class IngestStats:
    """Accumulate bounded runtime observations for one camera worker."""

    frames_decoded: int = 0
    frames_sampled: int = 0
    detector_requests: int = 0
    detector_results: int = 0
    detector_detections: int = 0
    invalid_detections: int = 0
    eligible_detections: int = 0
    pending_detector_requests: int = 0
    sanitized_errors: int = 0
    last_sanitized_error: str | None = None
    identity_switch_observations: int = 0
    stale_generation_results: int = 0
    scheduler_dispatches: int = 0
    dispatch_detector_requested: int = 0
    dispatch_no_frame: int = 0
    dispatch_detection_disabled: int = 0
    dispatch_generation_fenced: int = 0
    dispatch_closed: int = 0
    peak_detector_requests: int = 0
    _ingress_times: deque[float] = field(default_factory=deque, init=False, repr=False)
    _last_ingress_monotonic: float | None = field(default=None, init=False, repr=False)

    def record_decoded(self, ingress_monotonic: float) -> None:
        """Record one successfully decoded frame in the bounded rate window."""
        self.frames_decoded += 1
        self._last_ingress_monotonic = ingress_monotonic
        self._ingress_times.append(ingress_monotonic)
        cutoff = ingress_monotonic - _FPS_WINDOW_SECONDS
        while self._ingress_times and self._ingress_times[0] < cutoff:
            _ = self._ingress_times.popleft()

    def record_sampled(self) -> None:
        """Record one frame selected for detector work."""
        self.frames_sampled += 1

    def record_detector_request(self) -> None:
        """Record one detector request entering the camera's single-flight slot."""
        self.detector_requests += 1
        self.pending_detector_requests = 1
        self.peak_detector_requests = max(
            self.peak_detector_requests, self.pending_detector_requests
        )

    def record_dispatch(self) -> None:
        """Record one scheduler invocation awaiting an outcome."""
        self.scheduler_dispatches += 1

    def record_dispatch_outcome(self, outcome: str) -> None:
        """Record exactly one bounded outcome for a scheduler invocation."""
        match outcome:
            case "detector_requested":
                self.dispatch_detector_requested += 1
            case "no_frame":
                self.dispatch_no_frame += 1
            case "detection_disabled":
                self.dispatch_detection_disabled += 1
            case "generation_fenced":
                self.dispatch_generation_fenced += 1
            case "closed":
                self.dispatch_closed += 1
            case unreachable:
                raise DecoderInputError(detail=f"unknown dispatch outcome: {unreachable}")

    def record_detector_result(self) -> None:
        """Record one detector response and release the pending slot."""
        self.detector_results += 1
        self.pending_detector_requests = 0

    def record_detection_counts(self, *, total: int, invalid: int) -> None:
        """Record detector boxes retained and rejected by crop eligibility."""
        self.detector_detections += total
        self.invalid_detections += invalid
        self.eligible_detections += total - invalid

    def record_error(self, detail: str) -> None:
        """Record one sanitized operational error without retaining source credentials."""
        self.sanitized_errors += 1
        if detail == "detector result crossed the active generation":
            self.stale_generation_results += 1
        self.last_sanitized_error = detail[:200]
        self.pending_detector_requests = 0

    def record_source_error(self, detail: str) -> None:
        """Record a source status while preserving any detector request in flight."""
        self.sanitized_errors += 1
        self.last_sanitized_error = detail[:200]

    def record_identity_switch_observation(self) -> None:
        """Count a tracker identity observation reported by a real run."""
        self.identity_switch_observations += 1

    def reset_freshness(self) -> None:
        """Clear source-specific freshness clocks at a new generation boundary."""
        self._ingress_times.clear()
        self._last_ingress_monotonic = None
        self.pending_detector_requests = 0

    def snapshot(self, *, now_monotonic: float, dropped_frames: int) -> IngestStatsSnapshot:
        """Return counters with server-ingress frame age and measured decode rate."""
        rate = 0.0
        if len(self._ingress_times) >= _MIN_RATE_SAMPLES:
            elapsed = self._ingress_times[-1] - self._ingress_times[0]
            if elapsed > 0.0:
                rate = (len(self._ingress_times) - 1) / elapsed
        age = (
            None
            if self._last_ingress_monotonic is None
            else max(0.0, now_monotonic - self._last_ingress_monotonic)
        )
        return IngestStatsSnapshot(
            frames_decoded=self.frames_decoded,
            frames_sampled=self.frames_sampled,
            detector_requests=self.detector_requests,
            detector_results=self.detector_results,
            detector_detections=self.detector_detections,
            invalid_detections=self.invalid_detections,
            eligible_detections=self.eligible_detections,
            dropped_frames=dropped_frames,
            pending_detector_requests=self.pending_detector_requests,
            actual_framerate=rate,
            frame_age_seconds=age,
            sanitized_errors=self.sanitized_errors,
            last_sanitized_error=self.last_sanitized_error,
            identity_switch_observations=self.identity_switch_observations,
            stale_generation_results=self.stale_generation_results,
            scheduler_dispatches=self.scheduler_dispatches,
            dispatch_detector_requested=self.dispatch_detector_requested,
            dispatch_no_frame=self.dispatch_no_frame,
            dispatch_detection_disabled=self.dispatch_detection_disabled,
            dispatch_generation_fenced=self.dispatch_generation_fenced,
            dispatch_closed=self.dispatch_closed,
            dispatch_outcomes=(
                self.dispatch_detector_requested
                + self.dispatch_no_frame
                + self.dispatch_detection_disabled
                + self.dispatch_generation_fenced
                + self.dispatch_closed
            ),
            peak_detector_requests=self.peak_detector_requests,
        )
