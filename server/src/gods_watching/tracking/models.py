"""Typed inputs and lifecycle values for camera-local tracking."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import Final, NewType, override

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.inference.detector import Detection
from gods_watching.media.models import SourceGenerationId

DetectorInputReference = NewType("DetectorInputReference", str)
LocalTrackId = NewType("LocalTrackId", int)

_MIN_THRESHOLD: Final = 0.1
_MAX_THRESHOLD: Final = 0.95
_TRACK_EXPIRY_SECONDS: Final = 2.0
_BOX_COORDINATE_COUNT: Final = 4


@dataclass(frozen=True, slots=True)
class TrackingConfigurationError(ValueError):
    """Describe a camera tracking setting outside the trusted contract."""

    threshold: float

    @override
    def __str__(self) -> str:
        return f"threshold must be finite and between 0.1 and 0.95: {self.threshold}"


@dataclass(frozen=True, slots=True)
class StaleFrameError(ValueError):
    """Describe a frame rejected by the active source or monotonic fence."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ClosedScopeError(RuntimeError):
    """Describe an attempt to feed a closed tracking scope."""

    detail: str = "tracking scope is closed"

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class TrackingInputError(ValueError):
    """Describe malformed detector or lifecycle input at the tracking boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class LifecycleKind(StrEnum):
    """Name the state transition represented by a lifecycle event."""

    START = "start"
    UPDATE = "update"
    END = "end"


class ResetReason(StrEnum):
    """Name a boundary that invalidates the current tracker state."""

    SOURCE_GENERATION_CHANGED = "source_generation_changed"
    DETECTION_DISABLED = "detection_disabled"
    DETECTION_ENABLED = "detection_enabled"
    THRESHOLD_CHANGED = "threshold_changed"
    EXPLICIT = "explicit"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TrackingParameters:
    """Pin the approved supervision ByteTrack parameters for one camera."""

    frame_rate: int
    lost_track_buffer: int
    minimum_matching_threshold: float
    minimum_consecutive_frames: int
    track_activation_threshold: float
    camera_threshold: float

    @classmethod
    def for_camera_threshold(cls, threshold: float) -> "TrackingParameters":
        """Build fixed tracker parameters from one validated camera threshold."""
        if not isfinite(threshold) or not _MIN_THRESHOLD <= threshold <= _MAX_THRESHOLD:
            raise TrackingConfigurationError(threshold=threshold)
        return cls(
            frame_rate=5,
            lost_track_buffer=60,
            minimum_matching_threshold=0.8,
            minimum_consecutive_frames=1,
            track_activation_threshold=max(0.0, threshold - 0.1),
            camera_threshold=threshold,
        )

    @property
    def library_max_time_lost(self) -> int:
        """Return supervision's sampled-frame loss budget at five fps."""
        return int(self.frame_rate / 30.0 * self.lost_track_buffer)


@dataclass(frozen=True, slots=True)
class TrackingScope:
    """Identify one camera, source session, generation, and detection mode."""

    camera_id: CameraId
    session_id: CameraSessionId
    source_generation_id: SourceGenerationId
    detection_enabled: bool


@dataclass(frozen=True, slots=True)
class DetectionFrame:
    """Carry detector output with server-assigned ingress clocks and frame reference."""

    camera_id: CameraId
    session_id: CameraSessionId
    source_generation_id: SourceGenerationId
    ingress_utc: datetime
    ingress_monotonic: float
    detector_result_monotonic: float
    source_width: int
    source_height: int
    detector_input_reference: DetectorInputReference
    detections: tuple[Detection, ...]
    detection_enabled: bool = True

    def __post_init__(self) -> None:
        """Normalize UTC and reject malformed ingress or source dimensions."""
        if self.ingress_utc.tzinfo is None or self.ingress_utc.utcoffset() is None:
            raise TrackingInputError(detail="ingress_utc must include a timezone")
        if not isfinite(self.ingress_monotonic) or self.ingress_monotonic < 0:
            raise TrackingInputError(detail="ingress_monotonic must be finite and non-negative")
        if (
            not isfinite(self.detector_result_monotonic)
            or self.detector_result_monotonic < self.ingress_monotonic
        ):
            raise TrackingInputError(
                detail="detector_result_monotonic must follow ingress_monotonic"
            )
        if self.source_width <= 0 or self.source_height <= 0:
            raise TrackingInputError(detail="source dimensions must be positive")
        object.__setattr__(self, "ingress_utc", self.ingress_utc.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class TrackKey:
    """Identify a local tracker ID within its camera/session/generation namespace."""

    camera_id: CameraId
    session_id: CameraSessionId
    source_generation_id: SourceGenerationId
    local_track_id: LocalTrackId


@dataclass(frozen=True, slots=True)
class TrackObservation:
    """Carry the original detector input and ingress clocks for one matched box."""

    bounding_box: tuple[float, float, float, float]
    confidence: float
    detector_input_reference: DetectorInputReference
    ingress_utc: datetime
    ingress_monotonic: float
    detector_result_monotonic: float

    def __post_init__(self) -> None:
        """Reject non-finite geometry, confidence, or clocks at the handoff boundary."""
        if len(self.bounding_box) != _BOX_COORDINATE_COUNT or not all(
            isfinite(value) for value in self.bounding_box
        ):
            raise TrackingInputError(detail="tracking bounding box must contain four finite values")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise TrackingInputError(detail="tracking confidence must be between 0 and 1")
        if self.ingress_utc.tzinfo is None or self.ingress_utc.utcoffset() is None:
            raise TrackingInputError(detail="observation ingress_utc must include a timezone")
        if not isfinite(self.ingress_monotonic) or self.ingress_monotonic < 0:
            raise TrackingInputError(
                detail="observation ingress_monotonic must be finite and non-negative"
            )
        if (
            not isfinite(self.detector_result_monotonic)
            or self.detector_result_monotonic < self.ingress_monotonic
        ):
            raise TrackingInputError(
                detail="observation detector_result_monotonic must follow ingress_monotonic"
            )
        object.__setattr__(self, "ingress_utc", self.ingress_utc.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class TrackLifecycle:
    """Describe a start, update, or end while retaining the first candidate handoff."""

    kind: LifecycleKind
    track_key: TrackKey
    source_generation_id: SourceGenerationId
    scope_epoch: int
    sequence: int
    observation: TrackObservation
    first_candidate: TrackObservation
    first_seen: datetime
    last_seen: datetime
    t_detect_monotonic: float
    ended_at: datetime | None = None
    ended_monotonic: float | None = None
    end_reason: ResetReason | None = None

    def __post_init__(self) -> None:
        """Normalize lifecycle timestamps and enforce monotonic observation ordering."""
        if self.scope_epoch < 0 or self.sequence <= 0:
            raise TrackingInputError(detail="lifecycle fence values are outside their contract")
        if self.first_seen.tzinfo is None or self.first_seen.utcoffset() is None:
            raise TrackingInputError(detail="first_seen must include a timezone")
        if self.last_seen.tzinfo is None or self.last_seen.utcoffset() is None:
            raise TrackingInputError(detail="last_seen must include a timezone")
        if self.ended_at is not None and (
            self.ended_at.tzinfo is None or self.ended_at.utcoffset() is None
        ):
            raise TrackingInputError(detail="ended_at must include a timezone")
        if not isfinite(self.t_detect_monotonic) or self.t_detect_monotonic < 0:
            raise TrackingInputError(detail="t_detect_monotonic must be finite and non-negative")
        if self.ended_monotonic is not None and (
            not isfinite(self.ended_monotonic) or self.ended_monotonic < 0
        ):
            raise TrackingInputError(detail="ended_monotonic must be finite and non-negative")
        object.__setattr__(self, "first_seen", self.first_seen.astimezone(UTC))
        object.__setattr__(self, "last_seen", self.last_seen.astimezone(UTC))
        if self.ended_at is not None:
            object.__setattr__(self, "ended_at", self.ended_at.astimezone(UTC))


TRACK_EXPIRY_SECONDS: Final = _TRACK_EXPIRY_SECONDS
