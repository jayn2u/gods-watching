"""Camera/session-scoped ByteTrack lifecycle contracts."""

from .adapter import ByteTrackAdapter, TrackedDetection, observations_for_frame
from .models import (
    ClosedScopeError,
    DetectionFrame,
    DetectorInputReference,
    LifecycleKind,
    LocalTrackId,
    ResetReason,
    StaleFrameError,
    TrackingConfigurationError,
    TrackingInputError,
    TrackingParameters,
    TrackingScope,
    TrackKey,
    TrackLifecycle,
    TrackObservation,
)
from .scope import ByteTrackScope

__all__ = [
    "ByteTrackAdapter",
    "ByteTrackScope",
    "ClosedScopeError",
    "DetectionFrame",
    "DetectorInputReference",
    "LifecycleKind",
    "LocalTrackId",
    "ResetReason",
    "StaleFrameError",
    "TrackKey",
    "TrackLifecycle",
    "TrackObservation",
    "TrackedDetection",
    "TrackingConfigurationError",
    "TrackingInputError",
    "TrackingParameters",
    "TrackingScope",
    "observations_for_frame",
]
