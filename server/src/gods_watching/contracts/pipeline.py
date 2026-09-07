"""Bounded in-memory handoff between ingest, tracking, and appearance workers."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, override

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.media.models import GenerationEventKind, SourceGenerationId
from gods_watching.tracking import TrackKey, TrackLifecycle, TrackObservation

if TYPE_CHECKING:
    from gods_watching.cameras.lifecycle import CameraGenerationId

_RGB_CHANNELS: Final = 3
_MIN_CROP_WIDTH: Final = 32
_MIN_CROP_HEIGHT: Final = 64


@dataclass(frozen=True, slots=True)
class PipelineContractError(ValueError):
    """Describe malformed data crossing the bounded pipeline handoff."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class GenerationBinding:
    """Bind one DB camera session to exactly one MediaMTX source generation."""

    camera_id: CameraId
    camera_session_id: CameraSessionId
    db_generation_id: "CameraGenerationId"
    source_generation_id: SourceGenerationId
    camera_version: int

    def __post_init__(self) -> None:
        """Reject a generation binding that cannot fence late work."""
        if self.camera_version < 1:
            raise PipelineContractError(detail="camera_version must be positive")


@dataclass(frozen=True, slots=True)
class PipelineGenerationEvent:
    """Carry one authoritative generation lifecycle event to worker consumers."""

    binding: GenerationBinding
    kind: GenerationEventKind
    occurred_utc: datetime

    def __post_init__(self) -> None:
        """Normalize event time to UTC at the contract boundary."""
        if self.occurred_utc.tzinfo is None or self.occurred_utc.utcoffset() is None:
            raise PipelineContractError(detail="occurred_utc must include a timezone")
        object.__setattr__(self, "occurred_utc", self.occurred_utc.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class RgbCrop:
    """Hold one bounded source-resolution RGB crop in memory."""

    data: bytes
    width: int
    height: int

    def __post_init__(self) -> None:
        """Reject undersized or non-contiguous crop payloads."""
        if self.width < _MIN_CROP_WIDTH or self.height < _MIN_CROP_HEIGHT:
            raise PipelineContractError(detail="RGB crop is smaller than 32x64 pixels")
        expected_size = self.width * self.height * _RGB_CHANNELS
        if len(self.data) != expected_size:
            raise PipelineContractError(detail="RGB crop byte length does not match dimensions")


@dataclass(frozen=True, slots=True)
class CropCandidate:
    """Pair one tracked observation with its original-frame RGB crop."""

    track_key: TrackKey
    observation: TrackObservation
    crop: RgbCrop
    source_width: int
    source_height: int

    def __post_init__(self) -> None:
        """Reject a crop whose source geometry cannot be persisted safely."""
        if self.source_width <= 0 or self.source_height <= 0:
            raise PipelineContractError(detail="source dimensions must be positive")
        x_min, y_min, x_max, y_max = self.observation.bounding_box
        if not (
            0.0 <= x_min < x_max <= self.source_width and 0.0 <= y_min < y_max <= self.source_height
        ):
            raise PipelineContractError(detail="crop observation is outside source bounds")


@dataclass(frozen=True, slots=True)
class PipelineHandoff:
    """Deliver one lifecycle event and at most one bounded crop to publication."""

    generation: GenerationBinding
    lifecycle: TrackLifecycle
    candidate: CropCandidate | None

    def __post_init__(self) -> None:
        """Enforce generation identity and candidate rules for each lifecycle variant."""
        key = self.lifecycle.track_key
        if (
            key.camera_id != self.generation.camera_id
            or key.session_id != self.generation.camera_session_id
            or key.source_generation_id != self.generation.source_generation_id
        ):
            raise PipelineContractError(detail="lifecycle crossed the generation binding")
        match str(self.lifecycle.kind.value):
            case "start":
                if self.candidate is None:
                    raise PipelineContractError(detail="start lifecycle requires a crop candidate")
                if self.candidate.observation != self.lifecycle.first_candidate:
                    raise PipelineContractError(
                        detail="start candidate must preserve the first eligible observation"
                    )
            case "update":
                pass
            case "end":
                if self.candidate is not None:
                    raise PipelineContractError(
                        detail="end lifecycle cannot carry a crop candidate"
                    )
            case unreachable:
                raise PipelineContractError(detail=f"unsupported lifecycle kind: {unreachable}")
        if self.candidate is not None and self.candidate.track_key != key:
            raise PipelineContractError(detail="crop candidate crossed the lifecycle track key")


__all__ = [
    "CropCandidate",
    "GenerationBinding",
    "PipelineContractError",
    "PipelineGenerationEvent",
    "PipelineHandoff",
    "RgbCrop",
]
