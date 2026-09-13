"""Pure representative-candidate policy shared by queue and publisher."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from math import ceil, floor, isfinite, sqrt
from typing import Final, override
from uuid import UUID, uuid5

from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId

_APPEARANCE_NAMESPACE: Final = UUID("0b4f68f0-14d8-4c7c-9b48-1b4c5ad8e9a1")
_MIN_CROP_WIDTH: Final = 32
_MIN_CROP_HEIGHT: Final = 64
_UPGRADE_RATIO: Final = 1.10
_UPGRADE_COOLDOWN_SECONDS: Final = 2.0
_FLOAT_EPSILON: Final = 1e-9


class PolicyErrorCode(StrEnum):
    """Name malformed values rejected by representative policy."""

    CANDIDATE_SCORE_INVALID = "candidate_score_invalid"
    FIRST_SEEN_TIMEZONE = "first_seen_timezone"
    DETECT_CLOCK_INVALID = "detect_clock_invalid"
    SOURCE_DIMENSIONS = "source_dimensions"
    GEOMETRY_NONFINITE = "geometry_nonfinite"
    GEOMETRY_OUT_OF_BOUNDS = "geometry_out_of_bounds"
    GEOMETRY_DEGENERATE = "geometry_degenerate"
    CROP_WIDTH = "crop_width"
    CROP_HEIGHT = "crop_height"
    CONFIDENCE_INVALID = "confidence_invalid"
    SHARPNESS_INVALID = "sharpness_invalid"
    ELAPSED_INVALID = "elapsed_invalid"
    TRACK_ID_INVALID = "track_id_invalid"


@dataclass(frozen=True, slots=True)
class CropRejectedError(ValueError):
    """Describe a person crop that cannot become searchable media."""

    code: PolicyErrorCode

    @override
    def __str__(self) -> str:
        return self.code.value


CropRejected = CropRejectedError


@dataclass(frozen=True, slots=True)
class CandidateRank:
    """Order one candidate by frame completeness, then quality score."""

    fully_inside: bool
    score: float

    def __post_init__(self) -> None:
        """Reject non-finite quality values before ordering candidates."""
        if not isfinite(self.score) or self.score < 0.0:
            raise CropRejectedError(PolicyErrorCode.CANDIDATE_SCORE_INVALID)

    def __lt__(self, other: object) -> bool:
        """Compare completeness before the scalar quality score."""
        if not isinstance(other, CandidateRank):
            return NotImplemented
        return (self.fully_inside, self.score) < (other.fully_inside, other.score)


@dataclass(frozen=True, slots=True)
class LifecycleClocks:
    """Retain first ingress and detector-receipt clocks across retries."""

    first_seen: datetime
    t_detect_monotonic: float

    def __post_init__(self) -> None:
        """Validate the clocks at the appearance handoff."""
        if self.first_seen.tzinfo is None or self.first_seen.utcoffset() is None:
            raise CropRejectedError(PolicyErrorCode.FIRST_SEEN_TIMEZONE)
        if not isfinite(self.t_detect_monotonic) or self.t_detect_monotonic < 0.0:
            raise CropRejectedError(PolicyErrorCode.DETECT_CLOCK_INVALID)


def validate_crop_geometry(
    *,
    bounding_box: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
) -> tuple[int, int, int, int]:
    """Return integer crop geometry after enforcing source bounds and minimum size."""
    if source_width <= 0 or source_height <= 0:
        raise CropRejectedError(PolicyErrorCode.SOURCE_DIMENSIONS)
    x_min, y_min, x_max, y_max = bounding_box
    if not all(isfinite(value) for value in bounding_box):
        raise CropRejectedError(PolicyErrorCode.GEOMETRY_NONFINITE)
    if x_min < 0.0 or y_min < 0.0 or x_max > source_width or y_max > source_height:
        raise CropRejectedError(PolicyErrorCode.GEOMETRY_OUT_OF_BOUNDS)
    integer_box = (floor(x_min), floor(y_min), ceil(x_max), ceil(y_max))
    integer_x_min, integer_y_min, integer_x_max, integer_y_max = integer_box
    if integer_x_max <= integer_x_min or integer_y_max <= integer_y_min:
        raise CropRejectedError(PolicyErrorCode.GEOMETRY_DEGENERATE)
    if integer_x_max - integer_x_min < _MIN_CROP_WIDTH:
        raise CropRejectedError(PolicyErrorCode.CROP_WIDTH)
    if integer_y_max - integer_y_min < _MIN_CROP_HEIGHT:
        raise CropRejectedError(PolicyErrorCode.CROP_HEIGHT)
    return integer_box


def rank_candidate(
    *,
    bounding_box: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    confidence: float,
    laplacian_variance: float,
) -> CandidateRank:
    """Calculate the prescribed border-first representative rank."""
    if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise CropRejectedError(PolicyErrorCode.CONFIDENCE_INVALID)
    if not isfinite(laplacian_variance) or laplacian_variance < 0.0:
        raise CropRejectedError(PolicyErrorCode.SHARPNESS_INVALID)
    x_min, y_min, x_max, y_max = bounding_box
    fully_inside = x_min > 0.0 and y_min > 0.0 and x_max < source_width and y_max < source_height
    width = max(0.0, x_max - x_min)
    height = max(0.0, y_max - y_min)
    area_term = sqrt(width * height)
    sharpness_term = min(laplacian_variance / 100.0, 1.0)
    return CandidateRank(
        fully_inside=fully_inside,
        score=area_term * confidence * sharpness_term,
    )


def should_upgrade(
    current: CandidateRank,
    candidate: CandidateRank,
    elapsed_since_upgrade: float,
) -> bool:
    """Return whether a candidate passes border, gain, and cooldown policy."""
    if not isfinite(elapsed_since_upgrade) or elapsed_since_upgrade < 0.0:
        raise CropRejectedError(PolicyErrorCode.ELAPSED_INVALID)
    if elapsed_since_upgrade < _UPGRADE_COOLDOWN_SECONDS:
        return False
    if candidate.fully_inside and not current.fully_inside:
        return True
    if candidate.fully_inside != current.fully_inside:
        return False
    return candidate.score + _FLOAT_EPSILON >= current.score * _UPGRADE_RATIO


def appearance_id_for_track(
    camera_id: CameraId,
    session_id: CameraSessionId,
    track_id: int,
) -> AppearanceId:
    """Map one camera/session/local-track tuple to a stable UUID namespace."""
    if track_id < 0:
        raise CropRejectedError(PolicyErrorCode.TRACK_ID_INVALID)
    name = f"{camera_id}:{session_id}:{track_id}"
    return AppearanceId(uuid5(_APPEARANCE_NAMESPACE, name))
