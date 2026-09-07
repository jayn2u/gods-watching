"""Extract bounded source-resolution RGB crops from decoded frames."""

from dataclasses import dataclass
from math import ceil, floor, isfinite
from typing import override

from gods_watching.contracts.pipeline import RgbCrop
from gods_watching.tracking.models import TrackObservation

from .models import DecodedFrame

_MIN_CROP_WIDTH = 32
_MIN_CROP_HEIGHT = 64


@dataclass(frozen=True, slots=True)
class CropExtractionError(ValueError):
    """Describe a crop that cannot be copied from the source frame safely."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


def bounded_crop_dimensions(
    *,
    frame_width: int,
    frame_height: int,
    bounding_box: tuple[float, float, float, float],
) -> tuple[int, int]:
    """Return integer crop dimensions when a detector box is safely crop eligible."""
    x_min, y_min, x_max, y_max = bounding_box
    if not all(isfinite(value) for value in bounding_box):
        raise CropExtractionError(detail="crop geometry must be finite")
    if x_min < 0.0 or y_min < 0.0 or x_max > frame_width or y_max > frame_height:
        raise CropExtractionError(detail="crop geometry is outside the decoded frame")
    width = ceil(x_max) - floor(x_min)
    height = ceil(y_max) - floor(y_min)
    if width < _MIN_CROP_WIDTH:
        raise CropExtractionError(detail="crop width is below 32 pixels")
    if height < _MIN_CROP_HEIGHT:
        raise CropExtractionError(detail="crop height is below 64 pixels")
    return width, height


def extract_rgb_crop(*, frame: DecodedFrame, observation: TrackObservation) -> RgbCrop:
    """Copy exactly one valid source-resolution RGB crop into bounded memory."""
    x_min, y_min, x_max, y_max = observation.bounding_box
    width, height = bounded_crop_dimensions(
        frame_width=frame.width,
        frame_height=frame.height,
        bounding_box=observation.bounding_box,
    )
    integer_x_min = floor(x_min)
    integer_y_min = floor(y_min)
    integer_x_max = ceil(x_max)
    integer_y_max = ceil(y_max)

    row_stride = frame.width * 3
    left = integer_x_min * 3
    right = integer_x_max * 3
    rows = (
        frame.rgb_bytes[row * row_stride + left : row * row_stride + right]
        for row in range(integer_y_min, integer_y_max)
    )
    data = b"".join(rows)
    return RgbCrop(data=data, width=width, height=height)


def crop_for_observation(frame: DecodedFrame, observation: TrackObservation) -> RgbCrop:
    """Provide a positional convenience wrapper for crop extraction callers."""
    return extract_rgb_crop(frame=frame, observation=observation)


__all__ = [
    "CropExtractionError",
    "bounded_crop_dimensions",
    "crop_for_observation",
    "extract_rgb_crop",
]
