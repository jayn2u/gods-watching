"""Live detector overlay response contracts."""

from typing import Annotated

from pydantic import Field

from .base import ContractModel
from .identifiers import CameraId, CameraSessionId
from .primitives import UtcDatetime


class DetectionBox(ContractModel):
    """Represent one source-resolution person bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class LiveDetectionResponse(ContractModel):
    """Return the newest bounded detector result for one camera."""

    camera_id: CameraId
    camera_session_id: CameraSessionId | None
    frame_at: UtcDatetime | None
    frame_age_seconds: float | None
    width: Annotated[int, Field(gt=0)] | None
    height: Annotated[int, Field(gt=0)] | None
    boxes: tuple[DetectionBox, ...]


__all__ = ["DetectionBox", "LiveDetectionResponse"]
