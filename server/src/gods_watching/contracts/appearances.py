"""Appearance persistence and API response contracts."""

from typing import Annotated

from pydantic import Field, field_validator

from .base import ContractModel
from .identifiers import AppearanceId, CameraId, CameraSessionId
from .primitives import UtcDatetime

PositiveInt = Annotated[int, Field(gt=0)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]
# The initial catalog contains 512- and 768-dimensional CLIP packages.  The
# model identity carried alongside every publication keeps equal-width spaces
# distinct at the retrieval boundary.
Embedding = Annotated[tuple[float, ...], Field(min_length=1, max_length=4096)]


class BoundingBox(ContractModel):
    """Represent an original-pixel person bounding box."""

    x_min: Annotated[int, Field(ge=0)]
    y_min: Annotated[int, Field(ge=0)]
    x_max: PositiveInt
    y_max: PositiveInt


class AppearancePublication(ContractModel):
    """Publish one immutable representative revision at the storage boundary."""

    appearance_id: AppearanceId
    camera_id: CameraId
    session_id: CameraSessionId
    track_id: Annotated[int, Field(ge=0)]
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    ended_at: UtcDatetime | None
    representative_version: PositiveInt
    crop_object_key: str
    bounding_box: BoundingBox
    source_width: PositiveInt
    source_height: PositiveInt
    detector_confidence: UnitScore
    crop_quality: Annotated[float, Field(ge=0.0)]
    byte_size: PositiveInt
    embedded_at: UtcDatetime
    model_id: str
    model_revision: str
    embedding: Embedding

    @field_validator("embedding")
    @classmethod
    def require_supported_dimension(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        """Keep publication vectors within the deployed CLIP dimensions."""
        if len(value) not in {512, 768}:
            error_message = "embedding dimension must be 512 or 768"
            raise ValueError(error_message)
        return value


class AppearanceResponse(ContractModel):
    """Expose searchable appearance metadata without internal object paths or vectors."""

    appearance_id: AppearanceId
    camera_id: CameraId
    camera_name: str
    session_id: CameraSessionId
    track_id: Annotated[int, Field(ge=0)]
    first_seen: UtcDatetime
    last_seen: UtcDatetime
    ended_at: UtcDatetime | None
    representative_version: PositiveInt
    bounding_box: BoundingBox
    source_width: PositiveInt
    source_height: PositiveInt
    detector_confidence: UnitScore
    crop_quality: Annotated[float, Field(ge=0.0)]
    model_id: str
    model_revision: str
    similarity: Annotated[float, Field(ge=-1.0, le=1.0)] | None
