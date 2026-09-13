"""Appearance persistence and API response contracts."""

from typing import Annotated

from pydantic import Field

from .base import ContractModel
from .identifiers import AppearanceId, CameraId, CameraSessionId
from .primitives import UtcDatetime

PositiveInt = Annotated[int, Field(gt=0)]
UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]
Embedding = Annotated[tuple[float, ...], Field(min_length=512, max_length=512)]


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
