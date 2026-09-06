"""Camera request and response contracts."""

from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator
from pydantic_core import PydanticCustomError

from .base import ContractModel
from .identifiers import CameraId
from .primitives import DetectionThreshold, RtspUrl, UtcDatetime

CameraName = Annotated[str, StringConstraints(min_length=1, max_length=80, strip_whitespace=True)]


class CameraCreateRequest(ContractModel):
    """Create a configured RTSP camera without initiating network access."""

    name: CameraName
    source_url: RtspUrl
    detection_enabled: bool = True
    detection_threshold: DetectionThreshold = 0.5


class CameraPatchRequest(ContractModel):
    """Change one or more persisted camera fields."""

    name: CameraName | None = None
    source_url: RtspUrl | None = None
    detection_enabled: bool | None = None
    detection_threshold: DetectionThreshold | None = None

    @model_validator(mode="after")
    def require_change(self) -> Self:
        """Reject an update that cannot change persisted state."""
        if all(
            value is None
            for value in (
                self.name,
                self.source_url,
                self.detection_enabled,
                self.detection_threshold,
            )
        ):
            error_code = "empty_patch"
            error_message = "at least one camera field is required"
            raise PydanticCustomError(error_code, error_message)
        return self


class CameraTestRequest(ContractModel):
    """Describe an RTSP source to probe within the fixed connection deadline."""

    source_url: RtspUrl


class CameraTestResponse(ContractModel):
    """Return sanitized metadata from a successful camera probe."""

    source_host: str
    source_port: int | None
    codec: str
    width: Annotated[int, Field(gt=0)]
    height: Annotated[int, Field(gt=0)]


class CameraResponse(ContractModel):
    """Expose camera state without source credentials or a source URL."""

    camera_id: CameraId
    name: CameraName
    source_host: str
    source_port: int | None
    detection_enabled: bool
    detection_threshold: DetectionThreshold
    version: Annotated[int, Field(ge=1)]
    deleted_at: UtcDatetime | None
