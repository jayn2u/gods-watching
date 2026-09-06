"""Validated primitive types shared by boundary models."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, AnyUrl, Field, UrlConstraints
from pydantic_core import PydanticCustomError


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        error_code = "timezone_required"
        error_message = "timestamp must include a UTC offset"
        raise PydanticCustomError(error_code, error_message)
    return value.astimezone(UTC)


def _require_valid_rtsp_port(value: AnyUrl) -> AnyUrl:
    if value.port is not None and value.port < 1:
        error_code = "rtsp_port_out_of_range"
        error_message = "RTSP port must be between 1 and 65535"
        raise PydanticCustomError(error_code, error_message)
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_to_utc)]
RtspUrl = Annotated[
    AnyUrl,
    UrlConstraints(allowed_schemes=["rtsp"], host_required=True, max_length=2048),
    AfterValidator(_require_valid_rtsp_port),
]
DetectionThreshold = Annotated[float, Field(ge=0.1, le=0.95)]
