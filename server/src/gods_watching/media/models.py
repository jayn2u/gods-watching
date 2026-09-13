"""Typed values shared by the media control and WHEP proxy adapters."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType
from uuid import UUID

from gods_watching.contracts.identifiers import CameraId

MediaPath = NewType("MediaPath", str)
MediaSessionId = NewType("MediaSessionId", str)
SourceGenerationId = NewType("SourceGenerationId", UUID)
WhepResourceId = NewType("WhepResourceId", UUID)


@dataclass(frozen=True, slots=True)
class AuthorizedMediaSession:
    """Identify the already-authorized application session that owns live media."""

    session_id: MediaSessionId


@dataclass(frozen=True, slots=True)
class GatewayResponse:
    """Carry the small response surface used by the private gateway adapter."""

    status_code: int
    body: bytes
    content_type: str | None = None
    location: str | None = None
    etag: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyResponse:
    """Return only headers safe to expose at the application boundary."""

    status_code: int
    body: bytes
    content_type: str | None = None
    location: str | None = None
    etag: str | None = None


class GenerationEventKind(StrEnum):
    """Describe the lifecycle of one camera source generation."""

    STARTED = "started"
    SOURCE_LOST = "source_lost"
    ENDED = "ended"


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """Fence downstream work to one camera source generation."""

    camera_id: CameraId
    generation_id: SourceGenerationId
    kind: GenerationEventKind
