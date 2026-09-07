"""Schema-checked task-9 runtime artifacts."""

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict


class ArtifactModel(BaseModel):
    """Keep task-9 evidence immutable and schema checked."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class RtcSample(ArtifactModel):
    """Record one real Chromium inbound-video counter sample."""

    frames_decoded: int
    frames_received: int
    packets_received: int


class DecodeObservation(ArtifactModel):
    """Record the real receive-only browser resource lifecycle."""

    post_status: int
    delete_status: int
    resource_location: str
    connection_state: str
    transceiver_direction: Literal["recvonly"]
    codec_mime_type: str
    samples: tuple[RtcSample, ...]
    video_width: int
    video_height: int


class DecodeArtifact(ArtifactModel):
    """Wrap the Chromium observation and console-error audit."""

    mode: Literal["decode"]
    observation: DecodeObservation
    browser_errors: tuple[str, ...]


class DeniedObservation(ArtifactModel):
    """Record an application request with no authorized session context."""

    status_code: int


class DeniedBrowserArtifact(ArtifactModel):
    """Wrap the real Chromium authorization-denial observation."""

    mode: Literal["denied"]
    observation: DeniedObservation
    browser_errors: tuple[str, ...]


class WriteArtifact(ArtifactModel):
    """Record the live container mount and writable-layer inspection."""

    gateway_read_only: bool
    mounts: tuple[str, ...]
    docker_diff: tuple[str, ...]
    recording_mount_present: bool


class DeniedArtifact(ArtifactModel):
    """Combine browser and RTSP publish/read denial observables."""

    browser_status: int
    rtsp_read_returncode: int
    rtsp_publish_returncode: int
    credential_text_persisted: bool


class PublisherArtifact(ArtifactModel):
    """Record the finite publisher boundary observed through the real gateway."""

    first_returncode: int
    path_not_ready_between_generations: bool
    second_generation_ready: bool
    seamless_stream_loop_used: bool


class StackPaths(ArtifactModel):
    """Locate the persistent artifacts produced by one isolated stack."""

    browser: Path
    writes: Path
    denied: Path
    publisher: Path
