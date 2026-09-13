"""Bounded, credential-safe RTSP source parsing and frame probing."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final, override

import anyio
from pydantic import AnyUrl, TypeAdapter

from gods_watching.contracts.cameras import CameraTestResponse
from gods_watching.contracts.primitives import RtspUrl

_FFMPEG: Final = "/usr/bin/ffmpeg"
_PROBE_DEADLINE_SECONDS: Final = 10.0
_RTSP_ADAPTER: Final[TypeAdapter[RtspUrl]] = TypeAdapter(RtspUrl)
_VIDEO_LINE: Final = re.compile(
    r"Video:\s*(?P<codec>[A-Za-z0-9_.-]+).*?(?P<width>\d{2,5})x(?P<height>\d{2,5})",
    re.DOTALL,
)


class ProbeFailureCode(StrEnum):
    """Stable sanitized source probe failure classes."""

    INVALID_SOURCE = "invalid_source"
    AUTHENTICATION_FAILED = "authentication_failed"
    UNREACHABLE = "unreachable"
    UNSUPPORTED_CODEC = "unsupported_codec"
    DECODE_FAILED = "decode_failed"
    TIMEOUT = "timeout"
    PROCESS_UNAVAILABLE = "process_unavailable"


@dataclass
class RtspProbeError(RuntimeError):
    """Describe a probe failure without retaining source credentials or stderr."""

    code: ProbeFailureCode
    detail: str

    @override
    def __str__(self) -> str:
        """Return a stable sanitized diagnostic."""
        return f"{self.code.value}: {self.detail}"


@dataclass(frozen=True, slots=True)
class ParsedRtspSource:
    """Hold one validated RTSP URL and its safe routing fields."""

    url: str = field(repr=False)
    host: str
    port: int | None


def parse_rtsp_source(raw: str | AnyUrl) -> ParsedRtspSource:
    """Parse one bounded RTSP URL before any network or subprocess operation."""
    parsed = _RTSP_ADAPTER.validate_python(raw)
    host = parsed.unicode_host()
    if host is None:
        raise RtspProbeError(
            code=ProbeFailureCode.INVALID_SOURCE,
            detail="RTSP source host is required",
        )
    return ParsedRtspSource(url=str(parsed), host=host, port=parsed.port)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Expose only sanitized stream metadata from one decoded frame."""

    source_host: str
    source_port: int | None
    codec: str
    width: int
    height: int

    def to_response(self) -> CameraTestResponse:
        """Convert the probe result to the existing public contract."""
        return CameraTestResponse(
            source_host=self.source_host,
            source_port=self.source_port,
            codec=self.codec,
            width=self.width,
            height=self.height,
        )


@dataclass(frozen=True, slots=True)
class RtspSourceProbe:
    """Decode exactly one H.264 video frame through RTSP over TCP."""

    deadline_seconds: float = _PROBE_DEADLINE_SECONDS

    @staticmethod
    def build_command(source: ParsedRtspSource) -> tuple[str, ...]:
        """Build shell-free FFmpeg arguments with an RTSP/TCP protocol fence."""
        return (
            _FFMPEG,
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostdin",
            "-protocol_whitelist",
            "rtsp,tcp",
            "-rtsp_transport",
            "tcp",
            "-stimeout",
            "10000000",
            "-i",
            source.url,
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            "1",
            "-f",
            "null",
            "-",
        )

    async def probe(self, source: ParsedRtspSource) -> ProbeResult:
        """Run one bounded process and parse metadata only after a frame decoded."""
        try:
            with anyio.fail_after(self.deadline_seconds):
                completed = await anyio.run_process(self.build_command(source), check=False)
        except TimeoutError as error:
            raise RtspProbeError(
                code=ProbeFailureCode.TIMEOUT,
                detail="RTSP source did not decode a frame before the 10 second deadline",
            ) from error
        except OSError as error:
            raise RtspProbeError(
                code=ProbeFailureCode.PROCESS_UNAVAILABLE,
                detail="FFmpeg is unavailable for RTSP probing",
            ) from error
        stderr = completed.stderr.decode("utf-8", errors="replace")
        if completed.returncode != 0:
            raise RtspProbeError(
                code=_failure_code(stderr),
                detail=_failure_detail(_failure_code(stderr)),
            )
        metadata = _parse_video_metadata(stderr)
        if metadata is None:
            raise RtspProbeError(
                code=ProbeFailureCode.DECODE_FAILED,
                detail="RTSP source did not provide a parseable video frame",
            )
        codec, width, height = metadata
        if codec.lower() != "h264":
            raise RtspProbeError(
                code=ProbeFailureCode.UNSUPPORTED_CODEC,
                detail="RTSP source must provide H.264 video",
            )
        return ProbeResult(
            source_host=source.host,
            source_port=source.port,
            codec="h264",
            width=width,
            height=height,
        )


def _parse_video_metadata(stderr: str) -> tuple[str, int, int] | None:
    match = _VIDEO_LINE.search(stderr)
    if match is None:
        return None
    codec = match.group("codec")
    width = int(match.group("width"))
    height = int(match.group("height"))
    if width < 1 or height < 1:
        return None
    return codec, width, height


def _failure_code(stderr: str) -> ProbeFailureCode:
    lowered = stderr.lower()
    if any(marker in lowered for marker in ("401", "unauthorized", "authentication", "forbidden")):
        return ProbeFailureCode.AUTHENTICATION_FAILED
    if any(
        marker in lowered for marker in ("unknown decoder", "unsupported codec", "codec not found")
    ):
        return ProbeFailureCode.UNSUPPORTED_CODEC
    if any(
        marker in lowered
        for marker in ("connection refused", "timed out", "could not connect", "no route")
    ):
        return ProbeFailureCode.UNREACHABLE
    return ProbeFailureCode.DECODE_FAILED


def _failure_detail(code: ProbeFailureCode) -> str:
    details = {
        ProbeFailureCode.AUTHENTICATION_FAILED: "RTSP source authentication failed",
        ProbeFailureCode.UNSUPPORTED_CODEC: "RTSP source must provide H.264 video",
        ProbeFailureCode.UNREACHABLE: "RTSP source could not be reached",
        ProbeFailureCode.DECODE_FAILED: "RTSP source did not decode a video frame",
        ProbeFailureCode.INVALID_SOURCE: "RTSP source is invalid",
        ProbeFailureCode.TIMEOUT: "RTSP source probe timed out",
        ProbeFailureCode.PROCESS_UNAVAILABLE: "FFmpeg is unavailable for RTSP probing",
    }
    return details[code]
