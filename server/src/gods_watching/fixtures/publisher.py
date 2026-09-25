"""Build shell-free FFmpeg commands for one RTSP source session."""

from pathlib import Path
from typing import Final

from pydantic import TypeAdapter, ValidationError

from gods_watching.contracts.primitives import RtspUrl

from .errors import FixturePreparationError

_FFMPEG: Final = "/usr/bin/ffmpeg"
_FFPROBE: Final = "/usr/bin/ffprobe"
_RTSP_ADAPTER: Final[TypeAdapter[RtspUrl]] = TypeAdapter(RtspUrl)


def _parse_rtsp_destination(destination: str) -> str:
    try:
        return str(_RTSP_ADAPTER.validate_python(destination))
    except ValidationError as error:
        raise FixturePreparationError(
            code="invalid_rtsp_destination",
            detail="publisher destination must be a bounded RTSP URL",
        ) from error


def build_publisher_command(source: Path, destination: str) -> tuple[str, ...]:
    """Publish one finite input once; the supervisor owns every reopen boundary."""
    return (
        _FFMPEG,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-nostdin",
        "-re",
        "-protocol_whitelist",
        "file",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-bf",
        "0",
        "-g",
        "60",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        _parse_rtsp_destination(destination),
    )


def build_read_probe_command(destination: str) -> tuple[str, ...]:
    """Probe one RTSP/TCP frame with an explicit protocol and I/O deadline."""
    return (
        _FFPROBE,
        "-v",
        "error",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        "2000000",
        "-protocol_whitelist",
        "rtsp,tcp,udp,rtp",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "%+0.1",
        "-show_entries",
        "stream=codec_name,width,height",
        "-of",
        "json",
        _parse_rtsp_destination(destination),
    )
