"""PyAV RTSP/TCP decoder that drains into one in-memory latest-frame slot."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fractions import Fraction
from math import isfinite
from threading import Event
from time import monotonic
from typing import Final, Protocol, override

import anyio
import av
import numpy as np
from anyio.to_thread import run_sync
from av.error import FFmpegError

from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.media.models import SourceGenerationId
from gods_watching.tracking.models import DetectorInputReference

from .models import DecodedFrame, LatestFrameSlot

_RECONNECT_DELAYS: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_JPEG_TIME_BASE: Final[Fraction] = Fraction(1, 25)
_MAX_RECONNECT_DELAY: Final[float] = 30.0


@dataclass(frozen=True, slots=True)
class RtspDecodeError(RuntimeError):
    """Describe a sanitized RTSP decode failure without source credentials."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class DecoderConfigurationError(ValueError):
    """Describe invalid connection timeout or reconnect backoff settings."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class DecoderStatusSink(Protocol):
    """Receive sanitized source-state observations from a reconnecting decoder."""

    async def __call__(self, detail: str) -> None:
        """Consume a source EOF or decode failure description."""
        ...


DecodedFrameSink = Callable[[DecodedFrame], None]
GenerationReconnect = Callable[[GenerationBinding], Awaitable[GenerationBinding]]


@dataclass(frozen=True, slots=True)
class DecoderOptions:
    """Pin bounded TCP RTSP connection and reconnect timing."""

    connection_timeout_seconds: float = 5.0
    reconnect_delays: tuple[float, ...] = _RECONNECT_DELAYS

    def __post_init__(self) -> None:
        """Reject unbounded connection or reconnect settings before opening PyAV."""
        if not isfinite(self.connection_timeout_seconds) or self.connection_timeout_seconds <= 0.0:
            raise DecoderConfigurationError(detail="connection timeout must be positive and finite")
        if not self.reconnect_delays or any(
            not isfinite(delay) or delay < 0.0 or delay > _MAX_RECONNECT_DELAY
            for delay in self.reconnect_delays
        ):
            raise DecoderConfigurationError(
                detail="reconnect delays must stay within 0..30 seconds"
            )


@dataclass(frozen=True, slots=True)
class DecoderConfiguration:
    """Carry one authorized source and its bounded decoder dependencies."""

    source_url: str
    source_generation_id: SourceGenerationId
    slot: LatestFrameSlot
    options: DecoderOptions = field(default_factory=DecoderOptions)
    frame_sink: DecodedFrameSink | None = None
    generation_binding: GenerationBinding | None = None
    generation_reconnect: GenerationReconnect | None = None

    def __post_init__(self) -> None:
        """Reject a generation callback that cannot preserve the source fence."""
        if self.generation_binding is not None and (
            self.generation_binding.source_generation_id != self.source_generation_id
        ):
            raise DecoderConfigurationError(
                detail="generation binding does not match source generation"
            )
        if self.generation_reconnect is not None and self.generation_binding is None:
            raise DecoderConfigurationError(
                detail="generation reconnect requires an initial generation binding"
            )


class PyAvRtspDecoder:
    """Drain an H.264 RTSP/TCP source while retaining only the newest frame."""

    def __init__(
        self,
        configuration: DecoderConfiguration,
    ) -> None:
        """Create a decoder for one already-authorized source generation."""
        self._source_url: str = configuration.source_url
        self._source_generation_id: SourceGenerationId = configuration.source_generation_id
        self._slot: LatestFrameSlot = configuration.slot
        self._options: DecoderOptions = configuration.options
        self._frame_sink: DecodedFrameSink | None = configuration.frame_sink
        self._generation_binding: GenerationBinding | None = configuration.generation_binding
        self._generation_reconnect: GenerationReconnect | None = configuration.generation_reconnect
        self._stop_event: Event = Event()
        self._frame_sequence: int = 0

    def stop(self) -> None:
        """Request cooperative decoder cancellation and release the latest slot."""
        self._stop_event.set()
        self._slot.clear()

    async def run(self) -> None:
        """Decode one connection until EOF, cancellation, or a typed source error."""
        try:
            await run_sync(self._decode_connection, abandon_on_cancel=True)
        except anyio.get_cancelled_exc_class():
            self.stop()
            raise

    async def run_forever(self, *, status_sink: DecoderStatusSink | None = None) -> None:
        """Reconnect after EOF/errors with bounded backoff until stopped."""
        delay_index = 0
        while not self._stop_event.is_set():
            try:
                await self.run()
            except RtspDecodeError as error:
                if status_sink is not None:
                    await status_sink(str(error))
            else:
                if status_sink is not None:
                    await status_sink("RTSP source reached EOF")
            if self._stop_event.is_set():
                return
            await self._refresh_generation()
            delay = self._options.reconnect_delays[
                min(delay_index, len(self._options.reconnect_delays) - 1)
            ]
            delay_index += 1
            remaining = delay
            while remaining > 0.0 and not self._stop_event.is_set():
                step = min(remaining, 0.1)
                await anyio.sleep(step)
                remaining -= step

    async def _refresh_generation(self) -> None:
        owner = self._generation_reconnect
        binding = self._generation_binding
        if owner is None or binding is None:
            return
        self._slot.clear()
        replacement = await owner(binding)
        self._generation_binding = replacement
        self._source_generation_id = replacement.source_generation_id

    def _decode_connection(self) -> None:
        options = {
            "rtsp_transport": "tcp",
            "timeout": str(int(self._options.connection_timeout_seconds * 1_000_000)),
        }
        try:
            with av.open(self._source_url, mode="r", options=options) as container:
                for video_frame in container.decode(video=0):
                    if self._stop_event.is_set():
                        return
                    self._frame_sequence += 1
                    ingress_utc = datetime.now(UTC)
                    ingress_monotonic = monotonic()
                    decoded = _decoded_frame(
                        video_frame,
                        source_generation_id=self._source_generation_id,
                        sequence=self._frame_sequence,
                        ingress_utc=ingress_utc,
                        ingress_monotonic=ingress_monotonic,
                    )
                    self._slot.put(decoded)
                    if self._frame_sink is not None:
                        self._frame_sink(decoded)
        except (FFmpegError, OSError, ValueError) as error:
            raise RtspDecodeError(detail="RTSP/TCP source decode failed") from error


def _decoded_frame(
    video_frame: av.VideoFrame,
    *,
    source_generation_id: SourceGenerationId,
    sequence: int,
    ingress_utc: datetime,
    ingress_monotonic: float,
) -> DecodedFrame:
    rgb = video_frame.to_ndarray(format="rgb24")
    rgb_array = np.ascontiguousarray(rgb, dtype=np.uint8)
    encoded = _encode_jpeg(video_frame)
    return DecodedFrame(
        source_generation_id=source_generation_id,
        ingress_utc=ingress_utc,
        ingress_monotonic=ingress_monotonic,
        reference=DetectorInputReference(f"{source_generation_id}:{sequence}"),
        width=video_frame.width,
        height=video_frame.height,
        rgb_bytes=rgb_array.tobytes(),
        encoded_image=encoded,
    )


def _encode_jpeg(video_frame: av.VideoFrame) -> bytes:
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = video_frame.width
    codec.height = video_frame.height
    codec.pix_fmt = "yuvj420p"
    codec.time_base = _JPEG_TIME_BASE
    converted = video_frame.reformat(format="yuvj420p")
    packets = [bytes(packet) for packet in codec.encode(converted)]
    packets.extend(bytes(packet) for packet in codec.encode(None))
    return b"".join(packets)


__all__ = [
    "DecodedFrameSink",
    "DecoderConfiguration",
    "DecoderConfigurationError",
    "DecoderOptions",
    "DecoderStatusSink",
    "GenerationReconnect",
    "PyAvRtspDecoder",
    "RtspDecodeError",
]
