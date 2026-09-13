# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "anyio==4.10.0",
#   "numpy==2.2.6",
#   "pydantic==2.11.7",
#   "tritonclient[grpc]==2.61.0",
# ]
# ///
# ─── How to run ───
# python qa/detector/probe.py ENDPOINT INPUT EMPTY OUTPUT_JSON ANNOTATED_IMAGE

"""Drive the real typed detector adapter and create inspectable QA evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar, Final

import anyio
from pydantic import BaseModel, ConfigDict
from tritonclient.utils import InferenceServerException

from gods_watching.inference.detector import (
    DetectorClient,
    DetectorRequest,
    TritonGrpcDetectorTransport,
)

_ARGUMENT_COUNT: Final = 6
_USAGE: Final = "usage: probe.py ENDPOINT INPUT EMPTY OUTPUT_JSON ANNOTATED_IMAGE"


class StreamDimensions(BaseModel):
    """Parse ffprobe's selected video dimensions."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    width: int
    height: int


class DimensionsEnvelope(BaseModel):
    """Parse the ffprobe stream envelope."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)
    streams: tuple[StreamDimensions, ...]


async def _run(
    endpoint: str,
    input_path: Path,
    empty_path: Path,
    output_path: Path,
    annotated_path: Path,
) -> None:
    transport = TritonGrpcDetectorTransport(url=endpoint)
    client = DetectorClient(transport=transport, timeout_seconds=20.0)
    encoded = input_path.read_bytes()
    empty = empty_path.read_bytes()
    try:
        low = await client.detect(DetectorRequest(encoded_image=encoded, confidence=0.1))
        high = await client.detect(DetectorRequest(encoded_image=encoded, confidence=0.95))
        empty_result = await client.detect(DetectorRequest(encoded_image=empty, confidence=0.1))
        malformed_error = ""
        try:
            _ = await client.detect(DetectorRequest(encoded_image=b"not-an-image", confidence=0.5))
        except InferenceServerException as error:
            malformed_error = str(error)
        post_error = await client.detect(DetectorRequest(encoded_image=encoded, confidence=0.5))
    finally:
        await transport.close()

    batch_counts = [0, 0, 0, 0]

    async def detect_concurrently(index: int, threshold: float) -> None:
        batch_transport = TritonGrpcDetectorTransport(url=endpoint)
        batch_client = DetectorClient(transport=batch_transport, timeout_seconds=20.0)
        try:
            result = await batch_client.detect(
                DetectorRequest(encoded_image=encoded, confidence=threshold)
            )
            batch_counts[index] = result.count
        finally:
            await batch_transport.close()

    async with anyio.create_task_group() as task_group:
        for index, threshold in enumerate((0.1, 0.5, 0.8, 0.95)):
            task_group.start_soon(detect_concurrently, index, threshold)

    dimensions_process = await anyio.run_process(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(input_path),
        ],
        check=True,
    )
    dimensions = DimensionsEnvelope.model_validate_json(dimensions_process.stdout).streams[0]
    width, height = dimensions.width, dimensions.height
    filters = ",".join(
        "".join(
            (
                f"drawbox=x={detection.x1}:y={detection.y1}:",
                f"w={detection.x2 - detection.x1}:h={detection.y2 - detection.y1}:",
                "color=red@0.9:t=5",
            )
        )
        for detection in low.detections
    )
    _ = await anyio.run_process(
        [
            "/usr/bin/ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-vf",
            filters,
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(annotated_path),
        ],
        check=True,
    )

    detections = [
        {
            "xyxy": list(detection.xyxy),
            "confidence": detection.confidence,
            "class_id": detection.class_id,
        }
        for detection in low.detections
    ]
    document = {
        "model": {
            "id": "ultralytics/yolo11s",
            "revision": "v8.3.0",
            "weights_sha256": "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5",
        },
        "source_dimensions": [width, height],
        "low_threshold": 0.1,
        "low_count": low.count,
        "low_min_confidence": min(
            (detection.confidence for detection in low.detections), default=None
        ),
        "high_threshold": 0.95,
        "high_count": high.count,
        "empty_count": empty_result.count,
        "malformed_error": malformed_error,
        "post_error_count": post_error.count,
        "concurrent_thresholds": [0.1, 0.5, 0.8, 0.95],
        "concurrent_counts": batch_counts,
        "detections": detections,
        "coordinates_within_original": all(
            0.0 <= detection.x1 < detection.x2 <= width
            and 0.0 <= detection.y1 < detection.y2 <= height
            for detection in low.detections
        ),
    }
    _ = output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    """Run real detector QA against an already-ready Triton endpoint."""
    if len(sys.argv) != _ARGUMENT_COUNT:
        raise SystemExit(_USAGE)
    anyio.run(
        _run,
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
        Path(sys.argv[5]),
    )


if __name__ == "__main__":
    main()
