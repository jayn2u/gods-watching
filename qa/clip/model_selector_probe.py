# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "anyio==4.10.0",
#   "numpy==2.2.6",
#   "pillow==11.3.0",
#   "pydantic==2.11.7",
#   "tritonclient[grpc]==2.61.0",
# ]
# ///

"""Exercise every registered CLIP package through the typed runtime boundary."""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from typing import cast

import anyio
from PIL import Image

from gods_watching.inference.clip import ClipAdapter, ClipRuntimeManager, TritonClipTransport
from gods_watching.inference.detector import (
    DetectorClient,
    DetectorRequest,
    TritonGrpcDetectorTransport,
)
from gods_watching.model_selection.registry import ClipModelRegistry


def _jpeg_fixture() -> bytes:
    """Create a deterministic in-memory image without touching deployment data."""
    output = BytesIO()
    with Image.new("RGB", (64, 64), color=(91, 117, 163)) as image:
        image.save(output, format="JPEG", quality=90)
    return output.getvalue()


async def _run(url: str, output_path: Path) -> None:
    """Verify detector inference and image/text CLIP dimensions and norms."""
    detector_transport = TritonGrpcDetectorTransport(url=url)
    # Cold Triton detector initialization can exceed a real-time worker deadline.
    detector = DetectorClient(transport=detector_transport, timeout_seconds=20.0)
    try:
        detector_ready = await detector_transport.ready()
        if not detector_ready:
            error_code = "detector_not_ready"
            raise RuntimeError(error_code)
        image_bytes = _jpeg_fixture()
        records: list[dict[str, object]] = []
        async with ClipRuntimeManager(url) as runtime:
            for package in ClipModelRegistry().packages:
                identity = await runtime.load_model(package)
                detector_result = await detector.detect(
                    DetectorRequest(encoded_image=image_bytes, confidence=0.5)
                )
                async with TritonClipTransport(url, package=package) as transport:
                    adapter = ClipAdapter(transport, package=package)
                    image = await adapter.embed_image(image_bytes)
                    text = await adapter.embed_text("a person with a red bag")
                image_norm = math.sqrt(sum(value * value for value in image))
                text_norm = math.sqrt(sum(value * value for value in text))
                if len(image) != package.dimension or len(text) != package.dimension:
                    error_code = f"clip_dimension_mismatch: {package.model_id}"
                    raise RuntimeError(error_code)
                if not math.isclose(image_norm, 1.0, abs_tol=1e-3):
                    error_code = f"clip_image_norm_invalid: {package.model_id}"
                    raise RuntimeError(error_code)
                if not math.isclose(text_norm, 1.0, abs_tol=1e-3):
                    error_code = f"clip_text_norm_invalid: {package.model_id}"
                    raise RuntimeError(error_code)
                records.append(
                    {
                        "model_id": identity.model_id,
                        "revision": identity.revision,
                        "dimension": package.dimension,
                        "detector_count": detector_result.count,
                        "image_norm": image_norm,
                        "text_norm": text_norm,
                    }
                )
            await runtime.unload_model()
    finally:
        await detector_transport.close()
    _ = output_path.write_text(
        json.dumps(
            {"detector_ready": detector_ready, "models": records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _main() -> int:
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("--url", default="localhost:8001")
    _ = parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = cast("str", args.url)
    output_path = cast("Path", args.output)
    _ = anyio.run(_run, url, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
