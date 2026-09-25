"""Offline deployment-GPU throughput proof for manual model transitions."""

# ruff: noqa: TC001, TC003, TRY003, EM101

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from PIL import Image

from gods_watching.model_selection.preflight import MIN_SAMPLE_COUNT, measurement_path
from gods_watching.model_selection.registry import ClipModelPackage
from gods_watching.storage import CropObjectStore


def benchmark_package(
    package: ClipModelPackage,
    *,
    assets_root: Path,
    crops_root: Path,
    sample_keys: list[str],
    detector_path: Path,
) -> Path:
    """Measure 16 or more real crops with YOLO resident, then publish atomically."""
    if len(sample_keys) < MIN_SAMPLE_COUNT:
        raise ValueError("at least 16 crop keys are required")
    import torch  # noqa: PLC0415
    from transformers import AutoProcessor, CLIPModel  # noqa: PLC0415
    from ultralytics import YOLO  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError("deployment CUDA device is unavailable")
    detector = YOLO(str(detector_path)).to("cuda")
    if detector is None:
        raise RuntimeError("detector is not resident")
    processor = AutoProcessor.from_pretrained(
        package.snapshot_path, local_files_only=True, trust_remote_code=False
    )
    model = CLIPModel.from_pretrained(
        package.snapshot_path, local_files_only=True, trust_remote_code=False
    ).to("cuda").eval()
    crop_store = CropObjectStore(crops_root)
    images = []
    for key in sample_keys:
        with Image.open(BytesIO(crop_store.read(key))) as image:
            images.append(image.convert("RGB"))
    # Warm the exact inference path before timing. Keep detector strongly referenced.
    with torch.inference_mode():
        first = processor(images=images[0], return_tensors="pt")
        _ = model.get_image_features(**{k: v.to("cuda") for k, v in first.items()})
        torch.cuda.synchronize()
        start = time.perf_counter()
        for image in images:
            inputs = processor(images=image, return_tensors="pt")
            features = model.get_image_features(
                **{k: v.to("cuda") for k, v in inputs.items()}
            )
            if not torch.isfinite(features).all():
                raise RuntimeError("nonfinite image embedding")
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError("invalid measured duration")
    record = {
        "kind": "embedding_only_diagnostic",
        "model_id": package.model_id,
        "revision": package.revision,
        "dimension": package.dimension,
        "device": torch.cuda.get_device_name(0),
        "device_uuid": _gpu_uuid(torch),
        "detector_resident": True,
        "sample_count": len(images),
        "measured_seconds": seconds,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    destination = measurement_path(assets_root, package)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            json.dump(record, output, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _gpu_uuid(torch: object) -> str:
    """Read the stable identity of CUDA device zero in this GPU namespace."""
    # Torch builds differ in whether device properties expose UUID.
    value = getattr(torch.cuda.get_device_properties(0), "uuid", "")
    if value:
        return str(value)
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return ""
    result = subprocess.run(  # noqa: S603
        [executable, "--query-gpu=uuid", "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True, check=False, timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else ""
