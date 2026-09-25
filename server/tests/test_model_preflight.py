"""Measured model-switch duration eligibility."""

# ruff: noqa: PLC0415

import math
from pathlib import Path

import pytest

from gods_watching.model_selection.preflight import estimate_switch


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize("rate", [None, 0.0, -1.0, math.nan, math.inf, -math.inf])
def test_missing_or_invalid_throughput_blocks_switch(rate: float | None) -> None:
    result = estimate_switch(
        target_model_id="target", retained_count=1,
        measured_crops_per_second=rate, estimated_missing_count=0,
    )
    assert not result.eligible
    assert result.estimated_seconds is None
    assert result.reason == "throughput_unavailable"


def test_zero_crops_still_requires_measured_throughput() -> None:
    result = estimate_switch(
        target_model_id="target", retained_count=0,
        measured_crops_per_second=1.0, estimated_missing_count=0,
    )
    assert result.eligible
    assert result.estimated_seconds == 0


def test_exact_900_second_boundary_is_eligible() -> None:
    result = estimate_switch(
        target_model_id="target", retained_count=901,
        measured_crops_per_second=1.0, estimated_missing_count=1,
    )
    assert result.eligible
    assert result.estimated_seconds == 900


def test_over_900_seconds_is_rejected() -> None:
    result = estimate_switch(
        target_model_id="target", retained_count=902,
        measured_crops_per_second=1.0, estimated_missing_count=1,
    )
    assert not result.eligible
    assert result.reason == "estimate_exceeds_limit"


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="counts are inconsistent"):
        estimate_switch(
            target_model_id="target", retained_count=1,
            measured_crops_per_second=1.0, estimated_missing_count=2,
        )


def test_measurement_requires_exact_fresh_deployment_identity(tmp_path: Path) -> None:
    import json
    from datetime import UTC, datetime, timedelta
    from pathlib import Path

    from gods_watching.model_selection.preflight import measured_rate, measurement_path
    from gods_watching.model_selection.registry import ClipModelPackage

    package = ClipModelPackage(
        model_id="fixture/target", revision="abc", snapshot_path=Path("/models/fixture"),
        dimension=512, processor="CLIPProcessor", runtime="transformers",
    )
    (tmp_path / "prepared-manifest.json").write_text(json.dumps({"cuda_device": "GPU-1"}))
    path = measurement_path(tmp_path, package)
    path.parent.mkdir()
    record = {
        "model_id": package.model_id, "revision": package.revision,
        "dimension": package.dimension, "device": "GPU-1", "detector_resident": True,
        "sample_count": 16, "measured_seconds": 4.0,
        "measured_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(record))
    assert measured_rate(tmp_path, package) == 4.0
    record["device"] = "GPU-2"
    path.write_text(json.dumps(record))
    assert measured_rate(tmp_path, package) is None
    record["device"] = "GPU-1"
    record["measured_at"] = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    path.write_text(json.dumps(record))
    assert measured_rate(tmp_path, package) is None
    record["measured_at"] = datetime.now(UTC).isoformat()
    record["sample_count"] = 0
    path.write_text(json.dumps(record))
    assert measured_rate(tmp_path, package) is None


def test_benchmark_refuses_insufficient_samples_without_publishing(tmp_path: Path) -> None:
    from gods_watching.model_selection.benchmark import benchmark_package
    from gods_watching.model_selection.preflight import measurement_path
    from gods_watching.model_selection.registry import ClipModelPackage

    package = ClipModelPackage(
        model_id="fixture/target", revision="abc", snapshot_path=Path("/models/fixture"),
        dimension=512, processor="CLIPProcessor", runtime="transformers",
    )
    with pytest.raises(ValueError, match="at least 16 crop keys"):
        benchmark_package(
            package, assets_root=tmp_path, crops_root=tmp_path / "crops",
            sample_keys=["one"], detector_path=tmp_path / "detector.pt",
        )
    assert not measurement_path(tmp_path, package).exists()


@pytest.mark.anyio
async def test_scan_counts_missing_and_corrupt_crops(tmp_path: Path) -> None:
    from io import BytesIO
    from uuid import uuid4

    from PIL import Image

    from gods_watching.model_selection.preflight import scan_retained
    from gods_watching.storage import CropObjectStore

    store = CropObjectStore(tmp_path / "crops")
    output = BytesIO()
    Image.new("RGB", (2, 2)).save(output, format="JPEG")
    good = store.write(output.getvalue()).object_key
    corrupt = store.write(b"not a jpeg").object_key
    absent_id = uuid4()
    absent = f"{absent_id.hex[:2]}/{absent_id.hex[2:4]}/{absent_id}.jpg"

    class _Rows:
        def all(self) -> list[str]:
            return [good, corrupt, absent]

    class _Session:
        async def scalars(self, statement: object) -> _Rows:
            del statement
            return _Rows()

    assert await scan_retained(_Session(), store) == (3, 2)  # type: ignore[arg-type]
