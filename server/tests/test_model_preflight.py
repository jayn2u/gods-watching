"""Measured model-switch duration eligibility."""

# ruff: noqa: PLC0415

import asyncio
import math
import threading
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from gods_watching.model_selection.preflight import estimate_switch


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.parametrize("rate", [None, 0.0, -1.0, math.nan, math.inf, -math.inf])
def test_missing_or_invalid_throughput_blocks_switch(rate: float | None) -> None:
    result = estimate_switch(
        target_model_id="target",
        retained_count=1,
        measured_crops_per_second=rate,
        estimated_missing_count=0,
    )
    assert not result.eligible
    assert result.estimated_seconds is None
    assert result.reason == "throughput_unavailable"


def test_zero_crops_still_requires_measured_throughput() -> None:
    result = estimate_switch(
        target_model_id="target",
        retained_count=0,
        measured_crops_per_second=1.0,
        estimated_missing_count=0,
        measured_fixed_seconds=0.0,
    )
    assert result.eligible
    assert result.estimated_seconds == 0


def test_exact_900_second_boundary_is_eligible() -> None:
    result = estimate_switch(
        target_model_id="target",
        retained_count=901,
        measured_crops_per_second=1.0,
        estimated_missing_count=1,
        measured_fixed_seconds=0.0,
    )
    assert result.eligible
    assert result.estimated_seconds == 900


def test_over_900_seconds_is_rejected() -> None:
    result = estimate_switch(
        target_model_id="target",
        retained_count=902,
        measured_crops_per_second=1.0,
        estimated_missing_count=1,
        measured_fixed_seconds=0.0,
    )
    assert not result.eligible
    assert result.reason == "estimate_exceeds_limit"


def test_invalid_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="counts are inconsistent"):
        estimate_switch(
            target_model_id="target",
            retained_count=1,
            measured_crops_per_second=1.0,
            estimated_missing_count=2,
        )


def test_rehearsal_requires_exact_fresh_gpu_uuid_and_full_path(tmp_path: Path) -> None:
    import json
    from datetime import UTC, datetime, timedelta

    from gods_watching.model_selection.preflight import (
        measured_rehearsal,
        rehearsal_path,
        runtime_code_sha256,
    )
    from gods_watching.model_selection.registry import ClipModelPackage

    package = ClipModelPackage(
        model_id="fixture/target",
        revision="abc",
        snapshot_path=Path("/models/fixture"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    (tmp_path / "prepared-manifest.json").write_text(
        json.dumps(
            {
                "cuda_device": "GPU model",
                "cuda_device_uuid": "GPU-actual",
                "image_id": "sha256:triton",
            }
        )
    )
    path = rehearsal_path(tmp_path, package)
    path.parent.mkdir()
    record = {
        "kind": "full_transition_rehearsal_v1",
        "model_id": package.model_id,
        "revision": package.revision,
        "dimension": package.dimension,
        "device": "GPU model",
        "device_uuid": "GPU-actual",
        "detector_resident": True,
        "triton_rpc_measured": True,
        "database_staging_measured": True,
        "activation_measured": True,
        "pipeline_restart_measured": True,
        "sample_count": 16,
        "measured_seconds": 4.0,
        "measured_fixed_seconds": 5.0,
        "measured_at": datetime.now(UTC).isoformat(),
        "retained_corpus_sha256": "a" * 64,
        "runtime_code_sha256": runtime_code_sha256(),
        "triton_image_id": "sha256:triton",
    }
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) == (4.0, 5.0)
    assert measured_rehearsal(tmp_path, package, corpus_sha256="b" * 64) is None
    record["runtime_code_sha256"] = "b" * 64
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) is None
    record["runtime_code_sha256"] = runtime_code_sha256()
    path.write_text(json.dumps(record))
    record["triton_image_id"] = "sha256:other"
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) is None
    record["triton_image_id"] = "sha256:triton"
    path.write_text(json.dumps(record))
    record["device_uuid"] = "GPU-other"
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) is None
    record["device_uuid"] = "GPU-actual"
    record["measured_at"] = (datetime.now(UTC) - timedelta(hours=25)).isoformat()
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) is None
    record["measured_at"] = datetime.now(UTC).isoformat()
    record["triton_rpc_measured"] = False
    path.write_text(json.dumps(record))
    assert measured_rehearsal(tmp_path, package, corpus_sha256="a" * 64) is None


@pytest.mark.anyio
async def test_snapshot_hash_changes_with_retained_crop_bytes() -> None:
    from io import BytesIO

    from PIL import Image

    from gods_watching.model_selection.preflight import scan_retained_snapshot

    class Rows:
        async def partitions(self, size: int) -> AsyncIterator[list[tuple[str, str]]]:
            assert size == 128
            yield [("appearance-1", "crop-1")]

    class Session:
        async def stream(self, statement: object) -> Rows:
            del statement
            return Rows()

    class Store:
        payload = b""

        def read(self, key: str) -> bytes:
            assert key == "crop-1"
            return self.payload

    store = Store()
    digests = []
    for color in ("red", "blue"):
        output = BytesIO()
        Image.new("RGB", (2, 2), color).save(output, format="PNG")
        store.payload = output.getvalue()
        count, missing, digest = await scan_retained_snapshot(Session(), store)  # type: ignore[arg-type]
        assert (count, missing) == (1, 0)
        digests.append(digest)
    assert digests[0] != digests[1]


def test_runtime_hash_includes_modules_outside_model_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gods_watching.model_selection import preflight

    package_root = tmp_path / "gods_watching"
    module = package_root / "pipeline_worker" / "service.py"
    module.parent.mkdir(parents=True)
    module.write_text("VALUE = 1\n")
    marker = package_root / "model_selection" / "preflight.py"
    marker.parent.mkdir()
    marker.write_text("pass\n")
    monkeypatch.setattr(preflight, "__file__", str(marker))
    before = preflight.runtime_code_sha256()
    module.write_text("VALUE = 2\n")
    assert preflight.runtime_code_sha256() != before


@pytest.mark.anyio
async def test_snapshot_decode_does_not_block_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    from io import BytesIO

    from PIL import Image

    from gods_watching.model_selection.preflight import scan_retained_snapshot

    image = BytesIO()
    Image.new("RGB", (2, 2), "red").save(image, format="PNG")
    entered = threading.Event()
    release = threading.Event()
    original_load = Image.Image.load

    def slow_load(self: Image.Image) -> object:
        entered.set()
        assert release.wait(timeout=2)
        return original_load(self)

    monkeypatch.setattr(Image.Image, "load", slow_load)

    class Rows:
        async def partitions(self, size: int) -> AsyncIterator[list[tuple[str, str]]]:
            assert size == 128
            yield [("one", "key")]

    class Session:
        async def stream(self, statement: object) -> Rows:
            del statement
            return Rows()

    class Store:
        def read(self, key: str) -> bytes:
            assert key == "key"
            return image.getvalue()

    scan = asyncio.create_task(scan_retained_snapshot(Session(), Store()))  # type: ignore[arg-type]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.wait_for(asyncio.sleep(0), timeout=1)
        assert not scan.done()
    finally:
        release.set()
    count, missing, digest = await scan
    assert (count, missing) == (1, 0)
    assert digest is not None


def test_benchmark_refuses_insufficient_samples_without_publishing(tmp_path: Path) -> None:
    from gods_watching.model_selection.benchmark import benchmark_package
    from gods_watching.model_selection.preflight import measurement_path
    from gods_watching.model_selection.registry import ClipModelPackage

    package = ClipModelPackage(
        model_id="fixture/target",
        revision="abc",
        snapshot_path=Path("/models/fixture"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    with pytest.raises(ValueError, match="at least 16 crop keys"):
        benchmark_package(
            package,
            assets_root=tmp_path,
            crops_root=tmp_path / "crops",
            sample_keys=["one"],
            detector_path=tmp_path / "detector.pt",
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
        async def partitions(self, size: int) -> AsyncIterator[list[str]]:
            assert size == 128
            yield [good, corrupt, absent]

    class _Session:
        async def stream_scalars(self, statement: object) -> _Rows:
            del statement
            return _Rows()

    assert await scan_retained(_Session(), store) == (3, 2)  # type: ignore[arg-type]


def test_embedding_only_benchmark_does_not_authorize_switch(tmp_path: Path) -> None:
    import json
    from datetime import UTC, datetime

    from gods_watching.model_selection.preflight import measured_rehearsal, measurement_path
    from gods_watching.model_selection.registry import ClipModelPackage

    package = ClipModelPackage(
        model_id="fixture/target",
        revision="abc",
        snapshot_path=Path("/models/fixture"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    (tmp_path / "prepared-manifest.json").write_text(
        json.dumps({"cuda_device": "GPU model", "cuda_device_uuid": "GPU-abc"})
    )
    path = measurement_path(tmp_path, package)
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "model_id": package.model_id,
                "revision": package.revision,
                "dimension": package.dimension,
                "device": "GPU model",
                "device_uuid": "GPU-abc",
                "detector_resident": True,
                "sample_count": 16,
                "measured_seconds": 4.0,
                "measured_at": datetime.now(UTC).isoformat(),
            }
        )
    )
    assert measured_rehearsal(tmp_path, package) is None


def test_embedding_only_900_seconds_and_missing_overhead_fail_closed() -> None:
    unmeasured = estimate_switch(
        retained_count=900,
        measured_crops_per_second=1.0,
        estimated_missing_count=0,
        target_model_id="target",
    )
    assert not unmeasured.eligible
    measured = estimate_switch(
        retained_count=900,
        measured_crops_per_second=1.0,
        estimated_missing_count=0,
        target_model_id="target",
        measured_fixed_seconds=1.0,
    )
    assert measured.estimated_seconds == 901
    assert not measured.eligible


def test_scan_propagates_storage_io_failure() -> None:
    from gods_watching.model_selection.preflight import _count_missing

    class _Store:
        def read(self, key: str) -> bytes:
            del key
            message = "crop inaccessible"
            raise PermissionError(message)

    with pytest.raises(PermissionError, match="crop inaccessible"):
        _count_missing(_Store(), ["key"])  # type: ignore[arg-type]


def test_truncated_jpeg_that_opens_but_cannot_load_is_skipped(tmp_path: Path) -> None:
    from io import BytesIO

    from PIL import Image

    from gods_watching.model_selection.preflight import _count_missing
    from gods_watching.storage import CropObjectStore

    output = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(output, format="JPEG")
    store = CropObjectStore(tmp_path / "crops")
    key = store.write(output.getvalue()[:-12]).object_key
    assert _count_missing(store, [key]) == 1


def test_diagnostic_uuid_does_not_guess_remapped_cuda_ordinal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gods_watching.model_selection.benchmark import _gpu_uuid

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            get_device_properties=lambda _index: SimpleNamespace(name="same GPU model")
        )
    )
    assert _gpu_uuid(fake_torch) == ""


def test_diagnostic_uuid_uses_selected_cuda_device_property(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from gods_watching.model_selection.benchmark import _gpu_uuid

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    seen: list[int] = []

    def properties(index: int) -> object:
        seen.append(index)
        return SimpleNamespace(uuid="GPU-selected")

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=properties))
    assert _gpu_uuid(fake_torch) == "GPU-selected"
    assert seen == [0]
