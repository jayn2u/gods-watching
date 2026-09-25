"""The rehearsal boundary fails closed before any resource is trusted."""

import asyncio
from pathlib import Path

import pytest

from gods_watching.model_selection.rehearsal_stack import (
    RehearsalInputs,
    RehearsalStackError,
    isolated_rehearsal_stack,
)


def _inputs(tmp_path: Path) -> RehearsalInputs:
    dump = tmp_path / "offline.dump"
    dump.write_bytes(b"offline")
    crops = tmp_path / "snapshot"
    crops.mkdir()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "prepared-manifest.json").write_text('{"cuda_device_uuid":"GPU-test"}')
    return RehearsalInputs(dump, crops, assets, "GPU-test", "imported/model")


def test_missing_dump_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.database_dump.unlink()
    with pytest.raises(RehearsalStackError, match="database_dump_unavailable"):
        asyncio.run(_enter(inputs))


def test_missing_snapshot_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.crop_snapshot.rmdir()
    with pytest.raises(RehearsalStackError, match="crop_snapshot_unavailable"):
        asyncio.run(_enter(inputs))


def test_wrong_gpu_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.gpu_uuid = "GPU-other"
    with pytest.raises(RehearsalStackError, match="gpu_uuid_mismatch"):
        asyncio.run(_enter(inputs))


def test_no_qualified_provisioner_refused(tmp_path: Path) -> None:
    with pytest.raises(RehearsalStackError, match="provisioner_unavailable"):
        asyncio.run(_enter(_inputs(tmp_path)))


async def _enter(inputs: RehearsalInputs) -> None:
    async with isolated_rehearsal_stack(inputs):
        pytest.fail("unsafe stack yielded")


def test_live_crop_alias_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setenv("GW_CROPS_ROOT", str(inputs.crop_snapshot))
    with pytest.raises(RehearsalStackError, match="live_crop_alias"):
        asyncio.run(_enter(inputs))
