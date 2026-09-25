"""Offline rehearsal resource safety checks with a fake Docker boundary."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

from gods_watching.model_selection.rehearsal_stack import (
    RehearsalInputs,
    RehearsalStackError,
    isolated_rehearsal_stack,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


# ruff: noqa: EM101


class FakeDocker:
    """Observe exact argv and emulate only the isolated container queries."""

    def __init__(
        self,
        *,
        fail_restore: bool = False,
        crop_keys: str = "",
        gpu: str = GPU,
        internal: bool = True,
        fail_pg_run: bool = False,
    ) -> None:
        self.calls: list[list[str]] = []
        self.fail_restore = fail_restore
        self.crop_keys = crop_keys
        self.gpu = gpu
        self.internal = internal
        self.fail_pg_run = fail_pg_run

    async def run(self, argv: Sequence[str], *, check: bool = True) -> str:  # noqa: PLR0911
        _ = check
        args = list(argv)
        self.calls.append(args)
        if (
            args[:2] == ["docker", "run"]
            and args[-1] == "pgvector/pgvector:0.8.1-pg17"
            and self.fail_pg_run
        ):
            raise RehearsalStackError("docker_command_failed")
        if "pg_restore" in args and self.fail_restore:
            raise RehearsalStackError("docker_command_failed")
        if args[:3] == ["docker", "network", "inspect"]:
            return "true" if self.internal else "false"
        if args[:2] == ["docker", "inspect"] and args[-2] == "{{json .HostConfig}}":
            network = args[-1].rsplit("-", 1)[0]
            return f'{{"NetworkMode":"{network}","PortBindings":null}}'
        if args[:2] == ["docker", "inspect"]:
            return "pg-container-id" if args[-1].endswith("-pg") else "triton-container-id"
        if "nvidia-smi" in args:
            return self.gpu
        if "psql" in args and "crop_object_key" in args[-1]:
            return self.crop_keys
        if "psql" in args:
            return "123456789|16384"
        return "created"


def _inputs(tmp_path: Path) -> RehearsalInputs:
    dump = tmp_path / "offline.dump"
    dump.write_bytes(b"offline")
    crops = tmp_path / "snapshot"
    crops.mkdir()
    assets = tmp_path / "assets"
    (assets / "clip").mkdir(parents=True)
    (assets / "prepared-manifest.json").write_text(
        f'{{"cuda_device_uuid":"{GPU}","detector_resident":true}}'
    )
    return RehearsalInputs(dump, crops, assets, GPU, "openai/clip-vit-base-patch16")


async def _enter(inputs: RehearsalInputs, runner: FakeDocker) -> None:
    async with isolated_rehearsal_stack(inputs, runner=runner) as stack:
        assert stack.network_name.startswith("gw-rehearsal-")
        assert stack.database_system_id == "123456789"
        assert stack.database_oid == "16384"
        assert stack.postgres_container_id != stack.triton_container_id
        assert stack.crops_root.is_dir()
        assert stack.triton_url.endswith(":8001")


def test_missing_dump_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.database_dump.unlink()
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="database_dump_unavailable"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_missing_snapshot_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.crop_snapshot.rmdir()
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="crop_snapshot_unavailable"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_wrong_gpu_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    inputs.gpu_uuid = "GPU-99999999-1234-1234-1234-123456789abc"
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="gpu_uuid_mismatch"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_internal_stack_tears_down(tmp_path: Path) -> None:
    runner = FakeDocker()
    asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[0][:6] == [
        "docker",
        "network",
        "create",
        "--internal",
        "--driver",
        "bridge",
    ]
    run_calls = [call for call in runner.calls if call[:2] == ["docker", "run"]]
    assert len(run_calls) == 2
    assert all("--publish" not in call and "--network" in call for call in run_calls)
    assert "--gpus" in run_calls[1]
    assert f"device={GPU}" in run_calls[1]
    assert runner.calls[-3][1:3] == ["rm", "--force"]
    assert runner.calls[-2][1:3] == ["rm", "--force"]
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_failed_restore_tears_down(tmp_path: Path) -> None:
    runner = FakeDocker(fail_restore=True)
    with pytest.raises(RehearsalStackError, match="docker_command_failed"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_incomplete_crops_refused_and_tears_down(tmp_path: Path) -> None:
    runner = FakeDocker(crop_keys="00/00/00000000-0000-4000-8000-000000000000.jpg")
    with pytest.raises(RehearsalStackError, match="missing_crops"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_live_crop_alias_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = _inputs(tmp_path)
    monkeypatch.setenv("GW_CROPS_ROOT", str(inputs.crop_snapshot))
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="live_crop_alias"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_wrong_container_gpu_refused(tmp_path: Path) -> None:
    runner = FakeDocker(gpu="GPU-99999999-1234-1234-1234-123456789abc")
    with pytest.raises(RehearsalStackError, match="gpu_uuid_mismatch"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_noninternal_network_refused(tmp_path: Path) -> None:
    runner = FakeDocker(internal=False)
    with pytest.raises(RehearsalStackError, match="network_isolation_unverified"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_partial_container_creation_cleanup(tmp_path: Path) -> None:
    runner = FakeDocker(fail_pg_run=True)
    with pytest.raises(RehearsalStackError, match="docker_command_failed"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-3][1:3] == ["rm", "--force"]
    assert runner.calls[-2][1:3] == ["rm", "--force"]
    assert runner.calls[-1][1:3] == ["network", "rm"]
