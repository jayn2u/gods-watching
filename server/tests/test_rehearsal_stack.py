"""Offline rehearsal resource safety checks with a fake Docker boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from gods_watching.model_selection.assets import IDENTITY_MARKER_NAME
from gods_watching.model_selection.registry import load_clip_registry
from gods_watching.model_selection.rehearsal_stack import (
    DockerCommandError,
    RehearsalInputs,
    RehearsalStackError,
    _wait_postgres,
    isolated_rehearsal_stack,
)
from gods_watching.setup.model_preparation import GpuModelProof, PreparedManifest

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

GPU = "GPU-12345678-1234-1234-1234-123456789abc"


# ruff: noqa: EM101


class FakeDocker:
    """Observe exact argv and emulate only the isolated container queries."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        fail_restore: bool = False,
        crop_keys: str = "",
        gpu: str = GPU,
        internal: bool = True,
        fail_pg_run: bool = False,
        fail_network_create: bool = False,
        gateway_mode: str = "isolated",
        ipv6: bool = False,
        fail_cleanup: bool = False,
        host_gateway: str | None = None,
        image_id: str = "sha256:image",
    ) -> None:
        self.calls: list[list[str]] = []
        self.fail_restore = fail_restore
        self.crop_keys = crop_keys
        self.gpu = gpu
        self.internal = internal
        self.fail_pg_run = fail_pg_run
        self.fail_network_create = fail_network_create
        self.created: set[str] = set()
        self.gateway_mode = gateway_mode
        self.ipv6 = ipv6
        self.fail_cleanup = fail_cleanup
        self.host_gateway = host_gateway
        self.image_id = image_id

    async def run(self, argv: Sequence[str], *, check: bool = True) -> str:  # noqa: C901, PLR0911, PLR0912
        _ = check
        args = list(argv)
        self.calls.append(args)
        if args[:3] == ["docker", "network", "create"]:
            if self.fail_network_create:
                raise DockerCommandError(stderr="network creation failed", returncode=1)
            self.created.add(args[-1])
            return "created"
        if args[:2] == ["docker", "run"]:
            if args[-1] == "pgvector/pgvector:0.8.1-pg17" and self.fail_pg_run:
                raise DockerCommandError(stderr="PostgreSQL start failed", returncode=1)
            self.created.add(args[args.index("--name") + 1])
            return "created"
        if args[:3] == ["docker", "rm", "--force"]:
            if self.fail_cleanup:
                raise DockerCommandError(stderr="daemon unavailable", returncode=1)
            if args[-1] not in self.created:
                raise DockerCommandError(stderr=f"No such container: {args[-1]}", returncode=1)
            self.created.remove(args[-1])
            return "removed"
        if args[:3] == ["docker", "network", "rm"]:
            if args[-1] not in self.created:
                raise DockerCommandError(
                    stderr=(
                        f"Error response from daemon: network {args[-1]} not found\nexit status 1"
                    ),
                    returncode=1,
                )
            self.created.remove(args[-1])
            return "removed"
        if "pg_restore" in args and self.fail_restore:
            raise RehearsalStackError("docker_command_failed")
        if args[:3] == ["docker", "image", "inspect"]:
            return self.image_id
        if args[:3] == ["docker", "network", "inspect"]:
            return json.dumps(
                {
                    "Internal": self.internal,
                    "EnableIPv6": self.ipv6,
                    "Options": {"com.docker.network.bridge.gateway_mode_ipv4": self.gateway_mode},
                    "IPAM": {"Config": [{"Subnet": "172.20.0.0/16", "Gateway": self.host_gateway}]},
                }
            )
        if args[:2] == ["docker", "inspect"] and args[-2] == "{{json .HostConfig}}":
            network = args[-1].rsplit("-", 1)[0]
            return f'{{"NetworkMode":"{network}","PortBindings":null}}'
        if args[:2] == ["docker", "inspect"]:
            return "pg-container-id" if args[-1].endswith("-pg") else "triton-container-id"
        if "nvidia-smi" in args:
            return self.gpu
        if any("PGPASSWORD" in argument and "SELECT 1" in argument for argument in args):
            return "1"
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
    snapshot = assets / "clip"
    snapshot.mkdir(parents=True)
    payload = b"immutable checkpoint fixture"
    (snapshot / "weights.bin").write_bytes(payload)
    package = load_clip_registry(assets).require("openai/clip-vit-base-patch16")
    lock = assets / "models.lock.json"
    lock.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": package.model_id,
                        "revision": package.revision,
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "files": [
                            {
                                "path": "clip/weights.bin",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload),
                            }
                        ],
                    }
                ],
            }
        )
    )
    (snapshot / IDENTITY_MARKER_NAME).write_text(
        json.dumps(
            {
                "model_id": package.model_id,
                "revision": package.revision,
                "dimension": package.dimension,
                "processor": package.processor,
                "runtime": package.runtime,
            }
        )
    )
    model_proof = GpuModelProof(
        model_id=package.model_id,
        revision=package.revision,
        dimension=package.dimension,
        image_dimension=package.dimension,
        text_dimension=package.dimension,
        image_norm=1.0,
        text_norm=1.0,
    )
    proof = PreparedManifest(
        schema_version="1",
        lock_sha256=hashlib.sha256(lock.read_bytes()).hexdigest(),
        dockerfile_sha256="d",
        inference_lock_sha256="i",
        image_id="sha256:image",
        base_digest="sha256:base",
        python_abi="cp312",
        torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128",
        cuda_version="12.8",
        cuda_device="test GPU",
        cuda_device_uuid=GPU,
        processor="CLIPProcessor",
        clip_class="CLIPModel",
        yolo_class="YOLO",
        files_validated=1,
        cuda_available=True,
        cuda_operation=1.0,
        detector_resident=True,
        model_proofs=(model_proof,),
    )
    (assets / "prepared-manifest.json").write_text(proof.model_dump_json())
    return RehearsalInputs(dump, crops, assets, GPU, package.model_id, model_lock=lock)


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
    network_call = next(
        call for call in runner.calls if call[:3] == ["docker", "network", "create"]
    )
    assert network_call[:3] == ["docker", "network", "create"]
    assert "--internal" in network_call
    assert "--ipv6=false" in network_call
    assert "com.docker.network.bridge.gateway_mode_ipv4=isolated" in network_call
    run_calls = [call for call in runner.calls if call[:2] == ["docker", "run"]]
    assert len(run_calls) == 2
    assert all("--publish" not in call and "--network" in call for call in run_calls)
    assert "--gpus" in run_calls[1]
    assert f"device={GPU}" in run_calls[1]
    assert runner.calls[-3][1:3] == ["rm", "--force"]
    assert runner.calls[-2][1:3] == ["rm", "--force"]
    assert runner.calls[-1][1:3] == ["network", "rm"]
    assert any(
        "SELECT crop_object_key FROM appearances WHERE tombstoned_at IS NULL" in call
        for command in runner.calls for call in command
    )


def test_postgres_wait_requires_tcp_connection_to_restored_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SocketReadyOnly:
        def __init__(self) -> None:
            self.calls: list[list[str]] = []

        async def run(self, argv: Sequence[str], *, check: bool = True) -> str:
            _ = check
            args = list(argv)
            self.calls.append(args)
            if "pg_isready" in args:
                return "accepting connections"
            raise DockerCommandError(
                stderr='FATAL: database "gods_watching" does not exist', returncode=2
            )

    async def no_wait(_seconds: float) -> None:
        return

    runner = SocketReadyOnly()
    monkeypatch.setattr("gods_watching.model_selection.rehearsal_stack.asyncio.sleep", no_wait)
    with pytest.raises(RehearsalStackError, match="postgres_not_ready"):
        asyncio.run(_wait_postgres(runner, "gw-rehearsal-test-pg"))

    assert len(runner.calls) == 100
    probe = runner.calls[0]
    assert probe[:3] == ["docker", "exec", "gw-rehearsal-test-pg"]
    command = " ".join(probe)
    assert "127.0.0.1" in command
    assert "gods_watching" in command
    assert "pg_isready" not in command
    assert 'PGPASSWORD="$POSTGRES_PASSWORD"' in command


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


def test_empty_target_directory_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    (inputs.assets / "clip" / "weights.bin").unlink()
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="target_package_unavailable"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_wrong_target_proof_revision_refused(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    manifest = inputs.assets / "prepared-manifest.json"
    raw = json.loads(manifest.read_text())
    raw["model_proofs"][0]["revision"] = "wrong"
    manifest.write_text(json.dumps(raw))
    runner = FakeDocker()
    with pytest.raises(RehearsalStackError, match="target_package_unavailable"):
        asyncio.run(_enter(inputs, runner))
    assert runner.calls == []


def test_gateway_mode_must_be_isolated(tmp_path: Path) -> None:
    runner = FakeDocker(gateway_mode="nat")
    with pytest.raises(RehearsalStackError, match="network_isolation_unverified"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_ipv6_must_be_disabled(tmp_path: Path) -> None:
    runner = FakeDocker(ipv6=True)
    with pytest.raises(RehearsalStackError, match="network_isolation_unverified"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_cleanup_failure_is_reported_after_all_attempts(tmp_path: Path) -> None:
    runner = FakeDocker(fail_cleanup=True)
    with pytest.raises(RehearsalStackError, match="cleanup_failed"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_cancellation_cleans_resources(tmp_path: Path) -> None:
    runner = FakeDocker()
    inputs = _inputs(tmp_path)

    async def cancel_inside() -> None:
        async with isolated_rehearsal_stack(inputs, runner=runner):
            asyncio.current_task().cancel()
            await asyncio.sleep(0)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_inside())
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_host_gateway_refused(tmp_path: Path) -> None:
    runner = FakeDocker(host_gateway="172.20.0.1")
    with pytest.raises(RehearsalStackError, match="network_isolation_unverified"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert runner.calls[-1][1:3] == ["network", "rm"]


def test_triton_image_mismatch_refused_before_provisioning(tmp_path: Path) -> None:
    runner = FakeDocker(image_id="sha256:wrong")
    with pytest.raises(RehearsalStackError, match="triton_image_mismatch"):
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert not any(call[:3] == ["docker", "network", "create"] for call in runner.calls)


def test_failed_network_create_preserves_original_error(tmp_path: Path) -> None:
    runner = FakeDocker(fail_network_create=True)
    with pytest.raises(DockerCommandError, match="docker_command_failed") as captured:
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert captured.value.stderr == "network creation failed"
    assert runner.created == set()


def test_failed_postgres_start_preserves_original_error(tmp_path: Path) -> None:
    runner = FakeDocker(fail_pg_run=True)
    with pytest.raises(DockerCommandError, match="docker_command_failed") as captured:
        asyncio.run(_enter(_inputs(tmp_path), runner))
    assert captured.value.stderr == "PostgreSQL start failed"
    assert runner.created == set()
