"""Disposable, network-isolated rehearsal resources from offline inputs."""

from __future__ import annotations

# ruff: noqa: EM101, PLR0913
import asyncio
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio

from gods_watching.storage.crops import CropObjectStore, CropPathError

from .registry import load_clip_registry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

PG_IMAGE = "pgvector/pgvector:0.8.1-pg17"
TRITON_IMAGE = "gods-watching-triton:25.02"
GPU_PATTERN = re.compile(r"GPU-[0-9a-fA-F-]{8,}")


class RehearsalStackError(ValueError):
    """A stable refusal code for an unverified rehearsal resource."""


@dataclass(slots=True)
class RehearsalInputs:
    """Explicit offline inputs; none defaults to a production resource."""

    database_dump: Path
    crop_snapshot: Path
    assets: Path
    gpu_uuid: str
    target_model_id: str


@dataclass(frozen=True, slots=True)
class RehearsalStack:
    """Verified network-local handles required by the containerized runner."""

    database_url: str
    crops_root: Path
    triton_url: str
    gpu_uuid: str
    postgres_container_id: str
    triton_container_id: str
    network_name: str
    database_system_id: str
    database_oid: str


class CommandRunner(Protocol):
    """Injectable subprocess boundary for Docker operations."""

    async def run(self, argv: Sequence[str], *, check: bool = True) -> str:
        """Return stdout or raise on a failed command."""


class DockerRunner:
    """Run Docker without a shell or inherited production connection URL."""

    async def run(self, argv: Sequence[str], *, check: bool = True) -> str:
        """Run one bounded command in a worker thread."""

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(  # noqa: S603
                list(argv),
                capture_output=True,
                check=False,
                text=True,
                timeout=3600 if "pg_restore" in argv else 120,
            )

        task = asyncio.create_task(asyncio.to_thread(execute))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            # Wait for any in-flight Docker create/remove to finish before teardown.
            await task
            raise
        if check and result.returncode:
            raise RehearsalStackError("docker_command_failed")
        return result.stdout.strip()


def _directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and os.access(path, os.R_OK | os.X_OK)
    except OSError:
        return False


def validate_rehearsal_inputs(inputs: RehearsalInputs) -> None:  # noqa: C901, PLR0912
    """Reject absent, aliased, or inconsistent inputs before Docker access."""
    try:
        dump_mode = inputs.database_dump.lstat().st_mode
    except OSError as error:
        raise RehearsalStackError("database_dump_unavailable") from error
    if not stat.S_ISREG(dump_mode) or not os.access(inputs.database_dump, os.R_OK):
        raise RehearsalStackError("database_dump_unavailable")
    if not _directory(inputs.crop_snapshot):
        raise RehearsalStackError("crop_snapshot_unavailable")
    if not _directory(inputs.assets):
        raise RehearsalStackError("assets_unavailable")
    if any("," in str(path.resolve()) for path in (inputs.database_dump, inputs.assets)):
        raise RehearsalStackError("unsupported_bind_path")
    live_crops = os.environ.get("GW_CROPS_ROOT")
    if live_crops and inputs.crop_snapshot.resolve() == Path(live_crops).resolve():
        raise RehearsalStackError("live_crop_alias")
    live_assets = os.environ.get("GW_MODEL_ASSETS_ROOT")
    if live_assets and inputs.assets.resolve() == Path(live_assets).resolve():
        raise RehearsalStackError("live_assets_alias")
    if not GPU_PATTERN.fullmatch(inputs.gpu_uuid) or not inputs.target_model_id.strip():
        raise RehearsalStackError("invalid_rehearsal_identity")
    manifest = inputs.assets / "prepared-manifest.json"
    try:
        if not stat.S_ISREG(manifest.lstat().st_mode):
            raise RehearsalStackError("preparation_manifest_unavailable")
        prepared = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RehearsalStackError("preparation_manifest_unavailable") from error
    if not isinstance(prepared, dict) or prepared.get("cuda_device_uuid") != inputs.gpu_uuid:
        raise RehearsalStackError("gpu_uuid_mismatch")
    if prepared.get("detector_resident") is not True:
        raise RehearsalStackError("detector_preparation_unverified")
    package = load_clip_registry(inputs.assets).get(inputs.target_model_id)
    if package is None:
        raise RehearsalStackError("target_package_unavailable")
    relative = package.snapshot_path.relative_to("/models")
    if not _directory(inputs.assets / relative):
        raise RehearsalStackError("target_package_unavailable")


def _copy_crops(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RehearsalStackError("unsafe_crop_snapshot")
    shutil.copytree(source, target)


async def _wait_postgres(runner: CommandRunner, name: str) -> None:
    for _ in range(100):
        try:
            await runner.run(
                ["docker", "exec", name, "pg_isready", "-U", "postgres", "-d", "gods_watching"]
            )
            return  # noqa: TRY300
        except RehearsalStackError:
            await asyncio.sleep(0.1)
    raise RehearsalStackError("postgres_not_ready")


async def _wait_triton(runner: CommandRunner, name: str) -> None:
    for _ in range(300):
        try:
            await runner.run(
                [
                    "docker",
                    "exec",
                    name,
                    "curl",
                    "--fail",
                    "--silent",
                    "http://127.0.0.1:8000/v2/models/detector/ready",
                ]
            )
            return  # noqa: TRY300
        except RehearsalStackError:
            await asyncio.sleep(0.2)
    raise RehearsalStackError("detector_not_ready")


async def _verify_docker_isolation(
    runner: CommandRunner, network: str, pg_name: str, triton_name: str
) -> None:
    """Verify Docker actually created the requested internal, unpublished topology."""
    details = await runner.run(
        ["docker", "network", "inspect", "--format", "{{json .Internal}}", network]
    )
    if details != "true":
        raise RehearsalStackError("network_isolation_unverified")
    for name in (pg_name, triton_name):
        raw = await runner.run(["docker", "inspect", "--format", "{{json .HostConfig}}", name])
        try:
            host = json.loads(raw)
        except json.JSONDecodeError as error:
            raise RehearsalStackError("container_isolation_unverified") from error
        if (
            not isinstance(host, dict)
            or host.get("NetworkMode") != network
            or host.get("PortBindings") not in (None, {})
        ):
            raise RehearsalStackError("container_isolation_unverified")


async def _verified_stack(
    inputs: RehearsalInputs,
    runner: CommandRunner,
    *,
    network: str,
    pg_name: str,
    triton_name: str,
    crops: Path,
    password: str,
) -> RehearsalStack:
    await _wait_postgres(runner, pg_name)
    await runner.run(
        [
            "docker",
            "exec",
            pg_name,
            "pg_restore",
            "--no-owner",
            "--no-acl",
            "--exit-on-error",
            "-U",
            "postgres",
            "-d",
            "gods_watching",
            "/backup/input.dump",
        ]
    )
    identity = await runner.run(
        [
            "docker",
            "exec",
            pg_name,
            "psql",
            "-U",
            "postgres",
            "-d",
            "gods_watching",
            "-At",
            "-F",
            "|",
            "-c",
            (
                "SELECT system_identifier, (SELECT oid FROM pg_database "
                "WHERE datname = 'gods_watching') FROM pg_control_system()"
            ),
        ]
    )
    parts = identity.split("|")
    if len(parts) != len(("system_id", "oid")) or not all(part.isdigit() for part in parts):
        raise RehearsalStackError("clone_identity_unverified")
    keys = await runner.run(
        [
            "docker",
            "exec",
            pg_name,
            "psql",
            "-U",
            "postgres",
            "-d",
            "gods_watching",
            "-At",
            "-c",
            "SELECT crop_object_key FROM appearances",
        ]
    )
    crop_store = CropObjectStore(crops)
    missing: list[str] = []
    for key in keys.splitlines():
        try:
            if not crop_store.read(key):
                missing.append(key)
        except (OSError, CropPathError):
            missing.append(key)
    if missing:
        code = f"missing_crops:{','.join(missing)}"
        raise RehearsalStackError(code)
    await _verify_docker_isolation(runner, network, pg_name, triton_name)
    pg_id = await runner.run(["docker", "inspect", "--format", "{{.Id}}", pg_name])
    triton_id = await runner.run(["docker", "inspect", "--format", "{{.Id}}", triton_name])
    if not pg_id or not triton_id or pg_id == triton_id:
        raise RehearsalStackError("container_identity_unverified")
    gpu = await runner.run(
        ["docker", "exec", triton_name, "nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]
    )
    if gpu.splitlines() != [inputs.gpu_uuid]:
        raise RehearsalStackError("gpu_uuid_mismatch")
    await _wait_triton(runner, triton_name)
    return RehearsalStack(
        f"postgresql+asyncpg://postgres:{password}@{pg_name}:5432/gods_watching",
        crops,
        f"{triton_name}:8001",
        inputs.gpu_uuid,
        pg_id,
        triton_id,
        network,
        parts[0],
        parts[1],
    )


@asynccontextmanager
async def isolated_rehearsal_stack(
    inputs: RehearsalInputs,
    *,
    runner: CommandRunner | None = None,
) -> AsyncIterator[RehearsalStack]:
    """Yield a verified disposable stack and always remove its Docker resources."""
    validate_rehearsal_inputs(inputs)
    command = runner or DockerRunner()
    suffix = secrets.token_hex(8)
    network = f"gw-rehearsal-{suffix}"
    pg_name = f"{network}-pg"
    triton_name = f"{network}-triton"
    password = secrets.token_hex(24)
    with tempfile.TemporaryDirectory(prefix="gw-rehearsal-") as temporary:
        crops = Path(temporary) / "crops"
        _copy_crops(inputs.crop_snapshot, crops)
        try:
            await command.run(
                ["docker", "network", "create", "--internal", "--driver", "bridge", network]
            )
            await command.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--pull=never",
                    "--name",
                    pg_name,
                    "--network",
                    network,
                    "--mount",
                    f"type=bind,src={inputs.database_dump.resolve()},dst=/backup/input.dump,readonly",
                    "--env",
                    f"POSTGRES_PASSWORD={password}",
                    "--env",
                    "POSTGRES_DB=gods_watching",
                    PG_IMAGE,
                ]
            )
            await command.run(
                [
                    "docker",
                    "run",
                    "--detach",
                    "--pull=never",
                    "--name",
                    triton_name,
                    "--network",
                    network,
                    "--gpus",
                    f"device={inputs.gpu_uuid}",
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,nosuid,size=1073741824",  # noqa: S108
                    "--mount",
                    f"type=bind,src={inputs.assets.resolve()},dst=/models,readonly",
                    TRITON_IMAGE,
                    "tritonserver",
                    "--model-repository=/models-repository",
                    "--strict-readiness=true",
                    "--model-control-mode=explicit",
                    "--load-model=detector",
                ]
            )
            stack = await _verified_stack(
                inputs,
                command,
                network=network,
                pg_name=pg_name,
                triton_name=triton_name,
                crops=crops,
                password=password,
            )
            yield stack
        finally:
            with anyio.CancelScope(shield=True):
                for resource in (
                    ["docker", "rm", "--force", "--volumes", triton_name],
                    ["docker", "rm", "--force", "--volumes", pg_name],
                    ["docker", "network", "rm", network],
                ):
                    with suppress(OSError, RehearsalStackError, subprocess.TimeoutExpired):
                        await command.run(resource, check=False)
