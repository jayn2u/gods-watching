"""Disposable, network-isolated rehearsal resources from offline inputs."""

from __future__ import annotations

# ruff: noqa: EM101, PLR0913
import asyncio
import hashlib
import json
import os
import re
import secrets
import shutil
import stat
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import anyio

from gods_watching.setup.model_preparation import GpuProof, PreparedManifest, validate_gpu_proof
from gods_watching.storage.crops import CropObjectStore, CropPathError

from .assets import PreparedModelCatalog
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
    model_lock: Path | None = None


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


class DockerCommandError(RehearsalStackError):
    """Preserve Docker stderr privately for precise cleanup classification."""

    def __init__(self, *, stderr: str, returncode: int) -> None:
        """Keep details for cleanup without exposing Docker stderr as public text."""
        self.stderr = stderr.strip()
        self.returncode = returncode
        super().__init__("docker_command_failed")


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
                timeout=3600 if "pg_restore" in argv or "rehearse-inner" in argv else 120,
            )

        task = asyncio.create_task(asyncio.to_thread(execute))
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            # Wait for any in-flight Docker create/remove to finish before teardown.
            await task
            raise
        _ = check
        if result.returncode:
            raise DockerCommandError(stderr=result.stderr, returncode=result.returncode)
        return result.stdout.strip()


def _directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and os.access(path, os.R_OK | os.X_OK)
    except OSError:
        return False


def validate_rehearsal_inputs(inputs: RehearsalInputs) -> PreparedManifest:  # noqa: C901, PLR0912
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
        prepared = PreparedManifest.model_validate_json(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RehearsalStackError("preparation_manifest_unavailable") from error
    if prepared.cuda_device_uuid != inputs.gpu_uuid:
        raise RehearsalStackError("gpu_uuid_mismatch")
    if not prepared.detector_resident:
        raise RehearsalStackError("detector_preparation_unverified")
    lock_path = inputs.model_lock or Path(__file__).resolve().parents[4] / "assets/models.lock.json"
    try:
        registry = load_clip_registry(inputs.assets)
        package = registry.get(inputs.target_model_id)
        if (
            package is None
            or not PreparedModelCatalog(registry, lock_path, inputs.assets).status(package).prepared
        ):
            raise RehearsalStackError("target_package_unavailable")
        matching = tuple(
            proof for proof in prepared.model_proofs if proof.model_id == package.model_id
        )
        if (
            len(matching) != 1
            or prepared.lock_sha256 != hashlib.sha256(lock_path.read_bytes()).hexdigest()
            or not prepared.image_id
        ):
            raise RehearsalStackError("target_package_unavailable")
        proof = GpuProof(
            python_abi=prepared.python_abi,
            torch_version=prepared.torch_version,
            torchvision_version=prepared.torchvision_version,
            cuda_version=prepared.cuda_version,
            cuda_device=prepared.cuda_device,
            cuda_device_uuid=prepared.cuda_device_uuid,
            cuda_available=prepared.cuda_available,
            cuda_operation=prepared.cuda_operation,
            processor=prepared.processor,
            clip_class=prepared.clip_class,
            yolo_class=prepared.yolo_class,
            detector_resident=prepared.detector_resident,
            models=matching,
        )
        validate_gpu_proof(proof, (package,))
    except (OSError, UnicodeError, ValueError, RuntimeError) as error:
        raise RehearsalStackError("target_package_unavailable") from error
    return prepared


def _copy_crops(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise RehearsalStackError("unsafe_crop_snapshot")
    shutil.copytree(source, target)


async def _wait_postgres(runner: CommandRunner, name: str) -> None:
    for _ in range(100):
        ready = False
        try:
            result = await runner.run(
                [
                    "docker",
                    "exec",
                    name,
                    "sh",
                    "-c",
                    (
                        'PGPASSWORD="$POSTGRES_PASSWORD" psql -h 127.0.0.1 '
                        '-U postgres -d gods_watching -Atc "SELECT 1"'
                    ),
                ]
            )
            ready = result.strip() == "1"
        except RehearsalStackError:
            pass
        if ready:
            return
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


async def verify_docker_isolation(
    runner: CommandRunner, network: str, pg_name: str, triton_name: str
) -> None:
    """Verify Docker actually created the requested internal, unpublished topology."""
    raw_network = await runner.run(
        ["docker", "network", "inspect", "--format", "{{json .}}", network]
    )
    try:
        details = json.loads(raw_network)
    except json.JSONDecodeError as error:
        raise RehearsalStackError("network_isolation_unverified") from error
    if (
        not isinstance(details, dict)
        or details.get("Internal") is not True
        or details.get("EnableIPv6") is not False
        or not isinstance(details.get("Options"), dict)
        or details["Options"].get("com.docker.network.bridge.gateway_mode_ipv4") != "isolated"
        or not isinstance(details.get("IPAM"), dict)
        or not isinstance(details["IPAM"].get("Config"), list)
        or any(
            not isinstance(config, dict) or config.get("Gateway")
            for config in details["IPAM"]["Config"]
        )
    ):
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
            "SELECT crop_object_key FROM appearances WHERE tombstoned_at IS NULL",
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
    await verify_docker_isolation(runner, network, pg_name, triton_name)
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


def _already_absent(resource: Sequence[str], error: DockerCommandError) -> bool:
    """Only Docker's explicit missing response for this resource is idempotent."""
    lines = error.stderr.lower().splitlines()
    if not lines or any(re.fullmatch(r"exit status [0-9]+", line) is None for line in lines[1:]):
        return False
    message = lines[0].removeprefix("error response from daemon: ")
    name = resource[-1].lower()
    if resource[1:2] == ["rm"]:
        return message == f"no such container: {name}"
    if resource[1:3] == ["network", "rm"]:
        return message in (f"no such network: {name}", f"network {name} not found")
    return False


@asynccontextmanager
async def isolated_rehearsal_stack(
    inputs: RehearsalInputs,
    *,
    runner: CommandRunner | None = None,
) -> AsyncIterator[RehearsalStack]:
    """Yield a verified disposable stack and always remove its Docker resources."""
    prepared = validate_rehearsal_inputs(inputs)
    command = runner or DockerRunner()
    observed_image = await command.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", TRITON_IMAGE]
    )
    if observed_image != prepared.image_id:
        raise RehearsalStackError("triton_image_mismatch")
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
                [
                    "docker",
                    "network",
                    "create",
                    "--internal",
                    "--ipv6=false",
                    "--driver",
                    "bridge",
                    "--opt",
                    "com.docker.network.bridge.gateway_mode_ipv4=isolated",
                    network,
                ]
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
                failures: list[Exception] = []
                for resource in (
                    ["docker", "rm", "--force", "--volumes", triton_name],
                    ["docker", "rm", "--force", "--volumes", pg_name],
                    ["docker", "network", "rm", network],
                ):
                    try:
                        await command.run(resource)
                    except DockerCommandError as error:
                        if not _already_absent(resource, error):
                            failures.append(error)
                    except (OSError, RehearsalStackError, subprocess.TimeoutExpired) as error:
                        failures.append(error)
                if failures:
                    raise RehearsalStackError("cleanup_failed") from failures[0]
