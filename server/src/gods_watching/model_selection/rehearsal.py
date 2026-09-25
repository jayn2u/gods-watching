"""Run one model switch against a disposable offline clone and publish measured proof."""

from __future__ import annotations

# ruff: noqa: EM101, PLR0913, PLR0915, PLR0912, SIM117, S108, PLC0415, PLR2004, C901
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast
from urllib.parse import urlsplit

import anyio
from cryptography.fernet import Fernet
from pydantic import BaseModel, ConfigDict, ValidationError
from sqlalchemy import func, select

from gods_watching.model_selection.preflight import (
    MIN_SAMPLE_COUNT,
    rehearsal_path,
    retained_corpus_sha256,
    runtime_code_sha256,
    scan_retained,
)
from gods_watching.model_selection.registry import ClipModelPackage, load_clip_registry
from gods_watching.model_selection.rehearsal_stack import (
    CommandRunner,
    DockerRunner,
    RehearsalInputs,
    isolated_rehearsal_stack,
    validate_rehearsal_inputs,
    verify_docker_isolation,
)
from gods_watching.model_selection.transition_observer import PHASES, TransitionObserver

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.model_selection.repository import TransitionRepository
    from gods_watching.model_selection.transition_observer import TransitionMeasurement
    from gods_watching.storage import CropObjectStore


class RehearsalError(ValueError):
    """A stable reason why this attempt cannot authorize a switch."""


@dataclass(frozen=True, slots=True)
class InnerInputs:
    """Network-local resources supplied only by the disposable stack."""

    database_url: str
    triton_url: str
    crops_root: Path
    assets_root: Path
    model_lock: Path
    target_model_id: str
    source_model_id: str
    source_revision: str
    source_dimension: int
    output: Path
    camera_cipher_key_file: Path


def _require_package(package: ClipModelPackage | None, model_id: str) -> ClipModelPackage:
    if package is None or package.model_id != model_id:
        raise RehearsalError("model_identity_unavailable")
    return package


def _measurement_payload(
    observation: TransitionMeasurement | None,
    *,
    source: ClipModelPackage,
    target: ClipModelPackage,
    job_id: str,
) -> dict[str, object]:
    """Accept only an entirely fresh, finite, committed production observation."""
    if observation is None or not observation.complete:
        raise RehearsalError("observation_incomplete")
    if observation.source_identity != (source.model_id, source.revision, source.dimension):
        raise RehearsalError("source_observation_mismatch")
    if observation.target_identity != (target.model_id, target.revision, target.dimension):
        raise RehearsalError("target_observation_mismatch")
    if (
        observation.completed_crops < MIN_SAMPLE_COUNT
        or len(observation.crop_seconds) != observation.completed_crops
    ):
        raise RehearsalError("insufficient_committed_crops")
    if set(observation.phase_seconds) != set(PHASES) or set(observation.phase_spans) != set(PHASES):
        raise RehearsalError("phase_incomplete")
    for phase in PHASES:
        elapsed = observation.phase_seconds[phase]
        start, end = observation.phase_spans[phase]
        if (
            not all(math.isfinite(value) for value in (elapsed, start, end))
            or elapsed <= 0
            or end <= start
            or not math.isclose(elapsed, end - start, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise RehearsalError("phase_invalid")
    if any(not math.isfinite(value) or value <= 0 for value in observation.crop_seconds):
        raise RehearsalError("crop_time_invalid")
    fixed = observation.measured_fixed_seconds
    if fixed is None or not math.isfinite(fixed) or fixed <= 0:
        raise RehearsalError("fixed_time_invalid")
    return {
        "job_id": job_id,
        "source_identity": list(observation.source_identity),
        "target_identity": list(observation.target_identity),
        "phase_seconds": observation.phase_seconds,
        "phase_spans": {key: list(value) for key, value in observation.phase_spans.items()},
        "crop_seconds": list(observation.crop_seconds),
        "sample_count": observation.completed_crops,
        "measured_seconds": observation.phase_seconds["crop_embedding"],
        "measured_fixed_seconds": fixed,
    }


async def _queue_verified_job(
    repository: TransitionRepository,
    session: AsyncSession,
    crop_store: CropObjectStore,
    *,
    source: ClipModelPackage,
    target: ClipModelPackage,
) -> str:
    """Refuse ambiguous clone state before creating one fresh isolated job."""
    active = await repository.active_identity(session)
    if (active.model_id, active.model_revision, active.embedding_dimension) != (
        source.model_id,
        source.revision,
        source.dimension,
    ):
        raise RehearsalError("source_database_mismatch")
    if await repository.active_job(session) is not None:
        raise RehearsalError("existing_transition_job")
    retained, missing = await scan_retained(session, crop_store)
    if retained < MIN_SAMPLE_COUNT or missing:
        raise RehearsalError("insufficient_decodable_crops")
    from gods_watching.storage.models import Appearance

    unique = await session.scalar(
        select(func.count(func.distinct(Appearance.crop_object_key))).where(
            Appearance.tombstoned_at.is_(None)
        )
    )
    if unique is None or unique < MIN_SAMPLE_COUNT:
        raise RehearsalError("insufficient_distinct_crops")
    job = await repository.create_job(session, target=target, default=source)
    return str(job.id)


async def run_inner(inputs: InnerInputs) -> dict[str, object]:
    """Queue and run the production one-shot worker inside the isolated network."""
    from gods_watching.inference.clip import ClipRuntimeManager
    from gods_watching.inference.detector import DetectorClient, TritonGrpcDetectorTransport
    from gods_watching.model_selection.assets import PreparedModelCatalog
    from gods_watching.model_selection.coordinator import TransitionCoordinator
    from gods_watching.model_selection.repository import TransitionRepository
    from gods_watching.pipeline_worker.app import compose_one_shot_transition
    from gods_watching.pipeline_worker.settings import PipelineWorkerSettings
    from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

    database_host = urlsplit(inputs.database_url).hostname or ""
    triton_host = inputs.triton_url.split(":", 1)[0]
    if (
        not database_host.startswith("gw-rehearsal-")
        or not database_host.endswith("-pg")
        or triton_host != database_host.removesuffix("-pg") + "-triton"
        or inputs.crops_root != Path("/rehearsal/crops")
        or inputs.assets_root != Path("/models")
        or inputs.output != Path("/rehearsal/proof/observation.json")
    ):
        raise RehearsalError("inner_stack_handles_required")
    registry = load_clip_registry(inputs.assets_root)
    target = _require_package(registry.get(inputs.target_model_id), inputs.target_model_id)
    source = _require_package(registry.get(inputs.source_model_id), inputs.source_model_id)
    if (source.revision, source.dimension) != (inputs.source_revision, inputs.source_dimension):
        raise RehearsalError("source_package_mismatch")
    if target.model_id == source.model_id and target.revision == source.revision:
        raise RehearsalError("target_equals_source")
    prepared = PreparedModelCatalog(registry, inputs.model_lock, inputs.assets_root)
    if not prepared.status(target).prepared:
        raise RehearsalError("target_unprepared")
    settings = PipelineWorkerSettings(
        database_url=inputs.database_url,
        triton_grpc_url=inputs.triton_url,
        crops_root=inputs.crops_root,
        camera_cipher_key=inputs.camera_cipher_key_file.read_text(encoding="ascii").strip(),
        model_lock_path=inputs.model_lock,
        model_assets_root=inputs.assets_root,
    )
    _ = Fernet(settings.camera_cipher_key.encode())
    database = Database.connect(inputs.database_url)
    crop_store = CropObjectStore(inputs.crops_root)
    detector_transport = TritonGrpcDetectorTransport(url=inputs.triton_url)
    detector = DetectorClient(transport=detector_transport)
    coordinator = TransitionCoordinator(database)
    repository = TransitionRepository()
    runtime = ClipRuntimeManager(inputs.triton_url)
    payload: dict[str, object] | None = None
    corpus_sha256: str | None = None
    try:
        async with coordinator.worker_ownership():
            async with database.transaction() as session:
                corpus_sha256 = await retained_corpus_sha256(session, crop_store)
                if corpus_sha256 is None:
                    raise RehearsalError("retained_corpus_unavailable")
                job_id = await _queue_verified_job(
                    repository, session, crop_store, source=source, target=target
                )
            async with runtime:
                async with anyio.create_task_group() as task_group:
                    composition = compose_one_shot_transition(
                        settings=settings,
                        database=database,
                        storage=StorageRepository(
                            CredentialCipher(settings.camera_cipher_key.encode())
                        ),
                        crop_store=crop_store,
                        detector_transport=detector_transport,
                        detector=detector,
                        registry=registry,
                        prepared=prepared,
                        coordinator=coordinator,
                        runtime=runtime,
                        task_group=task_group,
                        quality_policy_path=Path(
                            "/opt/gods-watching/assets/retrieval-quality-policy.json"
                        ),
                    )
                    observer = TransitionObserver()

                    async def probe_restart() -> None:
                        await anyio.sleep(0.25)
                        generation = composition.lifecycle.generation
                        if (
                            generation is None
                            or generation.package.model_id != target.model_id
                            or generation.package.revision != target.revision
                            or generation.done_event.is_set()
                        ):
                            raise RehearsalError("pipeline_restart_failed")

                    result = await composition.run_pending(
                        observer=observer, restart_probe=probe_restart
                    )
                    if result is None or not result.activated or str(result.state.id) != job_id:
                        raise RehearsalError("transition_not_activated")
                    payload = _measurement_payload(
                        observer.measurement, source=source, target=target, job_id=job_id
                    )
                    async with database.transaction() as session:
                        active = await repository.active_identity(session)
                        if (active.model_id, active.model_revision, active.embedding_dimension) != (
                            target.model_id,
                            target.revision,
                            target.dimension,
                        ):
                            raise RehearsalError("target_database_mismatch")
                    identity = await runtime.inspect_identity()
                    if (identity.model_id, identity.revision, identity.dimension) != (
                        target.model_id,
                        target.revision,
                        target.dimension,
                    ):
                        raise RehearsalError("target_triton_mismatch")
        if payload is None:
            raise RehearsalError("observation_missing")
        code_sha256 = runtime_code_sha256()
        if corpus_sha256 is None or code_sha256 is None:
            raise RehearsalError("proof_binding_unavailable")
        payload["retained_corpus_sha256"] = corpus_sha256
        payload["runtime_code_sha256"] = code_sha256
        _ = inputs.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return payload
    finally:
        await detector_transport.close()
        await database.close()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(bytes.fromhex(_hash_file(item)))
    return digest.hexdigest()


class InnerEvidence(BaseModel):
    """Parse the inner process's event evidence without accepting success flags."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)

    job_id: str
    source_identity: tuple[str, str, int]
    target_identity: tuple[str, str, int]
    phase_seconds: dict[str, float]
    phase_spans: dict[str, tuple[float, float]]
    crop_seconds: tuple[float, ...]
    sample_count: int
    measured_seconds: float
    measured_fixed_seconds: float
    retained_corpus_sha256: str
    runtime_code_sha256: str


def _verified_inner_payload(
    raw: object, source: ClipModelPackage, target: ClipModelPackage
) -> dict[str, object]:
    """Revalidate all inner events and independently derive accepted timing."""
    try:
        evidence = InnerEvidence.model_validate(raw)
    except ValidationError as error:
        raise RehearsalError("inner_observation_invalid") from error
    if evidence.source_identity != (
        source.model_id,
        source.revision,
        source.dimension,
    ) or evidence.target_identity != (target.model_id, target.revision, target.dimension):
        raise RehearsalError("inner_identity_mismatch")
    from gods_watching.model_selection.transition_observer import TransitionMeasurement

    measurement = TransitionMeasurement(
        source_identity=evidence.source_identity,
        target_identity=evidence.target_identity,
        phase_seconds=evidence.phase_seconds,
        phase_spans=evidence.phase_spans,
        crop_seconds=evidence.crop_seconds,
        completed_crops=evidence.sample_count,
        complete=True,
    )
    verified = _measurement_payload(
        measurement, source=source, target=target, job_id=evidence.job_id
    )
    if (
        evidence.measured_seconds != verified["measured_seconds"]
        or evidence.measured_fixed_seconds != verified["measured_fixed_seconds"]
    ):
        raise RehearsalError("inner_timing_mismatch")
    for digest in (evidence.retained_corpus_sha256, evidence.runtime_code_sha256):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RehearsalError("inner_binding_invalid")
    verified["retained_corpus_sha256"] = evidence.retained_corpus_sha256
    verified["runtime_code_sha256"] = evidence.runtime_code_sha256
    return verified


def _invalidate(path: Path) -> None:
    """Durably remove an earlier accepted proof before this attempt starts."""
    path.unlink(missing_ok=True)
    if path.parent.is_dir():
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _ = os.fsync(directory)
        finally:
            os.close(directory)


def _publish(path: Path, record: dict[str, object]) -> None:
    """Publish durably, removing an accepted path if post-rename sync fails."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    published = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=".rehearsal-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, separators=(",", ":"), allow_nan=False)
            stream.flush()
            _ = os.fsync(stream.fileno())
        _ = temporary.replace(path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _ = os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if published:
            try:
                path.unlink(missing_ok=True)
            finally:
                # Preserve the original publication failure; make removal durable
                # when the filesystem can still sync the directory.
                try:
                    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                    try:
                        _ = os.fsync(directory)
                    finally:
                        os.close(directory)
                except OSError:
                    pass
        raise
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


async def rehearse_switch(
    inputs: RehearsalInputs,
    *,
    source_model_id: str,
    source_revision: str,
    source_dimension: int,
    app_image: str,
    camera_cipher_key_file: Path,
    runner: CommandRunner | None = None,
) -> Path:
    """Keep the clone alive through the inner run and publish only verified evidence."""
    registry = load_clip_registry(inputs.assets)
    target = _require_package(registry.get(inputs.target_model_id), inputs.target_model_id)
    output = rehearsal_path(inputs.assets, target)
    _invalidate(output)
    prepared = validate_rehearsal_inputs(inputs)
    source = _require_package(registry.get(source_model_id), source_model_id)
    if (source.revision, source.dimension) != (source_revision, source_dimension):
        raise RehearsalError("source_package_mismatch")
    if not app_image.strip():
        raise RehearsalError("app_image_required")
    if not camera_cipher_key_file.is_file() or camera_cipher_key_file.is_symlink():
        raise RehearsalError("camera_cipher_key_unavailable")
    if inputs.model_lock is None:
        raise RehearsalError("model_lock_required")
    model_lock = inputs.model_lock
    command = runner or DockerRunner()
    app_image_id = await command.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", app_image]
    )
    dump_sha256 = _hash_file(inputs.database_dump)
    crop_sha256 = _hash_tree(inputs.crop_snapshot)
    async with isolated_rehearsal_stack(inputs, runner=command) as stack:
        with tempfile.TemporaryDirectory(prefix="gw-rehearsal-proof-") as scratch:
            proof = Path(scratch) / "observation.json"
            cidfile = Path(scratch) / "container-id"
            container_name = f"{stack.network_name}-worker"
            _ = await command.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--pull=never",
                    "--name",
                    container_name,
                    "--cidfile",
                    str(cidfile),
                    "--network",
                    stack.network_name,
                    "--network-alias",
                    "rehearsal-worker",
                    "--read-only",
                    "--tmpfs",
                    "/tmp:rw,nosuid,size=1073741824",
                    "--mount",
                    f"type=bind,src={stack.crops_root.resolve()},dst=/rehearsal/crops,readonly",
                    "--mount",
                    f"type=bind,src={inputs.assets.resolve()},dst=/models,readonly",
                    "--mount",
                    f"type=bind,src={model_lock.resolve()},dst=/rehearsal/models.lock.json,readonly",
                    "--mount",
                    f"type=bind,src={camera_cipher_key_file.resolve()},dst=/rehearsal/camera.key,readonly",
                    "--mount",
                    f"type=bind,src={Path(scratch).resolve()},dst=/rehearsal/proof",
                    app_image_id,
                    "gods-watching-cli",
                    "models",
                    "rehearse-inner",
                    "--database-url",
                    stack.database_url,
                    "--triton-url",
                    stack.triton_url,
                    "--crops",
                    "/rehearsal/crops",
                    "--assets",
                    "/models",
                    "--lock",
                    "/rehearsal/models.lock.json",
                    "--target-model-id",
                    target.model_id,
                    "--source-model-id",
                    source.model_id,
                    "--source-revision",
                    source.revision,
                    "--source-dimension",
                    str(source.dimension),
                    "--output",
                    "/rehearsal/proof/observation.json",
                    "--camera-cipher-key-file",
                    "/rehearsal/camera.key",
                ]
            )
            if not cidfile.is_file() or cidfile.is_symlink():
                raise RehearsalError("inner_container_id_missing")
            container_id = cidfile.read_text(encoding="ascii").strip()
            if len(container_id) != 64 or any(
                character not in "0123456789abcdef" for character in container_id
            ):
                raise RehearsalError("inner_container_id_invalid")
            if not proof.is_file() or proof.is_symlink():
                raise RehearsalError("inner_observation_missing")
            try:
                raw = cast("object", json.loads(proof.read_text(encoding="utf-8")))
                measurement = _verified_inner_payload(raw, source, target)
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise RehearsalError("inner_observation_invalid") from error
            await verify_docker_isolation(
                command,
                stack.network_name,
                f"{stack.network_name}-pg",
                f"{stack.network_name}-triton",
            )
            pg_id = await command.run(
                ["docker", "inspect", "--format", "{{.Id}}", f"{stack.network_name}-pg"]
            )
            triton_id = await command.run(
                ["docker", "inspect", "--format", "{{.Id}}", f"{stack.network_name}-triton"]
            )
            if pg_id != stack.postgres_container_id or triton_id != stack.triton_container_id:
                raise RehearsalError("stack_identity_changed")
            gpu = await command.run(
                [
                    "docker",
                    "exec",
                    stack.triton_container_id,
                    "nvidia-smi",
                    "--query-gpu=uuid",
                    "--format=csv,noheader",
                ]
            )
            if (
                gpu.splitlines() != [inputs.gpu_uuid]
                or inputs.gpu_uuid != prepared.cuda_device_uuid
            ):
                raise RehearsalError("runtime_gpu_mismatch")
            _ = await command.run(
                [
                    "docker",
                    "exec",
                    stack.triton_container_id,
                    "curl",
                    "--fail",
                    "--silent",
                    "http://127.0.0.1:8000/v2/models/detector/ready",
                ]
            )
            if validate_rehearsal_inputs(inputs) != prepared:
                raise RehearsalError("prepared_assets_changed")
            if _hash_file(inputs.database_dump) != dump_sha256:
                raise RehearsalError("database_dump_changed")
            if _hash_tree(stack.crops_root) != crop_sha256:
                raise RehearsalError("crop_snapshot_changed")
            record = {
                "kind": "full_transition_rehearsal_v1",
                "model_id": target.model_id,
                "revision": target.revision,
                "dimension": target.dimension,
                "device": prepared.cuda_device,
                "device_uuid": inputs.gpu_uuid,
                "detector_resident": True,
                "triton_rpc_measured": True,
                "database_staging_measured": True,
                "activation_measured": True,
                "pipeline_restart_measured": True,
                "measured_at": datetime.now(UTC).isoformat(),
                "sample_count": measurement["sample_count"],
                "measured_seconds": measurement["measured_seconds"],
                "measured_fixed_seconds": measurement["measured_fixed_seconds"],
                "run_id": stack.network_name,
                "container_name": container_name,
                "container_id": container_id,
                "postgres_container_id": stack.postgres_container_id,
                "triton_container_id": stack.triton_container_id,
                "database_system_id": stack.database_system_id,
                "database_oid": stack.database_oid,
                "app_image_id": app_image_id,
                "triton_image_id": prepared.image_id,
                "retained_corpus_sha256": measurement["retained_corpus_sha256"],
                "runtime_code_sha256": measurement["runtime_code_sha256"],
                "database_dump_sha256": dump_sha256,
                "crop_snapshot_sha256": crop_sha256,
                "job_id": measurement["job_id"],
                "phase_seconds": measurement["phase_seconds"],
            }
    _publish(output, record)
    return output
