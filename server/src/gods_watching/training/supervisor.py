"""Training request leasing, GPU admission, and owned child process identity."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

from sqlalchemy import select

from gods_watching.contracts.training import (
    TrainingConfig,
    TrainingPreflightResponse,
)
from gods_watching.training.calibration import CalibrationRefusedError
from gods_watching.training.dataset import (
    DatasetManifest,
    DatasetValidationError,
    validate_cuhk,
)
from gods_watching.training.memory import (
    INFERENCE_RESERVE_FRACTION,
    MIN_INFERENCE_RESERVE_BYTES,
    GpuSnapshot,
    MemoryProfile,
    StaleGpuSnapshotError,
    UnsupportedMemoryProfileError,
    assess_admission,
    estimate_memory,
)
from gods_watching.training.models import (
    TrainingExecutionSlot,
    TrainingJob,
    TrainingPhase,
    TrainingRequest,
    TrainingRequestKind,
    TrainingRequestPhase,
)
from gods_watching.training.repository import (
    TrainingJobConflictError,
    TrainingPhaseTransitionError,
    TrainingRepository,
    TrainingRequestLeaseLostError,
)
from gods_watching.training.settings import TrainingDatasetNotConfiguredError, TrainingSettings
from gods_watching.training.telemetry import current_gpu_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.storage import Database
    from gods_watching.training.memory import MemoryEstimate
    from gods_watching.training.models import TrainingJob


class TrainingSupervisorError(RuntimeError):
    """Base class for supervisor ownership failures."""


class TrainingOrphanProcessError(TrainingJobConflictError):
    """A durable GPU slot still has a live or unverifiable prior child."""


class TrainingChildLaunchError(TrainingSupervisorError):
    """The gated training child could not be prepared or released."""


class ProcessIdentityStatus(StrEnum):
    """Evidence available when checking a persisted child PID/start-time pair."""

    ALIVE = "alive"
    GONE = "gone"
    REUSED = "reused"
    UNVERIFIABLE = "unverifiable"


_PROC_STAT_FIELDS_AFTER_COMM = 2


def process_start_time(pid: int, *, proc_root: Path | None = None) -> int:
    """Read Linux /proc start ticks for one PID without trusting PID alone."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("process PID must be a positive integer")
    root = proc_root if proc_root is not None else Path("/proc")
    fields = (root / str(pid) / "stat").read_text(encoding="ascii").rsplit(")", maxsplit=1)
    if len(fields) != _PROC_STAT_FIELDS_AFTER_COMM:
        raise ValueError("process stat record is malformed")
    values = fields[1].strip().split()
    start_index = 19
    if len(values) <= start_index:
        raise ValueError("process stat record lacks a start time")
    start_time = int(values[start_index])
    if start_time <= 0:
        raise ValueError("process start time is invalid")
    return start_time


def process_identity_matches(
    pid: int,
    expected_start_time: int,
    *,
    proc_root: Path | None = None,
) -> bool:
    """Return true only while the exact PID/start-time pair still exists."""
    return (
        process_identity_status(
            pid,
            expected_start_time,
            proc_root=proc_root,
        )
        == ProcessIdentityStatus.ALIVE
    )


def process_identity_status(
    pid: int,
    expected_start_time: int,
    *,
    proc_root: Path | None = None,
) -> ProcessIdentityStatus:
    """Classify identity evidence without treating unreadable state as process exit."""
    if (
        type(pid) is not int
        or pid <= 0
        or type(expected_start_time) is not int
        or expected_start_time <= 0
    ):
        return ProcessIdentityStatus.UNVERIFIABLE
    try:
        actual_start_time = process_start_time(pid, proc_root=proc_root)
    except (FileNotFoundError, ProcessLookupError):
        return ProcessIdentityStatus.GONE
    except (OSError, ValueError):
        return ProcessIdentityStatus.UNVERIFIABLE
    if actual_start_time == expected_start_time:
        return ProcessIdentityStatus.ALIVE
    return ProcessIdentityStatus.REUSED


class TrainingChild(Protocol):
    """A child that remains blocked until durable PID ownership is recorded."""

    @property
    def pid(self) -> int:
        """Return the operating-system process id."""
        ...

    @property
    def start_time(self) -> int:
        """Return the PID-reuse-resistant process start ticks."""
        ...

    def release(self) -> None:
        """Allow the worker to begin after its child identity is committed."""
        ...

    def abort(self) -> None:
        """Terminate an unreleased child after a failed ownership commit."""
        ...

    def poll(self) -> int | None:
        """Return an exit code only after the operating system confirms process exit."""
        ...


class TrainingChildLauncher(Protocol):
    """Prepare one fixed runner command behind an ownership gate."""

    def prepare(self, job: TrainingJob) -> TrainingChild:
        """Start a child that waits for its recorded owner generation."""
        ...


class _SubprocessTrainingChild:
    _process: subprocess.Popen[bytes]
    _log_stream: object
    _start_time: int

    def __init__(self, process: subprocess.Popen[bytes], log_stream: object) -> None:
        self._process = process
        self._log_stream = log_stream
        self._start_time = process_start_time(process.pid)

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def start_time(self) -> int:
        return self._start_time

    def release(self) -> None:
        if self._process.poll() is not None or self._process.stdin is None:
            raise TrainingChildLaunchError("training child exited before ownership release")
        try:
            _ = self._process.stdin.write(b"owned\n")
            _ = self._process.stdin.flush()
            _ = self._process.stdin.close()
        except OSError as error:
            raise TrainingChildLaunchError("training child ownership gate failed") from error
        self._close_log()

    def abort(self) -> None:
        if self._process.stdin is not None:
            _ = self._process.stdin.close()
        if self._process.poll() is None:
            _ = self._process.terminate()
            try:
                _ = self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                _ = self._process.kill()
                _ = self._process.wait(timeout=3)
        self._close_log()

    def poll(self) -> int | None:
        """Reap and report an exit only after ``Popen`` observes process termination."""
        return self._process.poll()

    def _close_log(self) -> None:
        close = getattr(self._log_stream, "close", None)
        if callable(close):
            _ = close()


class SubprocessTrainingChildLauncher:
    """Spawn only the fixed package runner under a closed stdin ownership gate."""

    _training_root: Path

    def __init__(self, training_root: Path) -> None:
        """Set the root directory for isolated worker state and logs."""
        self._training_root = training_root

    def prepare(self, job: TrainingJob) -> TrainingChild:
        """Create a gated fixed-runner child and return its durable identity."""
        job_root = self._training_root / "jobs" / str(job.id)
        job_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        log_path = job_root / "worker.log"
        log_stream = log_path.open("ab")
        command = (
            sys.executable,
            "-m",
            "gods_watching.training.runner",
            "--job-id",
            str(job.id),
            "--owner-generation",
            str(job.owner_generation),
            "--wait-for-owner",
        )
        environment = os.environ.copy()
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.PIPE,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
                env=environment,
            )
            if process.poll() is not None:
                raise TrainingChildLaunchError(  # noqa: TRY301
                    "training runner exited before owner fencing"
                )
            return _SubprocessTrainingChild(process, log_stream)
        except Exception:
            log_stream.close()
            raise


class TrainingSupervisor:
    """Lease requests, recheck memory, and create jobs only inside admission CAS."""

    _database: Database
    _settings: TrainingSettings
    _repository: TrainingRepository
    _child_launcher: TrainingChildLauncher | None
    _gpu_snapshot_provider: Callable[[], GpuSnapshot]
    _memory_profiles_provider: Callable[[], tuple[MemoryProfile, ...]]
    _dataset_validator: Callable[[Path], DatasetManifest]
    _worker_id: str
    _children: dict[UUID, tuple[TrainingChild, int]]

    def __init__(  # noqa: PLR0913
        self,
        database: Database,
        settings: TrainingSettings,
        *,
        child_launcher: TrainingChildLauncher | None,
        repository: TrainingRepository | None = None,
        gpu_snapshot_provider: Callable[[], GpuSnapshot] = current_gpu_snapshot,
        memory_profiles_provider: Callable[[], tuple[MemoryProfile, ...]] | None = None,
        dataset_validator: Callable[[Path], DatasetManifest] = validate_cuhk,
        worker_id: str | None = None,
    ) -> None:
        """Wire CPU request storage to the isolated CUDA telemetry/child boundary."""
        self._database = database
        self._settings = settings
        self._repository = repository or TrainingRepository()
        self._child_launcher = child_launcher
        self._gpu_snapshot_provider = gpu_snapshot_provider
        self._memory_profiles_provider = memory_profiles_provider or (
            lambda: load_memory_profiles(settings.memory_profiles_path)
        )
        self._dataset_validator = dataset_validator
        self._worker_id = worker_id or f"training-supervisor-{uuid4().hex}"
        self._children = {}

    async def run(self, stop_event: asyncio.Event) -> None:
        """Poll durable requests until shutdown; no request runs from API memory."""
        while not stop_event.is_set():
            await self._reconcile_children()
            await self._recover_untracked_slot()
            processed = await self.process_one()
            if processed:
                continue
            try:
                _ = await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self._settings.request_poll_interval_seconds,
                )
            except TimeoutError:
                continue

    async def process_one(self) -> bool:  # noqa: C901, PLR0911, PLR0912
        """Lease and resolve one request, then launch only a durably accepted job."""
        now = datetime.now(UTC)
        async with self._database.transaction() as session:
            _ = await self._repository.expire_old_requests(session, now=now)
            request = await self._repository.claim_next_request(
                session,
                owner=self._worker_id,
                now=now,
                lease_until=now + timedelta(seconds=self._settings.request_lease_seconds),
            )
        if request is None:
            return False

        try:
            config = TrainingConfig.model_validate(request.config_snapshot)
            manifest = await self._validate_request_dataset(request)
            gpu = await asyncio.to_thread(self._gpu_snapshot_provider)
            profiles = await asyncio.to_thread(self._memory_profiles_provider)
            profile, estimate = _match_profile(config, gpu, profiles)
            admission = assess_admission(estimate, gpu)
        except DatasetValidationError:
            _ = await self._resolve_refusal(
                request,
                code="training_dataset_changed",
                response={"reason": "registered_dataset_fingerprint_mismatch"},
            )
            return True
        except (TrainingDatasetNotConfiguredError, OSError):
            _ = await self._resolve_failure(request, code="training_dataset_unavailable")
            return True
        except (CalibrationRefusedError, StaleGpuSnapshotError):
            _ = await self._resolve_failure(request, code="training_gpu_unavailable")
            return True
        except UnsupportedMemoryProfileError:
            gpu = await asyncio.to_thread(self._safe_gpu_snapshot)
            reserve = _reserve_bytes(gpu.total_bytes) if gpu is not None else 0
            _ = await self._resolve_refusal(
                request,
                code="memory_profile_unsupported",
                response={
                    "training_peak_bytes": 0,
                    "reserve_bytes": reserve,
                    "required_bytes": reserve,
                    "free_bytes": gpu.free_bytes if gpu is not None else 0,
                    "reason": "memory_profile_unsupported",
                },
            )
            return True

        if not admission.admitted:
            _ = await self._resolve_refusal(
                request,
                code="training_memory_refused",
                response={
                    "training_peak_bytes": estimate.training_peak_bytes,
                    "reserve_bytes": estimate.reserve_bytes,
                    "required_bytes": estimate.required_bytes,
                    "free_bytes": admission.free_bytes,
                    "profile_identity": estimate.profile_identity,
                    "observed_at": gpu.observed_at.isoformat(),
                    "reason": admission.reason,
                },
            )
            return True

        preflight = TrainingPreflightResponse(
            admitted=True,
            training_peak_bytes=estimate.training_peak_bytes,
            reserve_bytes=estimate.reserve_bytes,
            required_bytes=estimate.required_bytes,
            free_bytes=admission.free_bytes,
            profile_identity=estimate.profile_identity,
            observed_at=gpu.observed_at,
            reason="admitted",
        )
        if request.kind == TrainingRequestKind.PREFLIGHT.value:
            _ = await self._resolve_accept(request, response=preflight.model_dump(mode="json"))
            return True
        if self._child_launcher is None:
            _ = await self._resolve_failure(request, code="training_supervisor_unavailable")
            return True

        try:
            job = await self._create_admitted_job(request, config, manifest, profile)
        except TrainingJobConflictError:
            _ = await self._resolve_refusal(
                request,
                code="training_job_conflict",
                response={"reason": "another_job_or_orphan_child_owns_the_slot"},
            )
            return True
        except TrainingPhaseTransitionError:
            _ = await self._resolve_refusal(
                request,
                code="training_job_state_invalid",
                response={"reason": "job_state_or_checkpoint_is_not_resumable"},
            )
            return True
        except TrainingRequestLeaseLostError:
            return True
        except UnsupportedMemoryProfileError:
            _ = await self._resolve_failure(request, code="memory_profile_unsupported")
            return True
        except (CalibrationRefusedError, StaleGpuSnapshotError):
            _ = await self._resolve_failure(request, code="training_gpu_unavailable")
            return True
        if job is None:
            return True
        await self._launch_owned_child(job)
        return True

    async def _validate_request_dataset(self, request: TrainingRequest) -> DatasetManifest:
        root = self._settings.require_dataset_root()
        manifest = await asyncio.to_thread(self._dataset_validator, root)
        if request.dataset_fingerprint is not None and (
            manifest.fingerprint != request.dataset_fingerprint
        ):
            raise DatasetValidationError("registered dataset fingerprint changed")
        return manifest

    async def _create_admitted_job(  # noqa: C901
        self,
        request: TrainingRequest,
        config: TrainingConfig,
        manifest: DatasetManifest,
        profile: MemoryProfile,
    ) -> TrainingJob | None:
        async with self._database.transaction() as session:
            current_request = await self._repository.get_request(
                session,
                request.request_id,
                lock=True,
            )
            if not self._owns_request_lease(current_request, request):
                raise TrainingRequestLeaseLostError("request lease is no longer current")
            await self._recover_orphan_slot(session)
            prior: TrainingJob | None = None
            if request.kind == TrainingRequestKind.RESUME.value:
                if request.parent_job_id is None:
                    raise TrainingPhaseTransitionError("resume request is missing its parent job")
                prior = await self._repository.get_job(session, request.parent_job_id)
                if prior is None:
                    raise TrainingPhaseTransitionError("resume parent job no longer exists")
                if prior.source_fingerprint != self._repository.source_fingerprint:
                    raise TrainingPhaseTransitionError("training source changed since checkpoint")
            elif request.kind != TrainingRequestKind.SUBMIT.value:
                raise TrainingPhaseTransitionError("unknown supervisor request kind")

            # Recheck while the singleton execution-slot row remains locked. The CUDA/NVML
            # query is bounded telemetry; training, downloads, and child startup happen later.
            gpu = await asyncio.to_thread(self._gpu_snapshot_provider)
            estimate = estimate_memory(config, profile, gpu)
            admission = assess_admission(estimate, gpu)
            if not admission.admitted:
                response: dict[str, object] = {
                    "training_peak_bytes": estimate.training_peak_bytes,
                    "reserve_bytes": estimate.reserve_bytes,
                    "required_bytes": estimate.required_bytes,
                    "free_bytes": admission.free_bytes,
                    "profile_identity": estimate.profile_identity,
                    "observed_at": gpu.observed_at.isoformat(),
                    "reason": admission.reason,
                }
                resolved = await self._repository.resolve_request(
                    session,
                    request.request_id,
                    owner=self._worker_id,
                    lease_generation=request.lease_generation,
                    phase=TrainingRequestPhase.REFUSED,
                    response=response,
                    error="training_memory_refused",
                )
                if not resolved:
                    raise TrainingRequestLeaseLostError("request expired before memory refusal")
                return None

            if request.kind == TrainingRequestKind.SUBMIT.value:
                job = await self._repository.create(
                    session,
                    request.request_id,
                    config,
                    manifest.public_snapshot(),
                )
                job = await self._repository.claim_job_owner(
                    session,
                    job.id,
                    expected_generation=job.owner_generation,
                )
            else:
                if prior is None:
                    raise TrainingPhaseTransitionError("resume parent job is unavailable")
                job = await self._repository.resume_interrupted(
                    session,
                    prior.id,
                    expected_generation=prior.owner_generation,
                )
            resolved = await self._repository.resolve_request(
                session,
                request.request_id,
                owner=self._worker_id,
                lease_generation=request.lease_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response={"job_id": str(job.id)},
                job_id=job.id,
            )
            if not resolved:
                raise TrainingRequestLeaseLostError("request expired before job acceptance")
            return job

    async def _recover_orphan_slot(self, session: AsyncSession) -> None:
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None:
            raise RuntimeError("training execution slot is missing; apply database migrations")
        if slot.active_job_id is None:
            return
        job = await self._repository.get_job(session, slot.active_job_id, lock=True)
        if job is None:
            raise TrainingJobConflictError("GPU slot references an unknown job")
        if TrainingPhase(job.phase).terminal:
            slot.active_job_id = None
            await session.flush()
            return
        if (
            job.phase == TrainingPhase.EVALUATING.value
            and job.engine_completed_at is not None
        ):
            return
        if job.child_pid is None or job.child_start_time is None:
            recovered = await self._repository.interrupt_starting_job_without_child(
                session,
                job.id,
                expected_generation=job.owner_generation,
            )
            if recovered:
                return
            raise TrainingOrphanProcessError("existing child identity is not yet recorded")
        identity = process_identity_status(job.child_pid, job.child_start_time)
        if identity in {
            ProcessIdentityStatus.ALIVE,
            ProcessIdentityStatus.UNVERIFIABLE,
        }:
            raise TrainingOrphanProcessError("existing training child is still alive")
        _ = await self._repository.transition(
            session,
            job.id,
            job.phase,
            TrainingPhase.INTERRUPTED,
        )

    async def _launch_owned_child(self, job: TrainingJob) -> None:
        if self._child_launcher is None:
            return
        child: TrainingChild | None = None
        try:
            child = await asyncio.to_thread(self._child_launcher.prepare, job)
            async with self._database.transaction() as session:
                recorded = await self._repository.set_child_identity(
                    session,
                    job.id,
                    owner_generation=job.owner_generation,
                    pid=child.pid,
                    start_time=child.start_time,
                )
                if not recorded:
                    raise TrainingChildLaunchError(  # noqa: TRY301
                        "child identity lost its owner generation"
                    )
            await asyncio.to_thread(child.release)
            self._children[job.id] = (child, job.owner_generation)
        except Exception as error:  # noqa: BLE001
            if child is not None:
                await asyncio.to_thread(child.abort)
            await self._mark_child_start_failed(job, error)

    async def _reconcile_children(self) -> None:
        for job_id, (child, owner_generation) in tuple(self._children.items()):
            exit_code = await asyncio.to_thread(child.poll)
            if exit_code is None:
                continue
            _ = self._children.pop(job_id, None)
            async with self._database.transaction() as session:
                _ = await self._repository.finish_child_exit(
                    session,
                    job_id,
                    owner_generation=owner_generation,
                    pid=child.pid,
                    start_time=child.start_time,
                    exit_code=exit_code,
                )

    async def _recover_untracked_slot(self) -> None:
        async with self._database.transaction() as session:
            slot = await session.scalar(
                select(TrainingExecutionSlot)
                .where(TrainingExecutionSlot.singleton.is_(True))
                .with_for_update()
            )
            if slot is None or slot.active_job_id in self._children:
                return
            try:
                await self._recover_orphan_slot(session)
            except TrainingOrphanProcessError:
                # A live or unverifiable external process keeps the slot occupied.
                return

    async def _mark_child_start_failed(self, job: TrainingJob, error: Exception) -> None:
        async with self._database.transaction() as session:
            slot = await session.scalar(
                select(TrainingExecutionSlot)
                .where(TrainingExecutionSlot.singleton.is_(True))
                .with_for_update()
            )
            if slot is None or slot.active_job_id != job.id:
                return
            current = await self._repository.get_job(session, job.id, lock=True)
            if current is None or current.owner_generation != job.owner_generation:
                return
            if current.phase == TrainingPhase.STARTING.value:
                updated = await self._repository.transition(
                    session,
                    job.id,
                    TrainingPhase.STARTING,
                    TrainingPhase.FAILED,
                )
                updated.error = _bounded_error(error)
                await session.flush()

    async def _resolve_accept(
        self,
        request: TrainingRequest,
        *,
        response: dict[str, object],
    ) -> bool:
        async with self._database.transaction() as session:
            return await self._repository.resolve_request(
                session,
                request.request_id,
                owner=self._worker_id,
                lease_generation=request.lease_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response=response,
            )

    async def _resolve_refusal(
        self,
        request: TrainingRequest,
        *,
        code: str,
        response: dict[str, object],
    ) -> bool:
        async with self._database.transaction() as session:
            return await self._repository.resolve_request(
                session,
                request.request_id,
                owner=self._worker_id,
                lease_generation=request.lease_generation,
                phase=TrainingRequestPhase.REFUSED,
                response=response,
                error=code,
            )

    async def _resolve_failure(self, request: TrainingRequest, *, code: str) -> bool:
        async with self._database.transaction() as session:
            return await self._repository.resolve_request(
                session,
                request.request_id,
                owner=self._worker_id,
                lease_generation=request.lease_generation,
                phase=TrainingRequestPhase.FAILED,
                response={"reason": code},
                error=code,
            )

    def _safe_gpu_snapshot(self) -> GpuSnapshot | None:
        try:
            return self._gpu_snapshot_provider()
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _owns_request_lease(current: TrainingRequest | None, claimed: TrainingRequest) -> bool:
        return (
            current is not None
            and current.phase == TrainingRequestPhase.PENDING.value
            and current.lease_owner is not None
            and current.lease_generation == claimed.lease_generation
            and current.lease_owner == claimed.lease_owner
            and current.expires_at > datetime.now(UTC)
            and current.lease_expires_at is not None
            and current.lease_expires_at > datetime.now(UTC)
        )


def load_memory_profiles(path: Path) -> tuple[MemoryProfile, ...]:
    """Load only the versioned, locally measured profile file."""
    try:
        decoded: object = cast("object", json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise UnsupportedMemoryProfileError("memory profile file is unavailable") from error
    if not isinstance(decoded, dict):
        raise UnsupportedMemoryProfileError("memory profile schema is unsupported")
    payload = cast("dict[str, object]", decoded)
    if payload.get("schema_version") != 1:
        raise UnsupportedMemoryProfileError("memory profile schema is unsupported")
    rows_value = payload.get("profiles")
    if not isinstance(rows_value, list):
        raise UnsupportedMemoryProfileError("memory profile list is malformed")
    rows = cast("list[object]", rows_value)
    try:
        return tuple(MemoryProfile.model_validate(row) for row in rows)
    except ValueError as error:
        raise UnsupportedMemoryProfileError("memory profile entry is malformed") from error


def _match_profile(
    config: TrainingConfig,
    gpu: GpuSnapshot,
    profiles: tuple[MemoryProfile, ...],
) -> tuple[MemoryProfile, MemoryEstimate]:
    last_error: UnsupportedMemoryProfileError | None = None
    for profile in profiles:
        if (
            profile.gpu_uuid != gpu.uuid
            or profile.batch_size != config.micro_batch_size
            or profile.mixed_precision != config.mixed_precision
            or profile.gradient_checkpointing != config.gradient_checkpointing
        ):
            continue
        try:
            return profile, estimate_memory(config, profile, gpu)
        except UnsupportedMemoryProfileError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise UnsupportedMemoryProfileError("no calibrated profile for this config")


def _reserve_bytes(total_bytes: int) -> int:
    return max(MIN_INFERENCE_RESERVE_BYTES, math.ceil(total_bytes * INFERENCE_RESERVE_FRACTION))


def _bounded_error(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message[:1000] or error.__class__.__name__


__all__ = [
    "ProcessIdentityStatus",
    "SubprocessTrainingChildLauncher",
    "TrainingChild",
    "TrainingChildLaunchError",
    "TrainingChildLauncher",
    "TrainingOrphanProcessError",
    "TrainingSupervisor",
    "process_identity_matches",
    "process_identity_status",
    "process_start_time",
]
