"""Transaction-scoped persistence operations for durable training jobs."""

# ruff: noqa: TRY003, EM101, EM102

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from pydantic import BaseModel
from sqlalchemy import func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from gods_watching.training.models import (
    TrainingExecutionSlot,
    TrainingJob,
    TrainingPhase,
    TrainingRequest,
    TrainingRequestKind,
    TrainingRequestPhase,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.contracts.training import TrainingConfig, TrainingEvaluationSummary

_FINGERPRINT_LENGTH = 64
_MAX_CANDIDATE_IDENTITY_LENGTH = 255


@runtime_checkable
class _PublicSnapshot(Protocol):
    def public_snapshot(self) -> dict[str, object]:
        """Return an API-safe snapshot that omits raw dataset material."""
        ...


class TrainingRepositoryError(RuntimeError):
    """Base class for expected training repository conflicts."""


class TrainingJobConflictError(TrainingRepositoryError):
    """Another training job currently owns the GPU execution slot."""


class TrainingRequestConflictError(TrainingRepositoryError):
    """A request identity was reused with different immutable inputs."""


class TrainingPhaseTransitionError(TrainingRepositoryError):
    """A phase update was invalid or lost its compare-and-swap race."""


class TrainingRequestLeaseLostError(TrainingRepositoryError):
    """A supervisor response lost its request generation/expiry fence."""


class TrainingRequestNotFoundError(TrainingRepositoryError):
    """A durable supervisor request identity does not exist."""


class TrainingJobNotFoundError(TrainingRepositoryError):
    """A durable training job identity does not exist."""


_ALLOWED_TRANSITIONS: dict[TrainingPhase, frozenset[TrainingPhase]] = {
    TrainingPhase.STARTING: frozenset(
        {
            TrainingPhase.TRAINING,
            TrainingPhase.CANCELLING,
            TrainingPhase.FAILED,
            TrainingPhase.INTERRUPTED,
        }
    ),
    TrainingPhase.TRAINING: frozenset(
        {
            TrainingPhase.EVALUATING,
            TrainingPhase.CANCELLING,
            TrainingPhase.FAILED,
            TrainingPhase.INTERRUPTED,
        }
    ),
    TrainingPhase.EVALUATING: frozenset(
        {
            TrainingPhase.PUBLISHING,
            TrainingPhase.CANCELLING,
            TrainingPhase.FAILED,
            TrainingPhase.INTERRUPTED,
        }
    ),
    TrainingPhase.PUBLISHING: frozenset(
        {
            TrainingPhase.SUCCEEDED,
            TrainingPhase.CANCELLING,
            TrainingPhase.FAILED,
            TrainingPhase.INTERRUPTED,
        }
    ),
    TrainingPhase.CANCELLING: frozenset(
        {TrainingPhase.CANCELLED, TrainingPhase.FAILED, TrainingPhase.INTERRUPTED}
    ),
    TrainingPhase.SUCCEEDED: frozenset(),
    TrainingPhase.CANCELLED: frozenset(),
    TrainingPhase.FAILED: frozenset(),
    TrainingPhase.INTERRUPTED: frozenset(),
}


def _source_fingerprint() -> str:
    """Hash the installed application Python sources to bind a run to its code."""
    package_root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    source_files: Iterable[Path] = sorted(package_root.rglob("*.py"))
    for path in source_files:
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _json_default(value: object) -> object:  # noqa: PLR0911
    """Serialize additional immutable domain values nested in a JSON snapshot."""
    if isinstance(value, _PublicSnapshot):
        return value.public_snapshot()
    if isinstance(value, BaseModel):
        return cast("object", value.model_dump(mode="json"))
    if is_dataclass(value) and not isinstance(value, type):
        return cast("object", asdict(value))
    if isinstance(value, Enum):
        return cast("object", value.value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"unsupported training snapshot value: {type(value).__name__}")


def _json_value(value: object) -> object:
    """Convert supported domain values into an immutable JSON-compatible snapshot."""
    encoded = json.dumps(value, default=_json_default, allow_nan=False)
    return cast("object", json.loads(encoded))


def _fingerprint_from_snapshot(snapshot: dict[str, object]) -> str:
    """Extract the validated dataset fingerprint persisted with a run."""
    raw = snapshot.get("fingerprint", snapshot.get("dataset_fingerprint"))
    if (
        not isinstance(raw, str)
        or len(raw) != _FINGERPRINT_LENGTH
        or any(c not in "0123456789abcdef" for c in raw)
    ):
        raise ValueError("dataset snapshot must contain a lowercase SHA-256 fingerprint")
    return raw


def _phase(value: TrainingPhase | str) -> TrainingPhase:
    """Normalize a phase argument before constructing a SQL predicate."""
    if isinstance(value, TrainingPhase):
        return value
    try:
        return TrainingPhase(value)
    except ValueError as exc:
        raise TrainingPhaseTransitionError(f"unknown training phase: {value}") from exc


class TrainingRepository:
    """Persist jobs inside caller-owned SQLAlchemy transactions."""

    def __init__(self) -> None:
        """Bind subsequent jobs to the current installed Python source tree."""
        self._source_fingerprint: str = _source_fingerprint()

    @property
    def source_fingerprint(self) -> str:
        """Return the source identity embedded in jobs created by this repository."""
        return self._source_fingerprint

    async def create(
        self,
        session: AsyncSession,
        request_id: UUID,
        config: TrainingConfig,
        dataset: object,
    ) -> TrainingJob:
        """Create a job once, returning the same durable job for duplicate requests."""
        config_snapshot = cast("dict[str, object]", _json_value(config.model_dump(mode="json")))
        normalized_dataset = _json_value(dataset)
        if not isinstance(normalized_dataset, dict):
            raise TypeError("dataset snapshot must be a mapping or a mapping-backed contract")
        dataset_snapshot = cast("dict[str, object]", normalized_dataset)
        dataset_fingerprint = _fingerprint_from_snapshot(dataset_snapshot)

        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None:
            raise RuntimeError("training execution slot is missing; apply database migrations")

        # The slot lock comes first on every create path to keep row-lock ordering consistent.
        existing = await self._by_request_id(session, request_id, lock=True)
        if existing is not None:
            self._ensure_same_request(existing, config_snapshot, dataset_snapshot)
            return existing

        if slot.active_job_id is not None:
            active = await session.get(TrainingJob, slot.active_job_id)
            if active is not None and not TrainingPhase(active.phase).terminal:
                raise TrainingJobConflictError("another training job currently owns the GPU slot")
            slot.active_job_id = None

        job = TrainingJob(
            id=uuid4(),
            request_id=request_id,
            config_snapshot=config_snapshot,
            dataset_snapshot=dataset_snapshot,
            dataset_fingerprint=dataset_fingerprint,
            source_fingerprint=self._source_fingerprint,
            phase=TrainingPhase.STARTING.value,
            current_epoch=0,
            current_step=0,
            owner_generation=0,
            cancel_requested=False,
            attempts=0,
        )
        session.add(job)
        await session.flush()
        slot.active_job_id = job.id
        await session.flush()
        return job

    async def create_request(  # noqa: PLR0913
        self,
        session: AsyncSession,
        request_id: UUID,
        kind: TrainingRequestKind | str,
        config: TrainingConfig,
        *,
        expires_at: datetime,
        dataset: object | None = None,
        parent_job_id: UUID | None = None,
    ) -> TrainingRequest:
        """Persist an idempotent request before the API waits for the supervisor."""
        normalized_kind = (
            kind if isinstance(kind, TrainingRequestKind) else TrainingRequestKind(kind)
        )
        config_snapshot = cast("dict[str, object]", _json_value(config.model_dump(mode="json")))
        dataset_snapshot: dict[str, object] | None = None
        dataset_fingerprint: str | None = None
        if dataset is not None:
            normalized_dataset = _json_value(dataset)
            if not isinstance(normalized_dataset, dict):
                raise TypeError("dataset snapshot must be a mapping or a mapping-backed contract")
            dataset_snapshot = cast("dict[str, object]", normalized_dataset)
            dataset_fingerprint = _fingerprint_from_snapshot(dataset_snapshot)
        if (
            normalized_kind in {TrainingRequestKind.SUBMIT, TrainingRequestKind.RESUME}
            and dataset_snapshot is None
        ):
            raise ValueError("submit and resume requests require a dataset snapshot")
        if expires_at.tzinfo is None or expires_at.utcoffset() is None:
            raise ValueError("training request expiry must include a timezone")

        statement = (
            pg_insert(TrainingRequest)
            .values(
                id=uuid4(),
                request_id=request_id,
                kind=normalized_kind.value,
                parent_job_id=parent_job_id,
                config_snapshot=config_snapshot,
                dataset_snapshot=dataset_snapshot,
                dataset_fingerprint=dataset_fingerprint,
                source_fingerprint=self._source_fingerprint,
                phase=TrainingRequestPhase.PENDING.value,
                lease_generation=0,
                expires_at=expires_at,
            )
            .on_conflict_do_nothing(constraint="uq_training_requests_request_id")
        )
        _ = await session.execute(statement)
        existing = await session.scalar(
            select(TrainingRequest)
            .where(TrainingRequest.request_id == request_id)
            .with_for_update()
        )
        if existing is None:
            raise TrainingRequestNotFoundError("training request was not visible after insert")
        if (
            existing.kind != normalized_kind.value
            or existing.parent_job_id != parent_job_id
            or existing.config_snapshot != config_snapshot
            or existing.dataset_snapshot != dataset_snapshot
            or existing.dataset_fingerprint != dataset_fingerprint
        ):
            raise TrainingRequestConflictError(
                "request identity was already used with different immutable inputs"
            )
        return existing

    async def get_job(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        lock: bool = False,
    ) -> TrainingJob | None:
        """Load one job with an optional row lock for an action update."""
        statement = select(TrainingJob).where(TrainingJob.id == job_id)
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    async def list_jobs(
        self,
        session: AsyncSession,
        *,
        limit: int,
        before: tuple[datetime, UUID] | None = None,
    ) -> tuple[TrainingJob, ...]:
        """Return the newest bounded job page in stable descending order."""
        statement = select(TrainingJob)
        if before is not None:
            created_at, job_id = before
            statement = statement.where(
                or_(
                    TrainingJob.created_at < created_at,
                    (TrainingJob.created_at == created_at) & (TrainingJob.id < job_id),
                )
            )
        rows = await session.scalars(
            statement.order_by(TrainingJob.created_at.desc(), TrainingJob.id.desc()).limit(limit)
        )
        return tuple(rows.all())

    async def claim_job_owner(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        expected_generation: int,
    ) -> TrainingJob:
        """Fence one child launch and increment the durable attempt counter."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            raise TrainingPhaseTransitionError("training job does not own the GPU execution slot")
        result = await session.execute(
            update(TrainingJob)
            .where(
                TrainingJob.id == job_id,
                TrainingJob.phase == TrainingPhase.STARTING.value,
                TrainingJob.owner_generation == expected_generation,
                TrainingJob.child_pid.is_(None),
            )
            .values(
                owner_generation=TrainingJob.owner_generation + 1,
                attempts=TrainingJob.attempts + 1,
                started_at=func.coalesce(TrainingJob.started_at, func.now()),
                updated_at=func.now(),
            )
        )
        if result.rowcount != 1:
            raise TrainingPhaseTransitionError("training owner generation compare-and-swap failed")
        job = await session.get(TrainingJob, job_id, populate_existing=True)
        if job is None:
            raise TrainingPhaseTransitionError("training job no longer exists")
        return job

    async def set_child_identity(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
    ) -> bool:
        """Bind a child PID to the exact durable supervisor generation."""
        result = await session.execute(
            update(TrainingJob)
            .where(
                TrainingJob.id == job_id,
                TrainingJob.phase == TrainingPhase.STARTING.value,
                TrainingJob.owner_generation == owner_generation,
                TrainingJob.child_pid.is_(None),
            )
            .values(child_pid=pid, child_start_time=start_time, updated_at=func.now())
        )
        return result.rowcount == 1

    async def recover_job_without_child(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        expected_generation: int,
    ) -> bool:
        """Recover an active job when no child identity remains to supervise."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != expected_generation
            or job.child_pid is not None
            or job.child_start_time is not None
            or TrainingPhase(job.phase).terminal
        ):
            return False
        phase = TrainingPhase(job.phase)
        next_phase = (
            TrainingPhase.CANCELLED
            if phase == TrainingPhase.CANCELLING
            else TrainingPhase.INTERRUPTED
        )
        if next_phase not in _ALLOWED_TRANSITIONS[phase]:
            return False
        result = await session.execute(
            update(TrainingJob)
            .where(
                TrainingJob.id == job_id,
                TrainingJob.phase == phase.value,
                TrainingJob.owner_generation == expected_generation,
                TrainingJob.child_pid.is_(None),
                TrainingJob.child_start_time.is_(None),
            )
            .values(
                phase=next_phase.value,
                owner_generation=TrainingJob.owner_generation + 1,
                finished_at=func.now(),
                updated_at=func.now(),
            )
        )
        if result.rowcount != 1:
            return False
        slot.active_job_id = None
        await session.flush()
        return True

    async def mark_training_started(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
    ) -> bool:
        """Advance a gated child only if cancellation has not won the slot/job race."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or job.phase != TrainingPhase.STARTING.value
            or job.cancel_requested
        ):
            return False
        job.phase = TrainingPhase.TRAINING.value
        job.started_at = job.started_at or datetime.now(UTC)
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def report_training_progress(  # noqa: PLR0913
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
        epoch: int,
        step: int,
        checkpoint_path: Path | None = None,
        best_metric: float | None = None,
    ) -> bool:
        """Fence transient progress and completed-checkpoint pointers by child generation."""
        if epoch < 0 or step < 0 or (
            best_metric is not None and not 0 <= best_metric <= 1
        ):
            raise ValueError("training progress is outside its durable bounds")
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or job.phase
            not in {
                TrainingPhase.TRAINING.value,
                TrainingPhase.EVALUATING.value,
                TrainingPhase.PUBLISHING.value,
                TrainingPhase.CANCELLING.value,
            }
        ):
            return False
        job.current_epoch = epoch
        job.current_step = step
        if checkpoint_path is not None:
            job.checkpoint_path = checkpoint_path.as_posix()
        if best_metric is not None:
            job.best_metric = best_metric
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def report_training_error(  # noqa: PLR0913
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
        error: str,
    ) -> bool:
        """Persist a bounded child error without claiming a terminal process outcome."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or TrainingPhase(job.phase).terminal
        ):
            return False
        job.error = error[:1000]
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def mark_engine_staging(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
    ) -> bool:
        """Move a trained child to final-evaluation staging without claiming job success."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or job.phase != TrainingPhase.TRAINING.value
            or job.cancel_requested
        ):
            return False
        job.phase = TrainingPhase.EVALUATING.value
        job.engine_completed_at = datetime.now(UTC)
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def mark_publishing(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
    ) -> bool:
        """Fence immutable publication to the live child after final evaluation."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or job.phase != TrainingPhase.EVALUATING.value
            or job.cancel_requested
            or job.engine_completed_at is None
        ):
            return False
        job.phase = TrainingPhase.PUBLISHING.value
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def record_candidate_publication(  # noqa: PLR0913
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
        candidate_model_id: str,
        candidate_revision: str,
        evaluation: TrainingEvaluationSummary,
    ) -> bool:
        """Persist a safe report and package identity while the owning child is live."""
        if (
            not candidate_model_id
            or len(candidate_model_id) > _MAX_CANDIDATE_IDENTITY_LENGTH
        ):
            raise ValueError("candidate model identity is invalid")
        if not candidate_revision or len(candidate_revision) > _MAX_CANDIDATE_IDENTITY_LENGTH:
            raise ValueError("candidate revision is invalid")
        if candidate_revision != evaluation.package_sha256:
            raise ValueError("candidate revision must match the reported package digest")

        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return False
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
            or job.phase != TrainingPhase.PUBLISHING.value
            or job.cancel_requested
            or evaluation.dataset_sha256 != job.dataset_fingerprint
            or evaluation.training_source_fingerprint != job.source_fingerprint
            or evaluation.dataset_split != "test"
        ):
            return False
        job.candidate_model_id = candidate_model_id
        job.candidate_revision = candidate_revision
        job.evaluation_report = evaluation.model_dump(mode="json")
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return True

    async def finish_child_exit(  # noqa: PLR0911, PLR0913
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        owner_generation: int,
        pid: int,
        start_time: int,
        exit_code: int,
        now: datetime | None = None,
    ) -> TrainingJob | None:
        """Finalize a child outcome only after its supervisor confirmed process exit."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            return None
        job = await self.get_job(session, job_id, lock=True)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
        ):
            return None
        phase = TrainingPhase(job.phase)
        if phase.terminal:
            return job
        finished_at = now or datetime.now(UTC)
        if phase == TrainingPhase.CANCELLING:
            next_phase = (
                TrainingPhase.CANCELLED if exit_code == 0 else TrainingPhase.FAILED
            )
            updated = await self.transition(session, job_id, phase, next_phase)
            if exit_code != 0:
                updated.error = f"training child exited with status {exit_code}"
            await session.flush()
            return updated
        if exit_code != 0:
            updated = await self.transition(
                session,
                job_id,
                phase,
                TrainingPhase.FAILED,
            )
            updated.error = f"training child exited with status {exit_code}"
            await session.flush()
            return updated
        if phase != TrainingPhase.EVALUATING:
            if (
                phase == TrainingPhase.PUBLISHING
                and job.candidate_model_id is not None
                and job.candidate_revision is not None
                and job.evaluation_report is not None
            ):
                job.child_pid = None
                job.child_start_time = None
                job.engine_completed_at = job.engine_completed_at or finished_at
                await session.flush()
                return await self.transition(
                    session,
                    job_id,
                    TrainingPhase.PUBLISHING,
                    TrainingPhase.SUCCEEDED,
                )
            updated = await self.transition(
                session,
                job_id,
                phase,
                TrainingPhase.FAILED,
            )
            updated.error = (
                "training child exited before publication completed"
                if phase == TrainingPhase.PUBLISHING
                else "training child exited before evaluation staging"
            )
            await session.flush()
            return updated
        job.engine_completed_at = finished_at
        job.child_pid = None
        job.child_start_time = None
        job.updated_at = finished_at
        await session.flush()
        return job

    async def request_cancel(self, session: AsyncSession, job_id: UUID) -> TrainingJob:
        """Set cooperative cancellation and move a cancellable active job to cancelling."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None or slot.active_job_id != job_id:
            raise TrainingPhaseTransitionError("training job does not own the GPU execution slot")
        job = await self.get_job(session, job_id, lock=True)
        if job is None:
            raise TrainingJobNotFoundError("training job does not exist")
        phase = TrainingPhase(job.phase)
        if phase == TrainingPhase.CANCELLING:
            job.cancel_requested = True
            await session.flush()
            return job
        if phase not in {
            TrainingPhase.STARTING,
            TrainingPhase.TRAINING,
            TrainingPhase.EVALUATING,
            TrainingPhase.PUBLISHING,
        }:
            raise TrainingPhaseTransitionError("training job cannot be cancelled in this phase")
        job.phase = TrainingPhase.CANCELLING.value
        job.cancel_requested = True
        job.updated_at = datetime.now(UTC)
        await session.flush()
        return job

    async def resume_interrupted(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        expected_generation: int,
    ) -> TrainingJob:
        """Reclaim an interrupted job only after its prior child was verified absent."""
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None:
            raise RuntimeError("training execution slot is missing; apply database migrations")
        if slot.active_job_id is not None:
            raise TrainingJobConflictError("another training job currently owns the GPU slot")
        result = await session.execute(
            update(TrainingJob)
            .where(
                TrainingJob.id == job_id,
                TrainingJob.phase == TrainingPhase.INTERRUPTED.value,
                TrainingJob.owner_generation == expected_generation,
                TrainingJob.checkpoint_path.is_not(None),
            )
            .values(
                phase=TrainingPhase.STARTING.value,
                owner_generation=TrainingJob.owner_generation + 1,
                attempts=TrainingJob.attempts + 1,
                child_pid=None,
                child_start_time=None,
                cancel_requested=False,
                error=None,
                engine_completed_at=None,
                finished_at=None,
                updated_at=func.now(),
            )
        )
        if result.rowcount != 1:
            raise TrainingPhaseTransitionError(
                "interrupted job is not resumable or its generation changed"
            )
        slot.active_job_id = job_id
        job = await session.get(TrainingJob, job_id, populate_existing=True)
        if job is None:
            raise TrainingPhaseTransitionError("training job no longer exists")
        return job

    async def get_request(
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        lock: bool = False,
    ) -> TrainingRequest | None:
        """Load one supervisor request by its caller idempotency identity."""
        statement = select(TrainingRequest).where(TrainingRequest.request_id == request_id)
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    async def claim_next_request(
        self,
        session: AsyncSession,
        *,
        owner: str,
        now: datetime,
        lease_until: datetime,
    ) -> TrainingRequest | None:
        """Lease the oldest live request with row locking and generation fencing."""
        statement = (
            select(TrainingRequest)
            .where(
                TrainingRequest.phase == TrainingRequestPhase.PENDING.value,
                TrainingRequest.expires_at > now,
                or_(
                    TrainingRequest.lease_expires_at.is_(None),
                    TrainingRequest.lease_expires_at <= now,
                ),
            )
            .order_by(TrainingRequest.created_at, TrainingRequest.id)
            .with_for_update(skip_locked=True)
            .limit(1)
        )
        request = await session.scalar(statement)
        if request is None:
            return None
        request.lease_generation += 1
        request.lease_owner = owner
        request.lease_expires_at = lease_until
        await session.flush()
        return request

    async def resolve_request(  # noqa: PLR0913
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        owner: str,
        lease_generation: int,
        phase: TrainingRequestPhase,
        response: dict[str, object],
        job_id: UUID | None = None,
        error: str | None = None,
        now: datetime | None = None,
    ) -> bool:
        """Commit a result only while the request lease and expiry remain current."""
        if phase not in {
            TrainingRequestPhase.ACCEPTED,
            TrainingRequestPhase.REFUSED,
            TrainingRequestPhase.FAILED,
        }:
            raise ValueError("a supervisor can only resolve a request to a terminal phase")
        if now is None:
            # Lock first, then ask PostgreSQL for wall-clock time. Capturing Python
            # time before a row-lock wait could accept a request after expiry.
            current = await session.scalar(
                select(TrainingRequest)
                .where(TrainingRequest.request_id == request_id)
                .with_for_update()
            )
            if current is None:
                return False
            resolved_at = await session.scalar(select(func.clock_timestamp()))
            if resolved_at is None:
                raise RuntimeError("database clock is unavailable")
        else:
            resolved_at = now
        result = await session.execute(
            update(TrainingRequest)
            .where(
                TrainingRequest.request_id == request_id,
                TrainingRequest.phase == TrainingRequestPhase.PENDING.value,
                TrainingRequest.lease_owner == owner,
                TrainingRequest.lease_generation == lease_generation,
                TrainingRequest.lease_expires_at > resolved_at,
                TrainingRequest.expires_at > resolved_at,
            )
            .values(
                phase=phase.value,
                job_id=job_id,
                response_snapshot=response,
                error=error[:1000] if error is not None else None,
                lease_owner=None,
                lease_expires_at=None,
                resolved_at=resolved_at,
                updated_at=func.now(),
            )
        )
        return result.rowcount == 1

    async def expire_request(
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Expire only an unresolved request so a supervisor cannot launch it later."""
        expired_at = now or datetime.now(UTC)
        result = await session.execute(
            update(TrainingRequest)
            .where(
                TrainingRequest.request_id == request_id,
                TrainingRequest.phase == TrainingRequestPhase.PENDING.value,
            )
            .values(
                phase=TrainingRequestPhase.EXPIRED.value,
                lease_owner=None,
                lease_expires_at=None,
                resolved_at=expired_at,
                updated_at=func.now(),
            )
        )
        return result.rowcount == 1

    async def expire_old_requests(
        self,
        session: AsyncSession,
        *,
        now: datetime | None = None,
    ) -> int:
        """Reclaim expired pending requests during supervisor polling."""
        expired_at = now or datetime.now(UTC)
        result = await session.execute(
            update(TrainingRequest)
            .where(
                TrainingRequest.phase == TrainingRequestPhase.PENDING.value,
                TrainingRequest.expires_at <= expired_at,
            )
            .values(
                phase=TrainingRequestPhase.EXPIRED.value,
                lease_owner=None,
                lease_expires_at=None,
                resolved_at=expired_at,
                updated_at=func.now(),
            )
        )
        return int(result.rowcount or 0)

    async def transition(
        self,
        session: AsyncSession,
        job_id: UUID,
        expected_phase: TrainingPhase | str,
        next_phase: TrainingPhase | str,
    ) -> TrainingJob:
        """Compare-and-swap a legal phase change and release a terminal job's slot."""
        expected = _phase(expected_phase)
        next_value = _phase(next_phase)
        if next_value not in _ALLOWED_TRANSITIONS[expected]:
            raise TrainingPhaseTransitionError(
                f"training phase cannot move from {expected.value} to {next_value.value}"
            )

        # Acquire the slot before the job row, matching create() and avoiding lock-order cycles.
        slot = await session.scalar(
            select(TrainingExecutionSlot)
            .where(TrainingExecutionSlot.singleton.is_(True))
            .with_for_update()
        )
        if slot is None:
            raise RuntimeError("training execution slot is missing; apply database migrations")
        if slot.active_job_id != job_id:
            raise TrainingPhaseTransitionError("training job does not own the GPU execution slot")

        values: dict[str, object] = {"phase": next_value.value, "updated_at": func.now()}
        if next_value.terminal:
            values["finished_at"] = func.now()
            values["child_pid"] = None
            values["child_start_time"] = None
        result = await session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job_id, TrainingJob.phase == expected.value)
            .values(**values)
        )
        if result.rowcount != 1:
            raise TrainingPhaseTransitionError(
                "training phase compare-and-swap failed because the job changed"
            )

        if next_value.terminal:
            if slot.active_job_id != job_id:
                raise TrainingPhaseTransitionError(
                    "terminal training job did not own the GPU execution slot"
                )
            slot.active_job_id = None

        job = await session.get(TrainingJob, job_id, populate_existing=True)
        if job is None:
            raise TrainingPhaseTransitionError("training job no longer exists")
        return job

    @staticmethod
    async def _by_request_id(
        session: AsyncSession,
        request_id: UUID,
        *,
        lock: bool,
    ) -> TrainingJob | None:
        statement = select(TrainingJob).where(TrainingJob.request_id == request_id)
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    def _ensure_same_request(
        self,
        existing: TrainingJob,
        config_snapshot: dict[str, object],
        dataset_snapshot: dict[str, object],
    ) -> None:
        if (
            existing.config_snapshot != config_snapshot
            or existing.dataset_snapshot != dataset_snapshot
        ):
            raise TrainingRequestConflictError(
                "request identity was already used with different immutable inputs"
            )


__all__ = [
    "TrainingJobConflictError",
    "TrainingJobNotFoundError",
    "TrainingPhaseTransitionError",
    "TrainingRepository",
    "TrainingRepositoryError",
    "TrainingRequestConflictError",
    "TrainingRequestLeaseLostError",
    "TrainingRequestNotFoundError",
]
