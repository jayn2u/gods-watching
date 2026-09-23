"""Durable persistence for active model identity and transition stage rows."""

# ruff: noqa: TC001, TC002, TC003, TRY003, EM101, EM102, E501

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import Select, delete, exists, func, insert, literal, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.model_selection.models import (
    ACTIVE_PHASES,
    TransitionPhase,
    TransitionState,
)
from gods_watching.model_selection.registry import ClipModelPackage
from gods_watching.storage.models import (
    ActiveModelIdentity,
    Appearance,
    ModelTransitionJob,
    ModelTransitionStage,
)


@dataclass(frozen=True, slots=True)
class StageResult:
    """One durable result returned after processing a crop."""

    appearance_id: UUID
    embedding: tuple[float, ...] | None
    skip_reason: str | None


class TransitionSourceChangedError(RuntimeError):
    """The appearance metadata changed while a transition was staging."""


class TransitionRepository:
    """Execute model switch persistence inside caller-owned transactions."""

    async def active_identity(
        self,
        session: AsyncSession,
        *,
        default: ClipModelPackage | None = None,
        lock: bool = False,
    ) -> ActiveModelIdentity:
        """Return the singleton active identity, creating the approved default once."""
        statement: Select[tuple[ActiveModelIdentity]] = select(ActiveModelIdentity).where(
            ActiveModelIdentity.singleton.is_(True)
        )
        if lock:
            statement = statement.with_for_update()
        row = await session.scalar(statement)
        if row is None:
            if default is None:
                raise RuntimeError("active model identity is missing")
            row = ActiveModelIdentity(
                singleton=True,
                model_id=default.model_id,
                model_revision=default.revision,
                embedding_dimension=default.dimension,
            )
            session.add(row)
            await session.flush()
        return row

    async def latest_job(
        self,
        session: AsyncSession,
        *,
        lock: bool = False,
    ) -> ModelTransitionJob | None:
        """Return the newest durable transition, if any."""
        statement: Select[tuple[ModelTransitionJob]] = select(ModelTransitionJob).order_by(
            ModelTransitionJob.created_at.desc(), ModelTransitionJob.id.desc()
        )
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement.limit(1))

    async def active_job(
        self,
        session: AsyncSession,
        *,
        lock: bool = False,
    ) -> ModelTransitionJob | None:
        """Return the one job that currently owns maintenance mode."""
        statement: Select[tuple[ModelTransitionJob]] = select(ModelTransitionJob).where(
            ModelTransitionJob.phase.in_(tuple(phase.value for phase in ACTIVE_PHASES))
        )
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement.order_by(ModelTransitionJob.created_at.asc()).limit(1))

    async def get_job(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        lock: bool = False,
    ) -> ModelTransitionJob | None:
        """Load one transition by id."""
        statement: Select[tuple[ModelTransitionJob]] = select(ModelTransitionJob).where(
            ModelTransitionJob.id == job_id
        )
        if lock:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    async def create_job(
        self,
        session: AsyncSession,
        *,
        target: ClipModelPackage,
        default: ClipModelPackage,
    ) -> ModelTransitionJob:
        """Create one queued job after locking the active identity and queue."""
        active = await self.active_identity(session, default=default, lock=True)
        if await self.active_job(session, lock=True) is not None:
            raise RuntimeError("model transition already active")
        job = ModelTransitionJob(
            source_model_id=active.model_id,
            source_model_revision=active.model_revision,
            source_dimension=active.embedding_dimension,
            target_model_id=target.model_id,
            target_model_revision=target.revision,
            target_dimension=target.dimension,
            phase=TransitionPhase.QUEUED.value,
            total=0,
            processed=0,
            skipped=0,
            skip_reasons={},
        )
        session.add(job)
        await session.flush()
        return job

    async def populate_stages(self, session: AsyncSession, job: ModelTransitionJob) -> int:
        """Snapshot every currently retained appearance and its crop identity."""
        already = select(ModelTransitionStage.appearance_id).where(
            ModelTransitionStage.job_id == job.id,
            ModelTransitionStage.appearance_id == Appearance.id,
        )
        source = select(
            literal(job.id),
            Appearance.id,
            Appearance.crop_object_key,
            Appearance.representative_version,
            literal(job.target_dimension),
        ).where(
            Appearance.tombstoned_at.is_(None),
            ~exists(already),
        )
        # PostgreSQL's INSERT .. SELECT keeps preparation bounded by the
        # database and avoids materializing the retained corpus in Python.
        _ = await session.execute(
            insert(ModelTransitionStage).from_select(
                (
                    ModelTransitionStage.job_id,
                    ModelTransitionStage.appearance_id,
                    ModelTransitionStage.source_crop_object_key,
                    ModelTransitionStage.source_representative_version,
                    ModelTransitionStage.embedding_dimension,
                ),
                source,
            )
        )
        count = await session.scalar(
            select(func.count())
            .select_from(ModelTransitionStage)
            .where(ModelTransitionStage.job_id == job.id)
        )
        job.total = int(count or 0)
        if job.phase == TransitionPhase.QUEUED.value:
            job.phase = TransitionPhase.PREPARING.value
        await session.flush()
        return job.total

    async def pending_stages(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        limit: int = 32,
        lock: bool = False,
    ) -> tuple[tuple[ModelTransitionStage, Appearance], ...]:
        """Read an independently committable batch of unprocessed stage rows."""
        statement = (
            select(ModelTransitionStage, Appearance)
            .join(Appearance, Appearance.id == ModelTransitionStage.appearance_id)
            .where(ModelTransitionStage.job_id == job_id, ModelTransitionStage.processed_at.is_(None))
            .order_by(ModelTransitionStage.appearance_id.asc())
            .limit(limit)
        )
        if lock:
            statement = statement.with_for_update()
        return tuple((row[0], row[1]) for row in (await session.execute(statement)).tuples().all())

    async def retained_count(self, session: AsyncSession) -> int:
        """Count retained crops before staging allocates temporary rows."""
        count = await session.scalar(
            select(func.count(Appearance.id)).where(Appearance.tombstoned_at.is_(None))
        )
        return int(count or 0)

    async def record_stage_results(
        self,
        session: AsyncSession,
        job: ModelTransitionJob,
        results: Iterable[StageResult],
    ) -> None:
        """Commit a batch result and advance durable counters atomically."""
        skip_reasons = dict(job.skip_reasons or {})
        for result in results:
            stage = await session.scalar(
                select(ModelTransitionStage)
                .where(
                    ModelTransitionStage.job_id == job.id,
                    ModelTransitionStage.appearance_id == result.appearance_id,
                )
                .with_for_update()
            )
            if stage is None or stage.processed_at is not None:
                continue
            if result.embedding is None:
                stage.embedding = None
                reason = result.skip_reason or "unknown_crop_skip"
                stage.skip_reason = reason
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                job.skipped += 1
            else:
                stage.embedding = _serialize_vector(result.embedding)
                stage.skip_reason = None
            stage.processed_at = datetime.now(UTC)
            job.processed += 1
        job.skip_reasons = skip_reasons
        if job.processed >= job.total:
            job.phase = TransitionPhase.ACTIVATING.value
        await session.flush()

    async def activate(
        self,
        session: AsyncSession,
        job_id: UUID,
        *,
        expected_source: tuple[str, str, int],
    ) -> ModelTransitionJob:
        """Atomically publish all staged vectors and target identity."""
        job = await self.get_job(session, job_id, lock=True)
        if job is None:
            raise RuntimeError("model transition job is missing")
        active = await self.active_identity(session, lock=True)
        if (
            active.model_id,
            active.model_revision,
            active.embedding_dimension,
        ) != expected_source:
            raise TransitionSourceChangedError("active model identity changed during transition")
        if job.processed != job.total:
            raise RuntimeError("model transition has unprocessed stage rows")
        stale = await session.scalar(
            select(ModelTransitionStage.appearance_id)
            .join(Appearance, Appearance.id == ModelTransitionStage.appearance_id)
            .where(
                ModelTransitionStage.job_id == job.id,
                (
                    (Appearance.crop_object_key != ModelTransitionStage.source_crop_object_key)
                    | (
                        Appearance.representative_version
                        != ModelTransitionStage.source_representative_version
                    )
                ),
            )
            .limit(1)
        )
        if stale is not None:
            raise TransitionSourceChangedError(f"appearance {stale} changed during transition")
        # UPDATE .. FROM stages changes every appearance in one atomic SQL
        # statement.  The source check above is intentionally in the same
        # transaction and all rows are locked by the update.
        _ = await session.execute(
            update(Appearance)
            .where(
                Appearance.id == ModelTransitionStage.appearance_id,
                ModelTransitionStage.job_id == job.id,
            )
            .values(
                embedding=ModelTransitionStage.embedding,
                embedding_dimension=job.target_dimension,
                model_id=job.target_model_id,
                model_revision=job.target_model_revision,
                embedded_at=func.now(),
            )
        )
        active.model_id = job.target_model_id
        active.model_revision = job.target_model_revision
        active.embedding_dimension = job.target_dimension
        active.updated_at = datetime.now(UTC)
        job.phase = TransitionPhase.SUCCEEDED.value
        job.updated_at = datetime.now(UTC)
        job.finished_at = datetime.now(UTC)
        job.error = None
        await self.delete_stages(session, job.id)
        await session.flush()
        return job

    async def delete_stages(self, session: AsyncSession, job_id: UUID) -> None:
        """Drop temporary stage rows while retaining the transition audit row."""
        _ = await session.execute(
            delete(ModelTransitionStage).where(ModelTransitionStage.job_id == job_id)
        )

    async def set_phase(
        self,
        session: AsyncSession,
        job_id: UUID,
        phase: TransitionPhase,
        *,
        error: str | None = None,
    ) -> ModelTransitionJob:
        """Persist one bounded phase or safe operator-facing error."""
        job = await self.get_job(session, job_id, lock=True)
        if job is None:
            raise RuntimeError("model transition job is missing")
        job.phase = phase.value
        job.error = error
        job.updated_at = datetime.now(UTC)
        if phase in {TransitionPhase.SUCCEEDED, TransitionPhase.FAILED}:
            job.finished_at = datetime.now(UTC)
        await session.flush()
        return job

    async def state(
        self,
        session: AsyncSession,
        *,
        default: ClipModelPackage | None = None,
    ) -> tuple[ActiveModelIdentity, ModelTransitionJob | None]:
        """Read the current identity and newest transition for status endpoints."""
        active = await self.active_identity(session, default=default)
        job = await self.latest_job(session)
        return active, job


def transition_state(job: ModelTransitionJob) -> TransitionState:
    """Convert an ORM job to an immutable domain state."""
    return TransitionState(
        id=job.id,
        source_model_id=job.source_model_id,
        target_model_id=job.target_model_id,
        source_model_revision=job.source_model_revision,
        target_model_revision=job.target_model_revision,
        source_dimension=job.source_dimension,
        target_dimension=job.target_dimension,
        phase=TransitionPhase(job.phase),
        processed=job.processed,
        total=job.total,
        skipped=job.skipped,
        skip_reasons=dict(job.skip_reasons or {}),
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _serialize_vector(values: Iterable[float]) -> str:
    return "[" + ",".join(format(float(value), ".9g") for value in values) + "]"


__all__ = [
    "StageResult",
    "TransitionRepository",
    "TransitionSourceChangedError",
    "transition_state",
]
