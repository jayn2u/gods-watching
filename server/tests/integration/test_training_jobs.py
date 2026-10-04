from __future__ import annotations

import asyncio
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from gods_watching.contracts.training import TrainingConfig
from gods_watching.training.models import TrainingExecutionSlot, TrainingJob, TrainingPhase
from gods_watching.training.repository import (
    TrainingJobConflictError,
    TrainingPhaseTransitionError,
    TrainingRepository,
    TrainingRequestConflictError,
)

_DATASET = {
    "dataset_id": "cuhk-pedes",
    "fingerprint": "a" * 64,
    "split_counts": {"train": 1, "val": 1, "test": 1},
}


@pytest.mark.anyio
async def test_duplicate_request_is_idempotent(session: AsyncSession) -> None:
    repository = TrainingRepository()
    request_id = uuid4()

    first = await repository.create(session, request_id, TrainingConfig(), _DATASET)
    duplicate = await repository.create(session, request_id, TrainingConfig(), _DATASET)

    assert duplicate.id == first.id
    assert duplicate.request_id == request_id

    with pytest.raises(TrainingRequestConflictError):
        _ = await repository.create(
            session,
            request_id,
            TrainingConfig(epochs=31),
            _DATASET,
        )


@pytest.mark.anyio
async def test_concurrent_active_job_is_rejected(engine: AsyncEngine) -> None:
    repository = TrainingRepository()
    factory = async_sessionmaker(engine, expire_on_commit=False)
    start_together = asyncio.Barrier(2)

    async def submit(request_id: UUID) -> TrainingJob:
        async with factory() as session, session.begin():
            _ = await start_together.wait()
            return await repository.create(
                session,
                request_id,
                TrainingConfig(),
                _DATASET,
            )

    results = await asyncio.gather(
        submit(uuid4()),
        submit(uuid4()),
        return_exceptions=True,
    )

    jobs = [result for result in results if isinstance(result, TrainingJob)]
    job_ids = [job.id for job in jobs]
    try:
        assert sum(isinstance(result, TrainingJobConflictError) for result in results) == 1
        assert len(jobs) == 1
        async with factory() as session:
            active_count = await session.scalar(
                select(func.count())
                .select_from(TrainingJob)
                .where(
                    TrainingJob.phase.in_(tuple(phase.value for phase in TrainingPhase.active()))
                )
            )
        assert active_count == 1
    finally:
        if job_ids:
            async with factory.begin() as session:
                _ = await session.execute(
                    update(TrainingExecutionSlot)
                    .where(TrainingExecutionSlot.active_job_id.in_(job_ids))
                    .values(active_job_id=None)
                )
                _ = await session.execute(delete(TrainingJob).where(TrainingJob.id.in_(job_ids)))


@pytest.mark.anyio
async def test_terminal_phase_cannot_regress(session: AsyncSession) -> None:
    repository = TrainingRepository()
    job = await repository.create(session, uuid4(), TrainingConfig(), _DATASET)

    _ = await repository.transition(session, job.id, TrainingPhase.STARTING, TrainingPhase.TRAINING)
    _ = await repository.transition(
        session, job.id, TrainingPhase.TRAINING, TrainingPhase.EVALUATING
    )
    _ = await repository.transition(
        session, job.id, TrainingPhase.EVALUATING, TrainingPhase.PUBLISHING
    )
    _ = await repository.transition(
        session, job.id, TrainingPhase.PUBLISHING, TrainingPhase.SUCCEEDED
    )

    with pytest.raises(TrainingPhaseTransitionError):
        _ = await repository.transition(
            session,
            job.id,
            TrainingPhase.SUCCEEDED,
            TrainingPhase.TRAINING,
        )


@pytest.mark.anyio
async def test_immutable_training_inputs_cannot_be_updated(session: AsyncSession) -> None:
    repository = TrainingRepository()
    job = await repository.create(session, uuid4(), TrainingConfig(), _DATASET)

    with pytest.raises(IntegrityError):
        _ = await session.execute(
            update(TrainingJob)
            .where(TrainingJob.id == job.id)
            .values(config_snapshot={"epochs": 100})
        )


@pytest.mark.anyio
async def test_child_identity_requires_pid_and_start_time(session: AsyncSession) -> None:
    repository = TrainingRepository()
    job = await repository.create(session, uuid4(), TrainingConfig(), _DATASET)

    with pytest.raises(IntegrityError):
        _ = await session.execute(
            update(TrainingJob).where(TrainingJob.id == job.id).values(child_start_time=1)
        )
