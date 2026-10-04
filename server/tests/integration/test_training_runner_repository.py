from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import delete

from gods_watching.contracts.training import TrainingConfig, TrainingMetric
from gods_watching.storage import Database
from gods_watching.training.models import TrainingExecutionSlot, TrainingJob, TrainingPhase
from gods_watching.training.repository import TrainingRepository, TrainingRequestLeaseLostError
from gods_watching.training.runner import _RunnerReporter  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from pathlib import Path

_DATASET = {
    "dataset_id": "cuhk-pedes",
    "fingerprint": "a" * 64,
    "protocol": "cuhk-pedes-original-splits-v1",
    "split_counts": {
        "train": {"images": 2, "captions": 4, "identities": 2},
        "val": {"images": 1, "captions": 2, "identities": 1},
        "test": {"images": 1, "captions": 2, "identities": 1},
    },
}


async def _create_owned_job(database: Database, repository: TrainingRepository) -> TrainingJob:
    async with database.transaction() as session:
        job = await repository.create(session, uuid4(), TrainingConfig(), _DATASET)
        return await repository.claim_job_owner(
            session,
            job.id,
            expected_generation=job.owner_generation,
        )


@pytest.mark.anyio
async def test_reporter_progress_and_engine_completion_do_not_succeed_or_release_slot(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    job = await _create_owned_job(database, repository)
    try:
        async with database.transaction() as session:
            recorded = await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            assert recorded
            started = await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            assert started
            updated = await repository.report_training_progress(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
                epoch=1,
                step=3,
                checkpoint_path=tmp_path / "last.pt",
                best_metric=0.25,
            )
            assert updated
            staging = await repository.mark_engine_staging(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            assert staging

        async with database.transaction() as session:
            _ = await repository.finish_child_exit(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
                exit_code=0,
                now=datetime.now(UTC),
            )
            final = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert final is not None
            assert final.phase == TrainingPhase.EVALUATING.value
            assert final.engine_completed_at is not None
            assert final.current_epoch == 1
            assert final.current_step == 3
            assert final.best_metric == 0.25
            assert final.child_pid is None
            assert slot is not None
            assert slot.active_job_id == job.id
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_child_exit_releases_slot_only_after_cancelled_process_exits(
    database_url: str,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    job = await _create_owned_job(database, repository)
    try:
        async with database.transaction() as session:
            _ = await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            _ = await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            _ = await repository.request_cancel(session, job.id)
            slot_before = await session.get(TrainingExecutionSlot, True)
            assert slot_before is not None
            assert slot_before.active_job_id == job.id

        async with database.transaction() as session:
            _ = await repository.finish_child_exit(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
                exit_code=0,
                now=datetime.now(UTC),
            )
            final = await repository.get_job(session, job.id)
            slot_after = await session.get(TrainingExecutionSlot, True)
            assert final is not None
            assert final.phase == TrainingPhase.CANCELLED.value
            assert slot_after is not None
            assert slot_after.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_cancel_before_child_gate_prevents_training_allocation(database_url: str) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    job = await _create_owned_job(database, repository)
    try:
        async with database.transaction() as session:
            _ = await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            _ = await repository.request_cancel(session, job.id)
            started = await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            final = await repository.get_job(session, job.id)
            assert not started
            assert final is not None
            assert final.phase == TrainingPhase.CANCELLING.value
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_stale_runner_generation_cannot_append_metric_or_log(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    job = await _create_owned_job(database, repository)
    try:
        async with database.transaction() as session:
            assert await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
            assert await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=12345,
                start_time=67890,
            )
        reporter = _RunnerReporter(
            database,
            repository,
            job,
            tmp_path,
            asyncio.get_running_loop(),
            pid=12345,
            start_time=67890,
        )
        async with database.transaction() as session:
            current = await repository.get_job(session, job.id, lock=True)
            assert current is not None
            current.owner_generation += 1

        metric_path = tmp_path / "metrics.jsonl"
        with pytest.raises(TrainingRequestLeaseLostError):
            await asyncio.to_thread(
                reporter.metric,
                TrainingMetric(epoch=1, step=1, observed_at=datetime.now(UTC)),
            )
        assert not metric_path.exists()

        log_path = tmp_path / "logs.jsonl"
        with pytest.raises(TrainingRequestLeaseLostError):
            await reporter.log("info", "safe log")
        assert not log_path.exists()
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()
