from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import delete

from gods_watching.contracts.training import TrainingConfig
from gods_watching.storage import Database
from gods_watching.training.models import TrainingExecutionSlot, TrainingJob, TrainingPhase
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.settings import TrainingSettings
from gods_watching.training.supervisor import TrainingChild, TrainingSupervisor

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


class _Child:
    _pid: int
    _start_time: int
    exit_code: int | None
    released: bool

    def __init__(self, *, pid: int, start_time: int) -> None:
        self._pid = pid
        self._start_time = start_time
        self.exit_code = None
        self.released = False

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def start_time(self) -> int:
        return self._start_time

    def release(self) -> None:
        self.released = True

    def abort(self) -> None:
        self.exit_code = -9

    def poll(self) -> int | None:
        return self.exit_code


class _Launcher:
    child: _Child

    def __init__(self, child: _Child) -> None:
        self.child = child

    def prepare(self, job: TrainingJob) -> TrainingChild:
        _ = job
        return self.child


async def _create_job(
    database: Database,
    repository: TrainingRepository,
) -> TrainingJob:
    async with database.transaction() as session:
        job = await repository.create(
            session,
            uuid4(),
            TrainingConfig(micro_batch_size=2),
            _DATASET,
        )
        return await repository.claim_job_owner(
            session,
            job.id,
            expected_generation=job.owner_generation,
        )


@pytest.mark.anyio
async def test_terminal_phase_and_slot_release_wait_for_child_exit(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    child = _Child(pid=12345, start_time=67890)
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(training_root=tmp_path),
        child_launcher=_Launcher(child),
        repository=repository,
    )
    job = await _create_job(database, repository)
    try:
        await supervisor._launch_owned_child(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
            job
        )
        async with database.transaction() as session:
            _ = await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=child.pid,
                start_time=child.start_time,
            )
        assert child.released

        await supervisor._reconcile_children()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        async with database.transaction() as session:
            active = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert active is not None
            assert active.phase == TrainingPhase.TRAINING.value
            assert slot is not None
            assert slot.active_job_id == job.id

        child.exit_code = 23
        await supervisor._reconcile_children()  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
        async with database.transaction() as session:
            failed = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert failed is not None
            assert failed.phase == TrainingPhase.FAILED.value
            assert failed.finished_at is not None
            assert slot is not None
            assert slot.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_supervisor_startup_interrupts_a_dead_child_without_new_requests(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(
            training_root=tmp_path,
            request_poll_interval_seconds=0.01,
        ),
        child_launcher=_Launcher(_Child(pid=12345, start_time=67890)),
        repository=repository,
    )
    job = await _create_job(database, repository)
    async with database.transaction() as session:
        _ = await repository.set_child_identity(
            session,
            job.id,
            owner_generation=job.owner_generation,
            pid=2_147_483_647,
            start_time=1,
        )
    stop = asyncio.Event()
    task = asyncio.create_task(supervisor.run(stop))
    try:
        deadline = asyncio.get_running_loop().time() + 2
        phase = TrainingPhase.STARTING.value
        while asyncio.get_running_loop().time() < deadline:
            async with database.transaction() as session:
                current = await repository.get_job(session, job.id)
                assert current is not None
                phase = current.phase
                slot = await session.get(TrainingExecutionSlot, True)
            if phase == TrainingPhase.INTERRUPTED.value:
                assert slot is not None
                assert slot.active_job_id is None
                break
            await asyncio.sleep(0.01)
        assert phase == TrainingPhase.INTERRUPTED.value
    finally:
        stop.set()
        await task
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_supervisor_recovers_dead_evaluation_child_for_manual_resume(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(training_root=tmp_path),
        child_launcher=None,
        repository=repository,
    )
    job = await _create_job(database, repository)
    try:
        async with database.transaction() as session:
            assert await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_engine_staging(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )

        async with database.transaction() as session:
            await supervisor._recover_orphan_slot(session)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

        async with database.transaction() as session:
            interrupted = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert interrupted is not None
            assert interrupted.phase == TrainingPhase.INTERRUPTED.value
            assert interrupted.engine_completed_at is not None
            assert interrupted.child_pid is None
            assert slot is not None
            assert slot.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_supervisor_cancels_legacy_evaluation_without_child_identity(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(training_root=tmp_path),
        child_launcher=None,
        repository=repository,
    )
    job = await _create_job(database, repository)
    try:
        async with database.transaction() as session:
            assert await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_training_started(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_engine_staging(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            _ = await repository.request_cancel(session, job.id)
            current = await repository.get_job(session, job.id, lock=True)
            assert current is not None
            current.child_pid = None
            current.child_start_time = None

        async with database.transaction() as session:
            await supervisor._recover_orphan_slot(session)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

        async with database.transaction() as session:
            cancelled = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert cancelled is not None
            assert cancelled.phase == TrainingPhase.CANCELLED.value
            assert cancelled.cancel_requested
            assert slot is not None
            assert slot.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()


@pytest.mark.anyio
async def test_supervisor_startup_reclaims_starting_job_without_recorded_child(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(
            training_root=tmp_path,
            request_poll_interval_seconds=0.01,
        ),
        child_launcher=_Launcher(_Child(pid=12345, start_time=67890)),
        repository=repository,
    )
    job = await _create_job(database, repository)
    initial_generation = job.owner_generation
    stop = asyncio.Event()
    task = asyncio.create_task(supervisor.run(stop))
    try:
        deadline = asyncio.get_running_loop().time() + 2
        phase = TrainingPhase.STARTING.value
        while asyncio.get_running_loop().time() < deadline:
            async with database.transaction() as session:
                current = await repository.get_job(session, job.id)
                assert current is not None
                phase = current.phase
                generation = current.owner_generation
                slot = await session.get(TrainingExecutionSlot, True)
            if phase == TrainingPhase.INTERRUPTED.value:
                assert slot is not None
                assert slot.active_job_id is None
                assert generation > initial_generation
                break
            await asyncio.sleep(0.01)
        assert phase == TrainingPhase.INTERRUPTED.value
    finally:
        stop.set()
        await task
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()
