from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from PIL import Image
from sqlalchemy import delete, update

from gods_watching.contracts.training import TrainingConfig, TrainingJobSubmitRequest
from gods_watching.storage import Database
from gods_watching.training.dataset import validate_cuhk

if TYPE_CHECKING:
    from pathlib import Path
from gods_watching.training.models import (
    TrainingExecutionSlot,
    TrainingJob,
    TrainingRequest,
    TrainingRequestKind,
    TrainingRequestPhase,
)
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.service import TrainingService
from gods_watching.training.settings import TrainingSettings


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "CUHK-PEDES"
    records = [
        {"split": "train", "captions": ["one a", "one b"], "file_path": "train/a.png", "id": 1},
        {"split": "train", "captions": ["two a", "two b"], "file_path": "train/b.png", "id": 2},
        {"split": "val", "captions": ["val a", "val b"], "file_path": "val/a.png", "id": 3},
        {"split": "test", "captions": ["test a", "test b"], "file_path": "test/a.png", "id": 4},
    ]
    for row in records:
        path = root / "imgs" / str(row["file_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (3, 2), (32, 64, 96)).save(path, format="PNG")
    _ = (root / "reid_raw.json").write_text(json.dumps(records), encoding="utf-8")
    return root


async def _accept_one_request(
    database: Database,
    repository: TrainingRepository,
    *,
    expected_kind: TrainingRequestKind,
) -> UUID:
    deadline = asyncio.get_running_loop().time() + 5
    while asyncio.get_running_loop().time() < deadline:
        now = datetime.now(UTC)
        async with database.transaction() as session:
            request = await repository.claim_next_request(
                session,
                owner="integration-supervisor",
                now=now,
                lease_until=now + timedelta(seconds=4),
            )
            if request is None:
                await asyncio.sleep(0.01)
                continue
            assert request.kind == expected_kind.value
            if expected_kind == TrainingRequestKind.SUBMIT:
                config = TrainingConfig.model_validate(request.config_snapshot)
                assert request.dataset_snapshot is not None
                job = await repository.create(
                    session,
                    request.request_id,
                    config,
                    request.dataset_snapshot,
                )
                job = await repository.claim_job_owner(
                    session,
                    job.id,
                    expected_generation=job.owner_generation,
                )
            else:
                assert request.parent_job_id is not None
                prior = await repository.get_job(session, request.parent_job_id)
                assert prior is not None
                job = await repository.resume_interrupted(
                    session,
                    prior.id,
                    expected_generation=prior.owner_generation,
                )
            resolved = await repository.resolve_request(
                session,
                request.request_id,
                owner="integration-supervisor",
                lease_generation=request.lease_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response={"job_id": str(job.id)},
                job_id=job.id,
            )
            assert resolved
            return job.id
    pytest.fail("TrainingService did not commit a supervisor request before waiting")


async def _delete_job_and_requests(
    database: Database,
    request_ids: tuple[UUID, ...],
    job_ids: tuple[UUID, ...],
) -> None:
    async with database.transaction() as session:
        if request_ids:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id.in_(request_ids))
            )
        if job_ids:
            _ = await session.execute(
                update(TrainingExecutionSlot)
                .where(TrainingExecutionSlot.active_job_id.in_(job_ids))
                .values(active_job_id=None)
            )
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id.in_(job_ids)))


@pytest.mark.anyio
async def test_real_service_commits_submit_before_supervisor_accepts(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    service = TrainingService(
        database,
        TrainingSettings(
            dataset_root=_dataset_root(tmp_path),
            request_timeout_seconds=4,
            request_poll_interval_seconds=0.01,
            request_lease_seconds=5,
        ),
    )
    request_id = uuid4()
    supervisor = asyncio.create_task(
        _accept_one_request(
            database,
            repository,
            expected_kind=TrainingRequestKind.SUBMIT,
        )
    )
    job_ids: tuple[UUID, ...] = ()
    try:
        response = await service.submit(
            TrainingJobSubmitRequest(
                request_id=request_id,
                config=TrainingConfig(micro_batch_size=2),
            )
        )
        job_id = await supervisor
        job_ids = (job_id,)
        assert response.id == job_id
        assert response.phase == "starting"
        assert response.attempts == 1
        assert "first description" not in repr(response.dataset)
    finally:
        if not supervisor.done():
            _ = supervisor.cancel()
        await _delete_job_and_requests(database, (request_id,), job_ids)
        await database.close()


@pytest.mark.anyio
async def test_real_service_resume_replay_does_not_increment_generation_twice(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    service = TrainingService(
        database,
        TrainingSettings(
            dataset_root=_dataset_root(tmp_path),
            request_timeout_seconds=4,
            request_poll_interval_seconds=0.01,
            request_lease_seconds=5,
        ),
    )
    request_id = uuid4()
    job_id: UUID
    try:
        manifest = await asyncio.to_thread(
            validate_cuhk,
            _dataset_root(tmp_path),
        )
        async with database.transaction() as session:
            job = await repository.create(
                session,
                uuid4(),
                TrainingConfig(micro_batch_size=2),
                manifest.public_snapshot(),
            )
            job.checkpoint_path = str(tmp_path / "checkpoint.pt")
            _ = await repository.transition(
                session,
                job.id,
                "starting",
                "interrupted",
            )
            job_id = job.id

        supervisor = asyncio.create_task(
            _accept_one_request(
                database,
                repository,
                expected_kind=TrainingRequestKind.RESUME,
            )
        )
        first = await service.resume(job_id, request_id)
        accepted_id = await supervisor
        replay = await service.resume(job_id, request_id)

        assert first.id == accepted_id == replay.id == job_id
        assert first.owner_generation == replay.owner_generation == 1
        assert first.attempts == replay.attempts == 1
        await _delete_job_and_requests(database, (request_id,), (job_id,))
    finally:
        await database.close()


@pytest.mark.anyio
async def test_cancel_race_is_idempotent_under_the_singleton_slot_lock(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    service = TrainingService(
        database,
        TrainingSettings(dataset_root=_dataset_root(tmp_path)),
    )
    manifest = await asyncio.to_thread(validate_cuhk, _dataset_root(tmp_path))
    request_id = uuid4()
    async with database.transaction() as session:
        job = await repository.create(
            session,
            request_id,
            TrainingConfig(micro_batch_size=2),
            manifest.public_snapshot(),
        )
        job_id = job.id

    try:
        first, second = await asyncio.gather(
            service.cancel(job_id, uuid4()),
            service.cancel(job_id, uuid4()),
        )
        assert first.phase == second.phase == "cancelling"
        assert first.cancel_requested is True
        assert second.cancel_requested is True
    finally:
        await _delete_job_and_requests(database, (), (job_id,))
        await database.close()
