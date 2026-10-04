from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import delete, select

from gods_watching.contracts.training import TrainingConfig
from gods_watching.storage import Database
from gods_watching.training.models import TrainingRequest, TrainingRequestKind, TrainingRequestPhase
from gods_watching.training.repository import (
    TrainingRepository,
    TrainingRequestConflictError,
)

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


@pytest.mark.anyio
async def test_duplicate_training_request_is_idempotent_and_input_bound(database_url: str) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    request_id = uuid4()
    try:
        async with database.transaction() as session:
            first = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.SUBMIT,
                TrainingConfig(),
                dataset=_DATASET,
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
            )
            duplicate = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.SUBMIT,
                TrainingConfig(),
                dataset=_DATASET,
                expires_at=datetime.now(UTC) + timedelta(seconds=30),
            )
            assert duplicate.id == first.id
            with pytest.raises(TrainingRequestConflictError):
                _ = await repository.create_request(
                    session,
                    request_id,
                    TrainingRequestKind.SUBMIT,
                    TrainingConfig(epochs=31),
                    dataset=_DATASET,
                    expires_at=datetime.now(UTC) + timedelta(seconds=30),
                )
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id == request_id)
            )
        await database.close()


@pytest.mark.anyio
async def test_expired_request_cannot_be_accepted_after_supervisor_timeout(
    database_url: str,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    request_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with database.transaction() as session:
            _ = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.SUBMIT,
                TrainingConfig(),
                dataset=_DATASET,
                expires_at=now + timedelta(seconds=5),
            )
        async with database.transaction() as session:
            claimed = await repository.claim_next_request(
                session,
                owner="supervisor-a",
                now=now,
                lease_until=now + timedelta(seconds=10),
            )
            assert claimed is not None
            lease_generation = claimed.lease_generation
        async with database.transaction() as session:
            expired = await repository.expire_request(
                session,
                request_id,
                now=now + timedelta(seconds=6),
            )
            assert expired
        async with database.transaction() as session:
            resolved = await repository.resolve_request(
                session,
                request_id,
                owner="supervisor-a",
                lease_generation=lease_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response={"job_id": str(uuid4())},
                job_id=uuid4(),
                now=now + timedelta(seconds=7),
            )
            assert not resolved
            request = await repository.get_request(session, request_id)
            assert request is not None
            assert request.phase == TrainingRequestPhase.EXPIRED.value
            assert request.job_id is None
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id == request_id)
            )
        await database.close()


@pytest.mark.anyio
async def test_request_lease_generation_fences_an_old_supervisor(database_url: str) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    request_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with database.transaction() as session:
            _ = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.PREFLIGHT,
                TrainingConfig(),
                expires_at=now + timedelta(seconds=60),
            )
        async with database.transaction() as session:
            first_lease = await repository.claim_next_request(
                session,
                owner="supervisor-a",
                now=now,
                lease_until=now + timedelta(seconds=2),
            )
            assert first_lease is not None
            first_generation = first_lease.lease_generation
        async with database.transaction() as session:
            second_lease = await repository.claim_next_request(
                session,
                owner="supervisor-b",
                now=now + timedelta(seconds=3),
                lease_until=now + timedelta(seconds=10),
            )
            assert second_lease is not None
            assert second_lease.lease_generation == first_generation + 1
        async with database.transaction() as session:
            stale_result = await repository.resolve_request(
                session,
                request_id,
                owner="supervisor-a",
                lease_generation=first_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response={"admitted": False},
                now=now + timedelta(seconds=4),
            )
            assert not stale_result
            current_result = await repository.resolve_request(
                session,
                request_id,
                owner="supervisor-b",
                lease_generation=second_lease.lease_generation,
                phase=TrainingRequestPhase.ACCEPTED,
                response={"admitted": True},
                now=now + timedelta(seconds=4),
            )
            assert current_result
            request = await repository.get_request(session, request_id)
            assert request is not None
            assert request.phase == TrainingRequestPhase.ACCEPTED.value
            assert request.response_snapshot == {"admitted": True}
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id == request_id)
            )
        await database.close()


@pytest.mark.anyio
async def test_resolution_rechecks_database_time_after_waiting_for_request_lock(
    database_url: str,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    request_id = uuid4()
    now = datetime.now(UTC)
    try:
        async with database.transaction() as session:
            _ = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.PREFLIGHT,
                TrainingConfig(),
                expires_at=now + timedelta(seconds=1),
            )
        async with database.transaction() as session:
            claimed = await repository.claim_next_request(
                session,
                owner="supervisor-a",
                now=now,
                lease_until=now + timedelta(seconds=5),
            )
            assert claimed is not None
            generation = claimed.lease_generation

        async with database.session_factory() as blocker, blocker.begin():
            _ = await blocker.scalar(
                select(TrainingRequest)
                .where(TrainingRequest.request_id == request_id)
                .with_for_update()
            )
            resolver = asyncio.create_task(
                _resolve_preflight(database, repository, request_id, generation)
            )
            await asyncio.sleep(1.2)

        assert await resolver is False
        async with database.transaction() as session:
            request = await repository.get_request(session, request_id)
            assert request is not None
            assert request.phase == TrainingRequestPhase.PENDING.value
            assert request.job_id is None
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id == request_id)
            )
        await database.close()


async def _resolve_preflight(
    database: Database,
    repository: TrainingRepository,
    request_id: UUID,
    generation: int,
) -> bool:
    async with database.transaction() as session:
        return await repository.resolve_request(
            session,
            request_id,
            owner="supervisor-a",
            lease_generation=generation,
            phase=TrainingRequestPhase.ACCEPTED,
            response={"admitted": True},
        )
