from __future__ import annotations

# ruff: noqa: TRY003, EM101
import asyncio
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import delete

from gods_watching.contracts.training import TrainingConfig
from gods_watching.storage import Database
from gods_watching.training.engine import TrainingResult
from gods_watching.training.evaluation import EvaluationCancelledError
from gods_watching.training.models import TrainingExecutionSlot, TrainingJob, TrainingPhase
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.runner import run_training_child
from gods_watching.training.supervisor import process_start_time

if TYPE_CHECKING:
    from pathlib import Path
    from threading import Event

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
async def test_runner_maps_cooperative_evaluation_cancel_to_cancelled(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    training_root = tmp_path / "training"
    loop = asyncio.get_running_loop()
    pid = os.getpid()
    start_time = process_start_time(pid)
    request_id = uuid4()

    async with database.transaction() as session:
        job = await repository.create(session, request_id, TrainingConfig(), _DATASET)
        job = await repository.claim_job_owner(
            session,
            job.id,
            expected_generation=job.owner_generation,
        )
        assert await repository.set_child_identity(
            session,
            job.id,
            owner_generation=job.owner_generation,
            pid=pid,
            start_time=start_time,
        )

    async def cancel_durably() -> None:
        async with database.transaction() as session:
            _ = await repository.request_cancel(session, job.id)

    def train_to_best_checkpoint(*_args: object, **_kwargs: object) -> TrainingResult:
        return TrainingResult(
            completed=True,
            cancelled=False,
            epochs_completed=1,
            optimizer_steps=2,
            best_metric=0.5,
            best_checkpoint=training_root / "jobs" / str(job.id) / "best.pt",
        )

    def cancel_during_evaluation(
        _snapshot: object,
        _paths: object,
        _checkpoint: Path,
        *,
        cancellation: Event,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(cancel_durably(), loop)
        future.result(timeout=5)
        cancellation.set()
        raise EvaluationCancelledError("final evaluation cancelled")

    monkeypatch.setenv("GW_DATABASE_URL", database_url)
    monkeypatch.setenv("GW_TRAINING_DATASET_ROOT", str(dataset_root))
    monkeypatch.setenv("GW_TRAINING_TRAINING_ROOT", str(training_root))
    monkeypatch.setattr("gods_watching.training.engine.run_training", train_to_best_checkpoint)
    monkeypatch.setattr(
        "gods_watching.training.evaluation.evaluate_best_checkpoint",
        cancel_during_evaluation,
    )

    try:
        exit_code = await run_training_child(job.id, job.owner_generation)
        if exit_code != 0:
            async with database.transaction() as session:
                failed = await repository.get_job(session, job.id)
                details = None if failed is None else (failed.phase, failed.error)
            pytest.fail(
                f"runner returned {exit_code} before clean evaluation cancellation: {details}"
            )

        async with database.transaction() as session:
            cancelled = await repository.finish_child_exit(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=pid,
                start_time=start_time,
                exit_code=exit_code,
                now=datetime.now(UTC),
            )
            slot = await session.get(TrainingExecutionSlot, True)
            assert cancelled is not None
            assert cancelled.phase == TrainingPhase.CANCELLED.value
            assert cancelled.error is None
            assert slot is not None
            assert slot.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None and slot.active_job_id == job.id:
                slot.active_job_id = None
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job.id))
        await database.close()
