from __future__ import annotations

# ruff: noqa: EM101, TRY003
import asyncio
import json
import os
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from PIL import Image
from sqlalchemy import delete, select, update

import gods_watching.training.memory as memory_module
import gods_watching.training.supervisor as supervisor_module
from gods_watching.contracts.training import TrainingConfig
from gods_watching.storage import Database
from gods_watching.training.dataset import validate_cuhk
from gods_watching.training.memory import GIB, OPTIMIZER_RECIPE_ID, GpuSnapshot, MemoryProfile
from gods_watching.training.models import (
    TrainingExecutionSlot,
    TrainingJob,
    TrainingRequest,
    TrainingRequestKind,
    TrainingRequestPhase,
)
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.settings import TrainingSettings
from gods_watching.training.supervisor import (
    TrainingChild,
    TrainingSupervisor,
    process_start_time,
)

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

_GPU_UUID = "GPU-17913b0a-8144-5f39-7062-15265e5dca33"


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "CUHK-PEDES"
    rows = (
        ("train", 1, "train/a.png"),
        ("train", 2, "train/b.png"),
        ("val", 3, "val/a.png"),
        ("test", 4, "test/a.png"),
    )
    annotations: list[dict[str, object]] = []
    for split, person_id, relative in rows:
        image = root / "imgs" / relative
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (3, 2), (32, 64, 96)).save(image, format="PNG")
        annotations.append(
            {
                "split": split,
                "captions": [f"{split} description a", f"{split} description b"],
                "file_path": relative,
                "id": person_id,
            }
        )
    _ = (root / "reid_raw.json").write_text(json.dumps(annotations), encoding="utf-8")
    return root


class _NeverLaunch:
    def prepare(self, job: TrainingJob) -> TrainingChild:
        del job
        raise AssertionError("an orphan slot must not launch another child")


@pytest.mark.anyio
async def test_permission_denied_orphan_identity_blocks_new_job_without_creating_history(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset_root(tmp_path)
    manifest = await asyncio.to_thread(validate_cuhk, root)
    config = TrainingConfig(micro_batch_size=2)
    database = Database.connect(database_url)
    repository = TrainingRepository()
    monkeypatch.setattr(memory_module, "current_model_sha256", lambda: "c" * 64)
    profile = MemoryProfile(
        gpu_uuid=_GPU_UUID,
        gpu_total_bytes=16 * GIB,
        source_fingerprint=memory_module.current_source_fingerprint(),
        torch_version="2.7.1+cu128",
        cuda_version="12.8",
        processor_identity="openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3",
        model_sha256="c" * 64,
        mixed_precision="fp16",
        gradient_checkpointing=True,
        batch_size=2,
        optimizer_recipe=OPTIMIZER_RECIPE_ID,
        image_size=224,
        text_max_length=77,
        baseline_reserved_bytes=512 * 1024**2,
        peak_reserved_bytes=3 * GIB,
        peak_allocated_bytes=2 * GIB,
        baseline_device_used_bytes=1 * GIB,
        peak_device_used_bytes=4 * GIB,
        parameter_bytes=512 * 1024**2,
        gradient_bytes=512 * 1024**2,
        optimizer_state_bytes=1 * GIB,
        activation_workspace_bytes=512 * 1024**2,
        context_bytes=256 * 1024**2,
        calibrated_at=datetime.now(UTC),
    )
    gpu = GpuSnapshot(
        uuid=_GPU_UUID,
        total_bytes=16 * GIB,
        free_bytes=12 * GIB,
        observed_at=datetime.now(UTC),
    )
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(
            dataset_root=root,
            memory_profiles_path=tmp_path / "profiles.json",
            request_lease_seconds=5,
        ),
        child_launcher=_NeverLaunch(),
        memory_profiles_provider=lambda: (profile,),
        gpu_snapshot_provider=lambda: gpu,
    )
    active_request_id = uuid4()
    new_request_id = uuid4()
    job_id = None
    try:
        async with database.transaction() as session:
            job = await repository.create(
                session,
                active_request_id,
                config,
                manifest.public_snapshot(),
            )
            job = await repository.claim_job_owner(
                session,
                job.id,
                expected_generation=job.owner_generation,
            )
            _ = await repository.set_child_identity(
                session,
                job.id,
                owner_generation=job.owner_generation,
                pid=os.getpid(),
                start_time=process_start_time(os.getpid()),
            )
            job_id = job.id

        def denied_start_time(_pid: int, *, proc_root: Path | None = None) -> int:
            del proc_root
            raise PermissionError

        monkeypatch.setattr(supervisor_module, "process_start_time", denied_start_time)

        async with database.transaction() as session:
            _ = await repository.create_request(
                session,
                new_request_id,
                TrainingRequestKind.SUBMIT,
                config,
                dataset=manifest.public_snapshot(),
                expires_at=datetime.now(UTC) + timedelta(seconds=10),
            )

        assert await supervisor.process_one()

        async with database.transaction() as session:
            request = await repository.get_request(session, new_request_id)
            jobs = await session.scalars(select(TrainingJob))
            assert request is not None
            assert request.phase == TrainingRequestPhase.REFUSED.value
            assert request.error == "training_job_conflict"
            assert len(jobs.all()) == 1
            assert request.job_id is None
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                update(TrainingExecutionSlot)
                .where(TrainingExecutionSlot.active_job_id == job_id)
                .values(active_job_id=None)
            )
            _ = await session.execute(
                delete(TrainingRequest).where(
                    TrainingRequest.request_id.in_((active_request_id, new_request_id))
                )
            )
            if job_id is not None:
                _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job_id))
        await database.close()


@pytest.mark.anyio
async def test_memory_is_rechecked_under_slot_lock_before_job_creation(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _dataset_root(tmp_path)
    manifest = await asyncio.to_thread(validate_cuhk, root)
    config = TrainingConfig(micro_batch_size=2)
    database = Database.connect(database_url)
    repository = TrainingRepository()
    monkeypatch.setattr(memory_module, "current_model_sha256", lambda: "c" * 64)
    profile = MemoryProfile(
        gpu_uuid=_GPU_UUID,
        gpu_total_bytes=16 * GIB,
        source_fingerprint=memory_module.current_source_fingerprint(),
        torch_version="2.7.1+cu128",
        cuda_version="12.8",
        processor_identity="openai/clip-vit-base-patch16@57c216476eefef5ab752ec549e440a49ae4ae5f3",
        model_sha256="c" * 64,
        mixed_precision="fp16",
        gradient_checkpointing=True,
        batch_size=2,
        optimizer_recipe=OPTIMIZER_RECIPE_ID,
        image_size=224,
        text_max_length=77,
        baseline_reserved_bytes=512 * 1024**2,
        peak_reserved_bytes=3 * GIB,
        peak_allocated_bytes=2 * GIB,
        baseline_device_used_bytes=1 * GIB,
        peak_device_used_bytes=4 * GIB,
        parameter_bytes=512 * 1024**2,
        gradient_bytes=512 * 1024**2,
        optimizer_state_bytes=1 * GIB,
        activation_workspace_bytes=512 * 1024**2,
        context_bytes=256 * 1024**2,
        calibrated_at=datetime.now(UTC),
    )
    snapshots = iter(
        (
            GpuSnapshot(
                uuid=_GPU_UUID,
                total_bytes=16 * GIB,
                free_bytes=12 * GIB,
                observed_at=datetime.now(UTC),
            ),
            GpuSnapshot(
                uuid=_GPU_UUID,
                total_bytes=16 * GIB,
                free_bytes=5 * GIB,
                observed_at=datetime.now(UTC),
            ),
        )
    )
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(
            dataset_root=root,
            memory_profiles_path=tmp_path / "profiles.json",
            request_lease_seconds=5,
        ),
        child_launcher=_NeverLaunch(),
        memory_profiles_provider=lambda: (profile,),
        gpu_snapshot_provider=lambda: next(snapshots),
    )
    request_id = uuid4()
    try:
        async with database.transaction() as session:
            _ = await repository.create_request(
                session,
                request_id,
                TrainingRequestKind.SUBMIT,
                config,
                dataset=manifest.public_snapshot(),
                expires_at=datetime.now(UTC) + timedelta(seconds=10),
            )

        assert await supervisor.process_one()

        async with database.transaction() as session:
            request = await repository.get_request(session, request_id)
            jobs = await session.scalars(select(TrainingJob))
            assert request is not None
            assert request.phase == TrainingRequestPhase.REFUSED.value
            assert request.error == "training_memory_refused"
            assert request.response_snapshot is not None
            assert request.response_snapshot["free_bytes"] == 5 * GIB
            assert len(jobs.all()) == 0
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(TrainingRequest).where(TrainingRequest.request_id == request_id)
            )
        await database.close()


@pytest.mark.anyio
async def test_launch_failure_waits_for_slot_before_job_lock_during_cancel_race(  # noqa: PLR0915
    database_url: str,
    tmp_path: Path,
) -> None:
    root = _dataset_root(tmp_path)
    manifest = await asyncio.to_thread(validate_cuhk, root)
    database = Database.connect(database_url)
    repository = TrainingRepository()
    request_id = uuid4()
    async with database.transaction() as session:
        job = await repository.create(
            session,
            request_id,
            TrainingConfig(micro_batch_size=2),
            manifest.public_snapshot(),
        )
        job = await repository.claim_job_owner(
            session,
            job.id,
            expected_generation=job.owner_generation,
        )
        job_id = job.id

    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(dataset_root=root),
        child_launcher=None,
        repository=repository,
    )
    slot_owned_by_cancel = asyncio.Event()
    release_cancel = asyncio.Event()
    mark_locked_job = asyncio.Event()
    original_get_job = repository.get_job

    async def gated_get_job(
        session: AsyncSession,
        job_id: UUID,
        *,
        lock: bool = False,
    ) -> TrainingJob | None:
        task = asyncio.current_task()
        task_name = task.get_name() if task is not None else ""
        if task_name == "cancel":
            slot_owned_by_cancel.set()
            _ = await release_cancel.wait()
        result = await original_get_job(session, job_id, lock=lock)
        if task_name == "launch-failure" and lock:
            mark_locked_job.set()
        return result

    repository.get_job = gated_get_job

    async def cancel() -> TrainingJob:
        async with database.transaction() as session:
            return await repository.request_cancel(session, job_id)

    async def mark_start_failed() -> None:
        async with database.transaction():
            await supervisor._mark_child_start_failed(  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001
                job,
                RuntimeError("launch failed"),
            )

    cancel_task = asyncio.create_task(cancel(), name="cancel")
    failure_task: asyncio.Task[None] | None = None
    try:
        _ = await asyncio.wait_for(slot_owned_by_cancel.wait(), timeout=2)
        failure_task = asyncio.create_task(mark_start_failed(), name="launch-failure")
        await asyncio.sleep(0.1)
        assert not mark_locked_job.is_set()
        release_cancel.set()
        cancelled_job = await asyncio.wait_for(cancel_task, timeout=3)
        assert failure_task is not None
        _ = await asyncio.wait_for(failure_task, timeout=3)
        assert cancelled_job.phase == "cancelling"
        async with database.transaction() as session:
            final = await repository.get_job(session, job_id)
            assert final is not None
            assert final.phase == "cancelling"
    finally:
        release_cancel.set()
        for task in (cancel_task, failure_task):
            if task is not None and not task.done():
                _ = task.cancel()
        with suppress(asyncio.CancelledError):
            _ = await cancel_task
        if failure_task is not None:
            with suppress(asyncio.CancelledError):
                _ = await failure_task
        async with database.transaction() as session:
            _ = await session.execute(
                update(TrainingExecutionSlot)
                .where(TrainingExecutionSlot.active_job_id == job_id)
                .values(active_job_id=None)
            )
            _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job_id))
        await database.close()
