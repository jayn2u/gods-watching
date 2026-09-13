"""Drive the real pipeline worker process through retention and crash recovery."""

import os
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal
from uuid import UUID, uuid4

import anyio
from cryptography.fernet import Fernet

from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.search import BrowseSearchRequest
from gods_watching.contracts.settings import SettingsPatchRequest
from gods_watching.retention import RetentionService
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.search.errors import CropChangedDuringReadError, CropUnavailableError
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

from .pipeline_worker_runtime import (
    WorkerStackSettings,
    age_out_ended,
    create_cameras,
    kill_worker,
    row_count,
    start_worker,
    stop_worker,
    update_settings,
    visible_ids,
    visible_rows,
    wait_published,
    wait_until,
)
from .task13_errors import Task13ExecutionError
from .task13_models import CrashRecoveryEvidence, RetentionEvidence, Task13DriverEvidence

if TYPE_CHECKING:
    from anyio.abc import Process

_RETENTION_PUBLISHED_TARGET: Final = 8
_CRASH_PUBLISHED_TARGET: Final = 6
_PUBLISH_TIMEOUT_SECONDS: Final = 150.0
_SWEEP_TIMEOUT_SECONDS: Final = 40.0
_SETTLE_SECONDS: Final = 12.0
_SUPPRESSION_OBSERVATION_SECONDS: Final = 6.0
_RECOVERY_TIMEOUT_SECONDS: Final = 45.0
_AGE_BACKDATE_COUNT: Final = 3
_CAMERA_LABEL: Final = "task13"


@dataclass(frozen=True, slots=True)
class _PlantedPaths:
    files: tuple[tuple[Path, bytes], ...]
    link: Path


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise Task13ExecutionError(detail=f"required environment variable is missing: {name}")
    return value


def _plant_unrelated_paths(crop_root: Path) -> _PlantedPaths:
    outside = crop_root.parent / "task13-outside"
    outside.mkdir(parents=True, exist_ok=True)
    target = outside / "keep.jpg"
    _ = target.write_bytes(b"outside-the-crop-root")
    sibling = crop_root.parent / "task13-unrelated.txt"
    _ = sibling.write_bytes(b"unrelated-sibling")
    crop_root.mkdir(parents=True, exist_ok=True)
    note = crop_root / "operator-notes.txt"
    _ = note.write_bytes(b"not-a-generated-object")
    link_parent = crop_root / "00" / "00"
    link_parent.mkdir(parents=True, exist_ok=True)
    link = link_parent / "00000000-0000-4000-8000-000000000000.jpg"
    link.symlink_to(target)
    return _PlantedPaths(
        files=(
            (target, b"outside-the-crop-root"),
            (sibling, b"unrelated-sibling"),
            (note, b"not-a-generated-object"),
        ),
        link=link,
    )


def _planted_untouched(planted: _PlantedPaths) -> bool:
    files_intact = all(
        path.is_file() and path.read_bytes() == payload for path, payload in planted.files
    )
    return files_intact and planted.link.is_symlink()


async def _search_references_readable(
    database: Database, crop_store: CropObjectStore
) -> tuple[int, bool]:
    repository = SearchRepository()
    lookup = AppearanceLookupService(repository, crop_store)
    async with database.transaction() as session:
        response = await SearchService(repository).search(
            session, BrowseSearchRequest(mode="browse", limit=100)
        )
    readable = True
    for result in response.results:
        appearance_id = UUID(str(result.appearance_id))
        try:
            async with database.transaction() as session:
                _ = await lookup.get_crop(session, appearance_id)
        except (CropUnavailableError, CropChangedDuringReadError):
            # Eviction or an upgrade between search and read is correct; a row that is
            # still visible while its current crop is missing is a broken reference.
            async with database.transaction() as session:
                current = await repository.get_visible(session, appearance_id)
            current_key = None if current is None else current[0].crop_object_key
            if current_key is not None and not (crop_store.root / current_key).is_file():
                readable = False
    return len(response.results), readable


async def _run_retention(settings: WorkerStackSettings) -> RetentionEvidence:
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    crop_store = CropObjectStore(settings.crop_root)
    accounting = RetentionService(database=database, storage=storage, crop_store=crop_store)
    worker: Process | None = None
    try:
        cameras = CameraService(CameraRepository(storage))
        _ = await create_cameras(database, cameras, settings.rtsp_host, label=_CAMERA_LABEL)
        planted = _plant_unrelated_paths(settings.crop_root)
        worker = await start_worker(settings)
        published = await wait_published(
            database, _RETENTION_PUBLISHED_TARGET, _PUBLISH_TIMEOUT_SECONDS
        )
        aged = await age_out_ended(
            database,
            settings.crop_root,
            count=_AGE_BACKDATE_COUNT,
            deadline_seconds=_SWEEP_TIMEOUT_SECONDS + _PUBLISH_TIMEOUT_SECONDS,
        )

        await update_settings(database, SettingsPatchRequest(retention_days=7))
        snapshot = await visible_rows(database)
        snapshot_ids = [row.appearance_id for row in snapshot]
        active_at_snapshot = {row.appearance_id for row in snapshot if row.ended_at is None}
        before = await accounting.accounting()
        quota_bytes = max(
            1,
            int(
                (before.relation_bytes + before.pending_gc_bytes + before.physical_crop_bytes // 2)
                / before.cleanup_threshold
            ),
        )
        await update_settings(database, SettingsPatchRequest(quota_bytes=quota_bytes))

        async def any_evicted() -> bool:
            return len(await visible_ids(database, snapshot_ids)) < len(snapshot_ids)

        _ = await wait_until(any_evicted, _SWEEP_TIMEOUT_SECONDS)
        await anyio.sleep(_SETTLE_SECONDS)
        remaining = await visible_ids(database, snapshot_ids)
        evicted_positions = [
            index for index, item in enumerate(snapshot_ids) if item not in remaining
        ]
        remaining_positions = [
            index for index, item in enumerate(snapshot_ids) if item in remaining
        ]
        oldest_first = bool(evicted_positions) and (
            not remaining_positions or max(evicted_positions) < min(remaining_positions)
        )
        active_victims = [
            item for item in snapshot_ids if item not in remaining and item in active_at_snapshot
        ]
        await anyio.sleep(_SUPPRESSION_OBSERVATION_SECONDS)
        republished = await visible_ids(database, active_victims)

        after = await accounting.accounting()
        within_budget = after.managed_bytes <= after.quota_bytes or not await visible_rows(database)
        checked, readable = await _search_references_readable(database, crop_store)
        untouched = _planted_untouched(planted)

        exit_code = await stop_worker(worker, signal.SIGTERM)
        worker = None
        open_after_stop = [row for row in await visible_rows(database) if row.ended_at is None]
        return RetentionEvidence(
            published_before=len(published),
            age_backdated=len(aged.appearance_ids),
            age_evicted=aged.evicted,
            age_crops_removed=aged.crops_removed,
            quota_bytes=quota_bytes,
            quota_snapshot=len(snapshot_ids),
            quota_evicted=len(evicted_positions),
            quota_oldest_first=oldest_first,
            active_victims=len(active_victims),
            active_victims_not_republished=not republished,
            managed_bytes_after=after.managed_bytes,
            threshold_bytes=after.threshold_bytes,
            within_budget_or_full=within_budget,
            search_results_checked=checked,
            search_crops_readable=readable,
            unrelated_paths_untouched=untouched,
            worker_exit_code=exit_code,
            open_tracks_ended_on_stop=not open_after_stop,
        )
    finally:
        with anyio.CancelScope(shield=True):
            if worker is not None and worker.returncode is None:
                _ = await kill_worker(worker)
            await database.close()


async def _run_crash(settings: WorkerStackSettings) -> CrashRecoveryEvidence:
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    crop_store = CropObjectStore(settings.crop_root)
    worker: Process | None = None
    try:
        cameras = CameraService(CameraRepository(storage))
        _ = await create_cameras(database, cameras, settings.rtsp_host, label=_CAMERA_LABEL)
        worker = await start_worker(settings)
        published = await wait_published(
            database, _CRASH_PUBLISHED_TARGET, _PUBLISH_TIMEOUT_SECONDS
        )
        killed_exit_code = await kill_worker(worker)
        worker = None

        orphan = crop_store.write(b"task13-unreferenced-orphan")
        temporary = settings.crop_root / orphan.object_key.rsplit("/", 1)[0] / f".{uuid4()}.tmp"
        _ = temporary.write_bytes(b"task13-interrupted-write")
        snapshot = await visible_rows(database)
        victim = snapshot[0]
        async with database.transaction() as session:
            await storage.tombstone_appearance(session, victim.appearance_id)
        victim_crop = settings.crop_root / victim.crop_object_key
        snapshot_ids = {row.appearance_id for row in snapshot}

        worker = await start_worker(settings)

        async def recovered() -> bool:
            return (
                not (settings.crop_root / orphan.object_key).exists()
                and not temporary.exists()
                and not victim_crop.exists()
                and await row_count(database, [victim.appearance_id]) == 0
            )

        _ = await wait_until(recovered, _RECOVERY_TIMEOUT_SECONDS)
        orphan_removed = not (settings.crop_root / orphan.object_key).exists()
        temporary_removed = not temporary.exists()
        victim_unlinked = not victim_crop.exists()
        victim_finalized = await row_count(database, [victim.appearance_id]) == 0

        async def resumed() -> bool:
            return any(
                row.appearance_id not in snapshot_ids for row in await visible_rows(database)
            )

        publishing_resumed = await wait_until(resumed, _PUBLISH_TIMEOUT_SECONDS)
        current = await visible_rows(database)
        referenced_intact = all(
            (settings.crop_root / row.crop_object_key).is_file()
            for row in current
            if row.appearance_id in snapshot_ids
        )
        exit_code = await stop_worker(worker, signal.SIGTERM)
        worker = None
        return CrashRecoveryEvidence(
            published_before_kill=len(published),
            killed_exit_code=killed_exit_code,
            orphan_removed=orphan_removed,
            temporary_removed=temporary_removed,
            tombstoned_crop_unlinked=victim_unlinked,
            tombstoned_row_finalized=victim_finalized,
            referenced_crops_intact=referenced_intact,
            published_after_restart=len(current),
            publishing_resumed=publishing_resumed,
            worker_exit_code=exit_code,
        )
    finally:
        with anyio.CancelScope(shield=True):
            if worker is not None and worker.returncode is None:
                _ = await kill_worker(worker)
            await database.close()


async def _drive(mode: Literal["retention", "retention-crash"]) -> None:
    settings = WorkerStackSettings(
        database_url=_env("GW_TASK13_DATABASE_URL"),
        triton_url=_env("GW_TASK13_TRITON_URL"),
        rtsp_host=_env("GW_TASK13_RTSP_HOST"),
        crop_root=Path(_env("GW_TASK13_CROP_ROOT")),
        cipher_key=Fernet.generate_key().decode(),
        worker_log=Path(_env("GW_TASK13_WORKER_LOG")),
    )
    if mode == "retention":
        evidence = Task13DriverEvidence(mode=mode, retention=await _run_retention(settings))
    else:
        evidence = Task13DriverEvidence(mode=mode, crash=await _run_crash(settings))
    output = Path(_env("GW_TASK13_OUTPUT"))
    _ = output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")


async def _main() -> None:
    mode_value = _env("GW_TASK13_MODE")
    if mode_value not in ("retention", "retention-crash"):
        raise Task13ExecutionError(detail=f"unsupported Task 13 mode: {mode_value}")
    mode: Literal["retention", "retention-crash"] = mode_value
    async with anyio.create_task_group() as task_group:

        async def cancel_on_signal() -> None:
            # The scenario harness terminates this driver on deadline; cancelling lets the
            # finally blocks kill the worker process instead of orphaning it.
            with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
                async for _signum in signals:
                    task_group.cancel_scope.cancel()
                    return

        task_group.start_soon(cancel_on_signal)
        await _drive(mode)
        task_group.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(_main)
