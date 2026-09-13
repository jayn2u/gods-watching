"""Drive the real pipeline worker process through retention and crash recovery."""

import os
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal
from uuid import UUID, uuid4

import anyio
from anyio.abc import Process
from cryptography.fernet import Fernet
from sqlalchemy import func, select, update

from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.search import BrowseSearchRequest
from gods_watching.contracts.settings import SettingsPatchRequest
from gods_watching.retention import RetentionService
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.search.errors import CropChangedDuringReadError, CropUnavailableError
from gods_watching.settings import SettingsService
from gods_watching.storage import (
    Appearance,
    CredentialCipher,
    CropObjectStore,
    Database,
    StorageRepository,
)

from .task13_errors import Task13ExecutionError
from .task13_models import CrashRecoveryEvidence, RetentionEvidence, Task13DriverEvidence

_RETENTION_PUBLISHED_TARGET: Final = 8
_CRASH_PUBLISHED_TARGET: Final = 6
_PUBLISH_TIMEOUT_SECONDS: Final = 150.0
_SWEEP_TIMEOUT_SECONDS: Final = 40.0
_SETTLE_SECONDS: Final = 12.0
_SUPPRESSION_OBSERVATION_SECONDS: Final = 6.0
_RECOVERY_TIMEOUT_SECONDS: Final = 45.0
_STOP_GRACE_SECONDS: Final = 30.0
_POLL_SECONDS: Final = 0.5
_AGE_BACKDATE_COUNT: Final = 3
_BACKDATE_DAYS: Final = 3


@dataclass(frozen=True, slots=True)
class _DriverSettings:
    database_url: str
    triton_url: str
    rtsp_host: str
    crop_root: Path
    cipher_key: str
    worker_log: Path


@dataclass(frozen=True, slots=True)
class _VisibleRow:
    appearance_id: UUID
    first_seen: datetime
    ended_at: datetime | None
    crop_object_key: str


@dataclass(frozen=True, slots=True)
class _PlantedPaths:
    files: tuple[tuple[Path, bytes], ...]
    link: Path


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise Task13ExecutionError(detail=f"required environment variable is missing: {name}")
    return value


async def _start_worker(settings: _DriverSettings) -> Process:
    environment = {
        **os.environ,
        "GW_DATABASE_URL": settings.database_url,
        "GW_TRITON_GRPC_URL": settings.triton_url,
        "GW_CROPS_ROOT": str(settings.crop_root),
        "GW_CAMERA_CIPHER_KEY": settings.cipher_key,
        "GW_WORKER_POLL_SECONDS": "0.5",
    }
    with settings.worker_log.open("ab") as log:
        return await anyio.open_process(
            [sys.executable, "-m", "gods_watching.cli", "worker"],
            env=environment,
            stdout=log,
            stderr=log,
        )


async def _stop_worker(process: Process, signum: signal.Signals) -> int | None:
    if process.returncode is None:
        process.send_signal(signum)
        with anyio.move_on_after(_STOP_GRACE_SECONDS):
            _ = await process.wait()
    if process.returncode is None:
        process.kill()
        _ = await process.wait()
        return None
    return process.returncode


async def _kill_worker(process: Process) -> int | None:
    process.kill()
    _ = await process.wait()
    return process.returncode


async def _wait_until(predicate: Callable[[], Awaitable[bool]], deadline_seconds: float) -> bool:
    with anyio.move_on_after(deadline_seconds):
        while True:
            if await predicate():
                return True
            await anyio.sleep(_POLL_SECONDS)
    return False


async def _visible(database: Database) -> list[_VisibleRow]:
    async with database.transaction() as session:
        rows = (
            await session.execute(
                select(
                    Appearance.id,
                    Appearance.first_seen,
                    Appearance.ended_at,
                    Appearance.crop_object_key,
                )
                .where(Appearance.tombstoned_at.is_(None))
                .order_by(Appearance.first_seen, Appearance.id)
            )
        ).tuples()
        return [_VisibleRow(*row) for row in rows]


async def _visible_ids(database: Database, ids: Sequence[UUID]) -> set[UUID]:
    if not ids:
        return set()
    async with database.transaction() as session:
        return set(
            await session.scalars(
                select(Appearance.id).where(
                    Appearance.id.in_(ids), Appearance.tombstoned_at.is_(None)
                )
            )
        )


async def _row_count(database: Database, ids: Sequence[UUID]) -> int:
    async with database.transaction() as session:
        count = await session.scalar(
            select(func.count()).select_from(Appearance).where(Appearance.id.in_(ids))
        )
    return int(count or 0)


async def _wait_published(database: Database, target: int) -> list[_VisibleRow]:
    async def enough() -> bool:
        return len(await _visible(database)) >= target

    if not await _wait_until(enough, _PUBLISH_TIMEOUT_SECONDS):
        detail = f"real worker did not publish {target} appearances"
        raise Task13ExecutionError(detail=detail)
    return await _visible(database)


async def _create_cameras(database: Database, cameras: CameraService, rtsp_host: str) -> None:
    async with database.transaction() as session:
        for index in (1, 2):
            request = CameraCreateRequest.model_validate(
                {
                    "name": f"task13 camera {index} {uuid4().hex[:8]}",
                    "source_url": f"rtsp://{rtsp_host}:8554/camera-{index}",
                }
            )
            _ = await cameras.create(session, request)


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


async def _update_settings(database: Database, patch: SettingsPatchRequest) -> None:
    async with database.transaction() as session:
        _ = await SettingsService().update(session, patch)


async def _age_eviction(
    database: Database, crop_root: Path, published: list[_VisibleRow]
) -> tuple[int, int, bool]:
    ended = [row for row in published if row.ended_at is not None][:_AGE_BACKDATE_COUNT]
    if not ended:
        raise Task13ExecutionError(detail="no ended appearances were available to age out")
    ids = [row.appearance_id for row in ended]
    keys = [row.crop_object_key for row in ended]
    old = datetime.now(UTC) - timedelta(days=_BACKDATE_DAYS)
    async with database.transaction() as session:
        _ = await session.execute(
            update(Appearance)
            .where(Appearance.id.in_(ids))
            .values(first_seen=old, last_seen=old, ended_at=old)
        )
        _ = await SettingsService().update(session, SettingsPatchRequest(retention_days=1))

    async def aged_out() -> bool:
        gone = not await _visible_ids(database, ids)
        return gone and not any((crop_root / key).exists() for key in keys)

    _ = await _wait_until(aged_out, _SWEEP_TIMEOUT_SECONDS)
    evicted = len(ids) - len(await _visible_ids(database, ids))
    crops_removed = not any((crop_root / key).exists() for key in keys)
    return len(ids), evicted, crops_removed


async def _run_retention(settings: _DriverSettings) -> RetentionEvidence:
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    crop_store = CropObjectStore(settings.crop_root)
    accounting = RetentionService(database=database, storage=storage, crop_store=crop_store)
    worker: Process | None = None
    try:
        cameras = CameraService(CameraRepository(storage))
        await _create_cameras(database, cameras, settings.rtsp_host)
        planted = _plant_unrelated_paths(settings.crop_root)
        worker = await _start_worker(settings)
        published = await _wait_published(database, _RETENTION_PUBLISHED_TARGET)
        backdated, age_evicted, age_crops_removed = await _age_eviction(
            database, settings.crop_root, published
        )

        await _update_settings(database, SettingsPatchRequest(retention_days=7))
        snapshot = await _visible(database)
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
        await _update_settings(database, SettingsPatchRequest(quota_bytes=quota_bytes))

        async def any_evicted() -> bool:
            return len(await _visible_ids(database, snapshot_ids)) < len(snapshot_ids)

        _ = await _wait_until(any_evicted, _SWEEP_TIMEOUT_SECONDS)
        await anyio.sleep(_SETTLE_SECONDS)
        remaining = await _visible_ids(database, snapshot_ids)
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
        republished = await _visible_ids(database, active_victims)

        after = await accounting.accounting()
        within_budget = after.managed_bytes <= after.quota_bytes or not await _visible(database)
        checked, readable = await _search_references_readable(database, crop_store)
        untouched = _planted_untouched(planted)

        exit_code = await _stop_worker(worker, signal.SIGTERM)
        worker = None
        open_after_stop = [row for row in await _visible(database) if row.ended_at is None]
        return RetentionEvidence(
            published_before=len(published),
            age_backdated=backdated,
            age_evicted=age_evicted,
            age_crops_removed=age_crops_removed,
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
                _ = await _kill_worker(worker)
            await database.close()


async def _run_crash(settings: _DriverSettings) -> CrashRecoveryEvidence:
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    crop_store = CropObjectStore(settings.crop_root)
    worker: Process | None = None
    try:
        cameras = CameraService(CameraRepository(storage))
        await _create_cameras(database, cameras, settings.rtsp_host)
        worker = await _start_worker(settings)
        published = await _wait_published(database, _CRASH_PUBLISHED_TARGET)
        killed_exit_code = await _kill_worker(worker)
        worker = None

        orphan = crop_store.write(b"task13-unreferenced-orphan")
        temporary = settings.crop_root / orphan.object_key.rsplit("/", 1)[0] / f".{uuid4()}.tmp"
        _ = temporary.write_bytes(b"task13-interrupted-write")
        snapshot = await _visible(database)
        victim = snapshot[0]
        async with database.transaction() as session:
            await storage.tombstone_appearance(session, victim.appearance_id)
        victim_crop = settings.crop_root / victim.crop_object_key
        snapshot_ids = {row.appearance_id for row in snapshot}

        worker = await _start_worker(settings)

        async def recovered() -> bool:
            return (
                not (settings.crop_root / orphan.object_key).exists()
                and not temporary.exists()
                and not victim_crop.exists()
                and await _row_count(database, [victim.appearance_id]) == 0
            )

        _ = await _wait_until(recovered, _RECOVERY_TIMEOUT_SECONDS)
        orphan_removed = not (settings.crop_root / orphan.object_key).exists()
        temporary_removed = not temporary.exists()
        victim_unlinked = not victim_crop.exists()
        victim_finalized = await _row_count(database, [victim.appearance_id]) == 0

        async def resumed() -> bool:
            return any(row.appearance_id not in snapshot_ids for row in await _visible(database))

        publishing_resumed = await _wait_until(resumed, _PUBLISH_TIMEOUT_SECONDS)
        current = await _visible(database)
        referenced_intact = all(
            (settings.crop_root / row.crop_object_key).is_file()
            for row in current
            if row.appearance_id in snapshot_ids
        )
        exit_code = await _stop_worker(worker, signal.SIGTERM)
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
                _ = await _kill_worker(worker)
            await database.close()


async def _drive(mode: Literal["retention", "retention-crash"]) -> None:
    settings = _DriverSettings(
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
