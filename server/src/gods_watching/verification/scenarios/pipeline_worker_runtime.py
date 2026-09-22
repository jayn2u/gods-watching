"""Drive the real pipeline worker process and observe its committed appearances."""

import os
import signal
import sys
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, final, override
from uuid import UUID, uuid4

import anyio
from anyio.abc import Process
from sqlalchemy import func, select, update

from gods_watching.cameras import CameraService
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.settings import SettingsPatchRequest
from gods_watching.settings import SettingsService
from gods_watching.storage import Appearance, Database

from .task10_runtime import fixture_rtsp_port

_POLL_SECONDS: Final = 0.5
_STOP_GRACE_SECONDS: Final = 30.0
_BACKDATE_DAYS: Final = 3


@final
class WorkerStackError(RuntimeError):
    """Report a worker-stack step that could not produce observations."""

    def __init__(self, *, detail: str) -> None:
        """Retain the operator-safe failure detail."""
        self.detail = detail
        super().__init__(detail)

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class WorkerStackSettings:
    """Name the owned services and paths one driver run connects the worker to."""

    database_url: str
    triton_url: str
    rtsp_host: str
    crop_root: Path
    cipher_key: str
    worker_log: Path
    rtsp_port: int = field(default_factory=fixture_rtsp_port)


@dataclass(frozen=True, slots=True)
class VisibleRow:
    """Describe one committed, non-tombstoned appearance."""

    appearance_id: UUID
    camera_id: UUID
    first_seen: datetime
    last_seen: datetime
    ended_at: datetime | None
    crop_object_key: str


@dataclass(frozen=True, slots=True)
class AgeOutResult:
    """Record ended appearances backdated past retention and what the sweep removed."""

    appearance_ids: tuple[UUID, ...]
    evicted: int
    crops_removed: bool


async def start_worker(settings: WorkerStackSettings) -> Process:
    """Start the real `gods-watching worker` command against the owned stack."""
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


async def stop_worker(process: Process, signum: signal.Signals) -> int | None:
    """Signal the worker and return its exit code, or None when it had to be killed."""
    if process.returncode is None:
        process.send_signal(signum)
        with anyio.move_on_after(_STOP_GRACE_SECONDS):
            _ = await process.wait()
    if process.returncode is None:
        process.kill()
        _ = await process.wait()
        return None
    return process.returncode


async def kill_worker(process: Process) -> int | None:
    """Kill the worker without letting it clean up, as a crash would."""
    process.kill()
    _ = await process.wait()
    return process.returncode


async def wait_until(predicate: Callable[[], Awaitable[bool]], deadline_seconds: float) -> bool:
    """Poll a predicate until it holds or the deadline passes."""
    with anyio.move_on_after(deadline_seconds):
        while True:
            if await predicate():
                return True
            await anyio.sleep(_POLL_SECONDS)
    return False


async def visible_rows(database: Database) -> list[VisibleRow]:
    """Return non-tombstoned appearances ordered by first sighting."""
    async with database.transaction() as session:
        rows = (
            await session.execute(
                select(
                    Appearance.id,
                    Appearance.camera_id,
                    Appearance.first_seen,
                    Appearance.last_seen,
                    Appearance.ended_at,
                    Appearance.crop_object_key,
                )
                .where(Appearance.tombstoned_at.is_(None))
                .order_by(Appearance.first_seen, Appearance.id)
            )
        ).tuples()
        return [VisibleRow(*row) for row in rows]


async def visible_ids(database: Database, ids: Sequence[UUID]) -> set[UUID]:
    """Return which of the given appearances are still visible."""
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


async def row_count(database: Database, ids: Sequence[UUID]) -> int:
    """Count appearance rows, visible or tombstoned, among the given ids."""
    async with database.transaction() as session:
        count = await session.scalar(
            select(func.count()).select_from(Appearance).where(Appearance.id.in_(ids))
        )
    return int(count or 0)


async def wait_published(
    database: Database, target: int, deadline_seconds: float
) -> list[VisibleRow]:
    """Wait until the worker has published at least `target` visible appearances."""

    async def enough() -> bool:
        return len(await visible_rows(database)) >= target

    if not await wait_until(enough, deadline_seconds):
        detail = f"real worker did not publish {target} appearances"
        raise WorkerStackError(detail=detail)
    return await visible_rows(database)


async def create_cameras(
    database: Database,
    cameras: CameraService,
    rtsp_host: str,
    *,
    label: str,
    rtsp_port: int | None = None,
) -> tuple[CameraId, ...]:
    """Create two detection-enabled fixture cameras through the camera service."""
    resolved_rtsp_port = fixture_rtsp_port() if rtsp_port is None else rtsp_port
    created: list[CameraId] = []
    async with database.transaction() as session:
        for index in (1, 2):
            request = CameraCreateRequest.model_validate(
                {
                    "name": f"{label} camera {index} {uuid4().hex[:8]}",
                    "source_url": f"rtsp://{rtsp_host}:{resolved_rtsp_port}/camera-{index}",
                }
            )
            mutation = await cameras.create(session, request)
            created.append(mutation.camera.camera_id)
    return tuple(created)


async def update_settings(database: Database, patch: SettingsPatchRequest) -> None:
    """Commit one retention or quota settings change the worker will read."""
    async with database.transaction() as session:
        _ = await SettingsService().update(session, patch)


async def age_out_ended(
    database: Database,
    crop_root: Path,
    *,
    count: int,
    deadline_seconds: float,
) -> AgeOutResult:
    """Backdate ended appearances past a one-day retention and wait for the sweep."""

    async def has_ended() -> bool:
        return any(row.ended_at is not None for row in await visible_rows(database))

    # Only ended tracks are aged: live publication may rewrite an active row's timestamps.
    if not await wait_until(has_ended, deadline_seconds):
        raise WorkerStackError(detail="no appearance track ended before the age deadline")
    ended = [row for row in await visible_rows(database) if row.ended_at is not None][:count]
    ids = tuple(row.appearance_id for row in ended)
    keys = tuple(row.crop_object_key for row in ended)
    old = datetime.now(UTC) - timedelta(days=_BACKDATE_DAYS)
    async with database.transaction() as session:
        _ = await session.execute(
            update(Appearance)
            .where(Appearance.id.in_(ids))
            .values(first_seen=old, last_seen=old, ended_at=old)
        )
        _ = await SettingsService().update(session, SettingsPatchRequest(retention_days=1))

    async def aged_out() -> bool:
        gone = not await visible_ids(database, ids)
        return gone and not any((crop_root / key).exists() for key in keys)

    _ = await wait_until(aged_out, deadline_seconds)
    evicted = len(ids) - len(await visible_ids(database, ids))
    crops_removed = not any((crop_root / key).exists() for key in keys)
    return AgeOutResult(appearance_ids=ids, evicted=evicted, crops_removed=crops_removed)


__all__ = [
    "AgeOutResult",
    "VisibleRow",
    "WorkerStackError",
    "WorkerStackSettings",
    "age_out_ended",
    "create_cameras",
    "kill_worker",
    "row_count",
    "start_worker",
    "stop_worker",
    "update_settings",
    "visible_ids",
    "visible_rows",
    "wait_published",
    "wait_until",
]
