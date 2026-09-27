from __future__ import annotations

from typing import TYPE_CHECKING, cast, final

import anyio
import pytest

from gods_watching.appearances.pipeline import PublicationStatsSnapshot
from gods_watching.pipeline_worker import app as worker_app
from gods_watching.retention import StorageAccounting

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from gods_watching.appearances import AppearanceHandoffConsumer
    from gods_watching.inference.detector import TritonGrpcDetectorTransport
    from gods_watching.retention import RetentionService
    from gods_watching.status import StatusReporter, WorkerStatusSnapshot
    from gods_watching.storage import Database


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@final
class _Transaction:
    async def __aenter__(self) -> object:
        return object()

    async def __aexit__(self, *_args: object) -> None:
        return None


@final
class _Database:
    def transaction(self) -> _Transaction:
        return _Transaction()


@final
class _RecordingReporter:
    def __init__(self) -> None:
        self.snapshots: list[WorkerStatusSnapshot] = []
        self._condition = anyio.Condition()

    async def report(
        self,
        _session: object,
        *,
        snapshot: WorkerStatusSnapshot,
    ) -> None:
        async with self._condition:
            self.snapshots.append(snapshot)
            self._condition.notify_all()

    async def wait_for_count(self, count: int) -> None:
        with anyio.fail_after(2):
            async with self._condition:
                while len(self.snapshots) < count:
                    await self._condition.wait()

    async def wait_for_managed_bytes(self, byte_count: int) -> None:
        with anyio.fail_after(2):
            async with self._condition:
                while not any(
                    snapshot.storage_managed_bytes == byte_count for snapshot in self.snapshots
                ):
                    await self._condition.wait()

    async def wait_for_paused_after(self, index: int) -> None:
        with anyio.fail_after(2):
            async with self._condition:
                while not any(snapshot.persistence_paused for snapshot in self.snapshots[index:]):
                    await self._condition.wait()


@final
class _GatedRetention:
    def __init__(self) -> None:
        self.calls = 0
        self.first_started = anyio.Event()
        self.release_first = anyio.Event()
        self.second_started = anyio.Event()
        self.release_second = anyio.Event()

    async def accounting(self) -> StorageAccounting:
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await self.release_first.wait()
            return _accounting(physical_crop_bytes=42)
        if self.calls == 2:
            self.second_started.set()
            await self.release_second.wait()
            raise OSError
        return _accounting(physical_crop_bytes=80)


@final
class _StaleRetention:
    def __init__(self) -> None:
        self.calls = 0
        self.second_started = anyio.Event()
        self.release_second = anyio.Event()

    async def accounting(self) -> StorageAccounting:
        self.calls += 1
        if self.calls == 1:
            return _accounting(physical_crop_bytes=42)
        self.second_started.set()
        await self.release_second.wait()
        return _accounting(physical_crop_bytes=80)


@final
class _Detector:
    async def ready(self) -> bool:
        return True


@final
class _Consumer:
    @property
    def stats(self) -> PublicationStatsSnapshot:
        return PublicationStatsSnapshot(
            pending_embeddings=3,
            last_searchable_latency_seconds=1.25,
        )


def _accounting(*, physical_crop_bytes: int) -> StorageAccounting:
    return StorageAccounting(
        physical_crop_bytes=physical_crop_bytes,
        pending_gc_bytes=0,
        relation_bytes=8,
        filesystem_free_bytes=20 * 1024**3,
        quota_bytes=1000,
        minimum_free_bytes=5 * 1024**3,
    )


@pytest.mark.anyio
async def test_status_heartbeat_continues_while_accounting_refresh_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_app,
        "_ACCOUNTING_REFRESH_INTERVAL_SECONDS",
        0.01,
        raising=False,
    )
    retention = _GatedRetention()
    reporter = _RecordingReporter()
    status_loop_factory = cast("Callable[..., object]", vars(worker_app)["_StatusLoop"])
    report_status = cast(
        "Callable[..., Awaitable[None]]",
        vars(worker_app)["_report_status"],
    )
    loop = status_loop_factory(
        database=cast("Database", cast("object", _Database())),
        reporter=cast("StatusReporter", cast("object", reporter)),
        retention=cast("RetentionService", cast("object", retention)),
        detector=cast("TritonGrpcDetectorTransport", cast("object", _Detector())),
        consumer=cast("AppearanceHandoffConsumer", cast("object", _Consumer())),
    )
    stop_event = anyio.Event()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(report_status, loop, stop_event, 0.005)
        try:
            await retention.first_started.wait()
            await reporter.wait_for_count(1)
            initial = reporter.snapshots[0]
            assert initial.persistence_paused
            assert initial.storage_managed_bytes == 0
            assert initial.storage_quota_bytes == 0

            retention.release_first.set()
            await reporter.wait_for_managed_bytes(50)
            await retention.second_started.wait()

            count_before_block = len(reporter.snapshots)
            await reporter.wait_for_count(count_before_block + 3)
            during_refresh = reporter.snapshots[count_before_block:]
            assert during_refresh
            assert all(snapshot.storage_managed_bytes == 50 for snapshot in during_refresh)
            assert all(snapshot.storage_quota_bytes == 1000 for snapshot in during_refresh)
            assert all(not snapshot.persistence_paused for snapshot in during_refresh)

            failure_index = len(reporter.snapshots)
            retention.release_second.set()
            await reporter.wait_for_paused_after(failure_index)
            failed_refresh = next(
                snapshot
                for snapshot in reporter.snapshots[failure_index:]
                if snapshot.persistence_paused
            )
            assert failed_refresh.storage_managed_bytes == 50
        finally:
            stop_event.set()
            retention.release_first.set()
            retention.release_second.set()


@pytest.mark.anyio
async def test_status_marks_old_accounting_unknown_while_refresh_is_stalled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_app,
        "_ACCOUNTING_REFRESH_INTERVAL_SECONDS",
        0.01,
        raising=False,
    )
    monkeypatch.setattr(worker_app, "_ACCOUNTING_STALE_AFTER_SECONDS", 0.03, raising=False)
    retention = _StaleRetention()
    reporter = _RecordingReporter()
    status_loop_factory = cast("Callable[..., object]", vars(worker_app)["_StatusLoop"])
    report_status = cast(
        "Callable[..., Awaitable[None]]",
        vars(worker_app)["_report_status"],
    )
    loop = status_loop_factory(
        database=cast("Database", cast("object", _Database())),
        reporter=cast("StatusReporter", cast("object", reporter)),
        retention=cast("RetentionService", cast("object", retention)),
        detector=cast("TritonGrpcDetectorTransport", cast("object", _Detector())),
        consumer=cast("AppearanceHandoffConsumer", cast("object", _Consumer())),
    )
    stop_event = anyio.Event()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(report_status, loop, stop_event, 0.005)
        try:
            await retention.second_started.wait()
            await reporter.wait_for_managed_bytes(50)
            assert not reporter.snapshots[-1].persistence_paused

            stale_index = len(reporter.snapshots)
            await reporter.wait_for_paused_after(stale_index)
            stale_snapshot = next(
                snapshot
                for snapshot in reporter.snapshots[stale_index:]
                if snapshot.persistence_paused
            )
            assert stale_snapshot.storage_managed_bytes == 0
            assert stale_snapshot.storage_quota_bytes == 0

            report_index = len(reporter.snapshots)
            await reporter.wait_for_count(report_index + 2)
            while_stalled = reporter.snapshots[report_index:]
            assert all(snapshot.persistence_paused for snapshot in while_stalled)
            assert all(snapshot.storage_managed_bytes == 0 for snapshot in while_stalled)
        finally:
            stop_event.set()
            retention.release_second.set()
