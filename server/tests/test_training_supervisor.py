from __future__ import annotations

import asyncio
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from gods_watching.storage import Database
    from gods_watching.training.models import TrainingRequest
    from gods_watching.training.repository import TrainingRepository

import gods_watching.training.supervisor as supervisor_module
from gods_watching.contracts.training import TrainingSupervisorStatus
from gods_watching.training.calibration import CalibrationRefusedError
from gods_watching.training.dataset import DatasetManifest, DatasetSplitCounts
from gods_watching.training.memory import GpuSnapshot
from gods_watching.training.readiness import read_supervisor_status, write_supervisor_status
from gods_watching.training.service import TrainingService, TrainingSupervisorUnavailableError
from gods_watching.training.settings import TrainingSettings
from gods_watching.training.supervisor import (
    ProcessIdentityStatus,
    TrainingSupervisor,
    process_identity_matches,
    process_identity_status,
    process_start_time,
)


class _ValidationDatabase:
    validation_finished: threading.Event
    transactions: int

    def __init__(self, validation_finished: threading.Event) -> None:
        self.validation_finished = validation_finished
        self.transactions = 0

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[object]:
        self.transactions += 1
        yield object()


class _EmptyRequestRepository:
    validation_finished: threading.Event
    claims: int
    source_fingerprint: str

    def __init__(self, validation_finished: threading.Event) -> None:
        self.validation_finished = validation_finished
        self.claims = 0
        self.source_fingerprint = "b" * 64

    async def expire_old_requests(self, session: object, *, now: datetime) -> int:
        del session, now
        return 0

    async def peek_next_request(
        self,
        session: object,
        *,
        now: datetime,
    ) -> TrainingRequest | None:
        del session, now
        return cast(
            "TrainingRequest",
            cast(
                "object",
                SimpleNamespace(
                    request_id=uuid4(),
                    config_snapshot={"micro_batch_size": 2},
                    dataset_fingerprint="a" * 64,
                    kind="preflight",
                    parent_job_id=None,
                ),
            ),
        )

    async def claim_request(
        self,
        session: object,
        request_id: object,
        *,
        owner: str,
        now: datetime,
        lease_until: datetime,
    ) -> TrainingRequest | None:
        del session, request_id, owner, now, lease_until
        assert self.validation_finished.is_set(), "request lease started before dataset warmup"
        self.claims += 1
        return None


class _QueuedRequestRepository:
    validation_finished: threading.Event
    claims: int
    pending: bool
    peek_calls: int
    source_fingerprint: str

    def __init__(self, validation_finished: threading.Event) -> None:
        self.validation_finished = validation_finished
        self.claims = 0
        self.pending = False
        self.peek_calls = 0
        self.source_fingerprint = "b" * 64

    async def expire_old_requests(self, session: object, *, now: datetime) -> int:
        del session, now
        return 0

    async def peek_next_request(
        self,
        session: object,
        *,
        now: datetime,
    ) -> TrainingRequest | None:
        del session, now
        self.peek_calls += 1
        if not self.pending:
            return None
        return cast(
            "TrainingRequest",
            cast(
                "object",
                SimpleNamespace(
                    request_id=uuid4(),
                    config_snapshot={"micro_batch_size": 2},
                    dataset_fingerprint="a" * 64,
                    kind="preflight",
                    parent_job_id=None,
                ),
            ),
        )

    async def claim_request(
        self,
        session: object,
        request_id: object,
        *,
        owner: str,
        now: datetime,
        lease_until: datetime,
    ) -> TrainingRequest | None:
        del session, request_id, owner, now, lease_until
        assert self.validation_finished.is_set(), "request lease started before dataset validation"
        self.claims += 1
        return None


class _NoPendingRequestRepository:
    source_fingerprint: str = "b" * 64
    peek_calls: int

    def __init__(self) -> None:
        self.peek_calls = 0

    async def expire_old_requests(self, session: object, *, now: datetime) -> int:
        del session, now
        return 0

    async def peek_next_request(
        self,
        session: object,
        *,
        now: datetime,
    ) -> TrainingRequest | None:
        del session, now
        self.peek_calls += 1
        return None


def _gpu_snapshot() -> GpuSnapshot:
    return GpuSnapshot(
        uuid="GPU-17913b0a-8144-5f39-7062-15265e5dca33",
        total_bytes=16 * 1024**3,
        free_bytes=8 * 1024**3,
        observed_at=datetime.now(UTC),
    )


def _manifest(root: Path) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="cuhk-pedes",
        fingerprint="a" * 64,
        protocol="cuhk-pedes-original-splits-v1",
        root=root,
        samples=(),
        split_counts={
            "train": DatasetSplitCounts(images=2, captions=2, identities=2),
            "val": DatasetSplitCounts(images=1, captions=1, identities=1),
            "test": DatasetSplitCounts(images=1, captions=1, identities=1),
        },
    )


def test_live_orphan_identity_is_detected_and_pid_reuse_is_rejected() -> None:
    pid = os.getpid()
    start_time = process_start_time(pid)

    assert process_identity_matches(pid, start_time)
    assert not process_identity_matches(pid, start_time + 1)


def test_missing_child_identity_fails_closed() -> None:
    assert not process_identity_matches(2_147_483_647, 1)
    assert process_identity_status(2_147_483_647, 1) == ProcessIdentityStatus.GONE


def test_startup_replaces_a_fresh_ready_heartbeat_before_recovery_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        validation_finished = threading.Event()
        recovery_started = threading.Event()
        recovery_release = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _NoPendingRequestRepository()
        training_root = tmp_path / "runs"
        write_supervisor_status(
            training_root,
            TrainingSupervisorStatus(
                state="ready",
                observed_at=datetime.now(UTC),
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint="a" * 64,
            ),
        )
        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=training_root),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=_gpu_snapshot,
            dataset_validator=lambda root: _manifest(root),
        )

        async def slow_recovery() -> None:
            recovery_started.set()
            assert await asyncio.to_thread(recovery_release.wait, 3)

        async def no_recovery() -> None:
            return None

        monkeypatch.setattr(supervisor, "_reconcile_children", slow_recovery)
        monkeypatch.setattr(supervisor, "_recover_untracked_slot", no_recovery)
        stop_event = asyncio.Event()
        supervisor_run = asyncio.create_task(supervisor.run(stop_event))
        try:
            assert await asyncio.to_thread(recovery_started.wait, 1)
            status = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=None,
            )
            assert status.state == "validating"
        finally:
            recovery_release.set()
            stop_event.set()
            await supervisor_run

    asyncio.run(scenario())


def test_process_identity_read_permission_error_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(_pid: int, *, proc_root: Path | None = None) -> int:
        del proc_root
        raise PermissionError

    monkeypatch.setattr(supervisor_module, "process_start_time", denied)

    assert process_identity_status(123, 987654) == ProcessIdentityStatus.UNVERIFIABLE


def test_proc_stat_reader_handles_spaces_and_parentheses_in_comm(tmp_path: Path) -> None:
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    # /proc/<pid>/stat field 22 is index 19 after the closing parenthesis.
    fields = ["S", *["0"] * 18, "987654", "0"]
    _ = (process_dir / "stat").write_text(
        f"123 (name with ) chars) {' '.join(fields)}", encoding="utf-8"
    )

    assert process_start_time(123, proc_root=tmp_path) == 987654


def test_slow_dataset_validator_finishes_before_supervisor_takes_request_lease(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        validation_started = threading.Event()
        validation_release = threading.Event()
        validation_finished = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _EmptyRequestRepository(validation_finished)

        def slow_validator(_root: Path) -> DatasetManifest:
            validation_started.set()
            assert validation_release.wait(timeout=3)
            validation_finished.set()
            return DatasetManifest(
                dataset_id="cuhk-pedes",
                fingerprint="a" * 64,
                protocol="cuhk-pedes-original-splits-v1",
                root=tmp_path,
                samples=(),
                split_counts={
                    "train": DatasetSplitCounts(images=2, captions=2, identities=2),
                    "val": DatasetSplitCounts(images=1, captions=1, identities=1),
                    "test": DatasetSplitCounts(images=1, captions=1, identities=1),
                },
            )

        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=tmp_path / "runs"),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=_gpu_snapshot,
            dataset_validator=slow_validator,
        )

        process = asyncio.create_task(supervisor.process_one())
        did_start = await asyncio.to_thread(validation_started.wait, 1)
        if not did_start:
            await process
        assert did_start, "supervisor tried to lease work before warming the dataset"
        assert repository.claims == 0

        validation_release.set()
        assert not await process
        assert repository.claims == 1
        assert database.transactions >= 1

    asyncio.run(scenario())


def test_unchanged_request_revalidation_keeps_ready_gate_open_without_early_claim(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        validation_started = threading.Event()
        validation_release = threading.Event()
        validation_finished = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _QueuedRequestRepository(validation_finished)
        training_root = tmp_path / "runs"
        manifest = _manifest(tmp_path)
        signature_probe_calls = 0
        full_validation_calls = 0

        def block_unchanged_signature_probe(_root: Path) -> DatasetManifest:
            nonlocal signature_probe_calls
            signature_probe_calls += 1
            validation_started.set()
            assert validation_release.wait(timeout=3)
            validation_finished.set()
            return manifest

        def full_validator(_root: Path) -> DatasetManifest:
            nonlocal full_validation_calls
            full_validation_calls += 1
            return manifest

        settings = TrainingSettings(dataset_root=tmp_path, training_root=training_root)
        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            settings,
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=_gpu_snapshot,
            dataset_validator=full_validator,
            dataset_signature_probe=block_unchanged_signature_probe,
        )
        assert not await supervisor.process_one()
        repository.pending = True
        api_service = TrainingService(
            cast("Database", cast("object", database)),
            settings,
            repository=cast("TrainingRepository", cast("object", repository)),
        )

        process = asyncio.create_task(supervisor.process_one())
        try:
            did_start = await asyncio.to_thread(validation_started.wait, 1)
            if not did_start:
                await process
            assert did_start, "request A did not recheck its cached dataset signatures"

            during = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=manifest.fingerprint,
            )
            assert during.state == "ready"
            api_service._require_supervisor_ready(manifest)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            assert repository.claims == 0

            validation_release.set()
            assert not await process
            assert signature_probe_calls == 1
            assert full_validation_calls == 1
            assert repository.claims == 1
            assert validation_finished.is_set()
        finally:
            validation_release.set()
            if not process.done():
                await process

    asyncio.run(scenario())


def test_cache_miss_revokes_ready_gate_until_full_dataset_validation_finishes(  # noqa: PLR0915
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        validation_started = threading.Event()
        validation_release = threading.Event()
        validation_finished = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _QueuedRequestRepository(validation_finished)
        training_root = tmp_path / "runs"
        cached_manifest = _manifest(tmp_path)
        changed_manifest = DatasetManifest(
            dataset_id=cached_manifest.dataset_id,
            fingerprint="c" * 64,
            protocol=cached_manifest.protocol,
            root=cached_manifest.root,
            samples=cached_manifest.samples,
            split_counts=cached_manifest.split_counts,
        )
        signature_probe_calls = 0
        full_validation_calls = 0

        def cache_miss(_root: Path) -> None:
            nonlocal signature_probe_calls
            signature_probe_calls += 1

        def full_validator(_root: Path) -> DatasetManifest:
            nonlocal full_validation_calls
            full_validation_calls += 1
            if full_validation_calls == 1:
                return cached_manifest
            validation_started.set()
            assert validation_release.wait(timeout=3)
            validation_finished.set()
            return changed_manifest

        settings = TrainingSettings(dataset_root=tmp_path, training_root=training_root)
        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            settings,
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=_gpu_snapshot,
            dataset_validator=full_validator,
            dataset_signature_probe=cache_miss,
        )
        assert not await supervisor.process_one()
        repository.pending = True
        api_service = TrainingService(
            cast("Database", cast("object", database)),
            settings,
            repository=cast("TrainingRepository", cast("object", repository)),
        )

        process = asyncio.create_task(supervisor.process_one())
        try:
            did_start = await asyncio.to_thread(validation_started.wait, 1)
            if not did_start:
                await process
            assert did_start, "cache miss did not start full dataset validation"
            assert signature_probe_calls == 1
            assert full_validation_calls == 2

            during = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=None,
            )
            assert during.state == "validating"
            assert during.dataset_fingerprint == cached_manifest.fingerprint
            with pytest.raises(TrainingSupervisorUnavailableError):
                api_service._require_supervisor_ready(cached_manifest)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            assert repository.claims == 0

            validation_release.set()
            assert not await process
            assert repository.claims == 1
            ready = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=changed_manifest.fingerprint,
            )
            assert ready.state == "ready"
            assert ready.dataset_fingerprint == changed_manifest.fingerprint
        finally:
            validation_release.set()
            if not process.done():
                await process

    asyncio.run(scenario())


def test_startup_child_and_orphan_recovery_precede_slow_dataset_warmup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        validation_started = threading.Event()
        validation_release = threading.Event()
        validation_finished = threading.Event()
        database = _ValidationDatabase(validation_finished)
        events: list[str] = []

        def slow_validator(_root: Path) -> DatasetManifest:
            validation_started.set()
            assert validation_release.wait(timeout=3)
            validation_finished.set()
            return DatasetManifest(
                dataset_id="cuhk-pedes",
                fingerprint="a" * 64,
                protocol="cuhk-pedes-original-splits-v1",
                root=tmp_path,
                samples=(),
                split_counts={
                    "train": DatasetSplitCounts(images=2, captions=2, identities=2),
                    "val": DatasetSplitCounts(images=1, captions=1, identities=1),
                    "test": DatasetSplitCounts(images=1, captions=1, identities=1),
                },
            )

        repository = _NoPendingRequestRepository()

        def warm_gpu() -> GpuSnapshot:
            events.append("gpu")
            return _gpu_snapshot()

        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=tmp_path / "runs"),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=warm_gpu,
            dataset_validator=slow_validator,
        )

        async def reconcile_children() -> None:
            events.append("children")

        async def recover_orphan() -> None:
            events.append("orphan")

        monkeypatch.setattr(supervisor, "_reconcile_children", reconcile_children)
        monkeypatch.setattr(supervisor, "_recover_untracked_slot", recover_orphan)
        stop_event = asyncio.Event()
        supervisor_run = asyncio.create_task(supervisor.run(stop_event))
        assert await asyncio.to_thread(validation_started.wait, 1)
        assert events == ["children", "orphan"]

        validation_release.set()
        for _attempt in range(100):
            if "gpu" in events:
                break
            await asyncio.sleep(0.01)
        assert events == ["children", "orphan", "gpu"]
        stop_event.set()
        await supervisor_run

    asyncio.run(scenario())


def test_gpu_telemetry_warms_before_ready_and_request_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        validation_finished = threading.Event()
        gpu_started = threading.Event()
        gpu_release = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _NoPendingRequestRepository()
        training_root = tmp_path / "runs"

        def slow_gpu_snapshot() -> GpuSnapshot:
            gpu_started.set()
            assert gpu_release.wait(timeout=8)
            return _gpu_snapshot()

        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=training_root),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=slow_gpu_snapshot,
            dataset_validator=lambda root: _manifest(root),
        )

        async def no_reconcile() -> None:
            return None

        monkeypatch.setattr(supervisor, "_reconcile_children", no_reconcile)
        monkeypatch.setattr(supervisor, "_recover_untracked_slot", no_reconcile)
        stop_event = asyncio.Event()
        supervisor_run = asyncio.create_task(supervisor.run(stop_event))
        try:
            assert await asyncio.to_thread(gpu_started.wait, 1)
            before = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=None,
            )
            assert before.state == "validating"
            assert repository.peek_calls == 0

            await asyncio.sleep(5.1)
            during = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=None,
            )
            assert during.state == "validating"
            assert before.observed_at is not None
            assert during.observed_at is not None
            assert during.observed_at > before.observed_at
            assert repository.peek_calls == 0

            gpu_release.set()
            ready = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint="a" * 64,
            )
            for _attempt in range(200):
                ready = read_supervisor_status(
                    training_root,
                    source_fingerprint=repository.source_fingerprint,
                    dataset_fingerprint="a" * 64,
                )
                if ready.state == "ready" and repository.peek_calls > 0:
                    break
                await asyncio.sleep(0.01)
            assert ready.state == "ready"
            assert repository.peek_calls > 0
        finally:
            gpu_release.set()
            stop_event.set()
            await supervisor_run

    asyncio.run(scenario())


def test_gpu_telemetry_failure_stays_unavailable_without_request_leasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        validation_finished = threading.Event()
        gpu_started = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _NoPendingRequestRepository()
        training_root = tmp_path / "runs"

        def failed_gpu_snapshot() -> GpuSnapshot:
            gpu_started.set()
            message = "sensor unavailable"
            raise CalibrationRefusedError(message)

        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=training_root),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=failed_gpu_snapshot,
            dataset_validator=lambda root: _manifest(root),
        )

        async def no_reconcile() -> None:
            return None

        monkeypatch.setattr(supervisor, "_reconcile_children", no_reconcile)
        monkeypatch.setattr(supervisor, "_recover_untracked_slot", no_reconcile)
        stop_event = asyncio.Event()
        supervisor_run = asyncio.create_task(supervisor.run(stop_event))
        try:
            assert await asyncio.to_thread(gpu_started.wait, 1)
            unavailable = read_supervisor_status(
                training_root,
                source_fingerprint=repository.source_fingerprint,
                dataset_fingerprint=None,
            )
            for _attempt in range(100):
                unavailable = read_supervisor_status(
                    training_root,
                    source_fingerprint=repository.source_fingerprint,
                    dataset_fingerprint=None,
                )
                if unavailable.reason == "training_gpu_unavailable":
                    break
                await asyncio.sleep(0.01)
            assert unavailable.state == "unavailable"
            assert unavailable.reason == "training_gpu_unavailable"
            assert repository.peek_calls == 0
        finally:
            stop_event.set()
            await supervisor_run

    asyncio.run(scenario())


def test_gpu_telemetry_process_failure_retries_before_request_leasing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        validation_finished = threading.Event()
        database = _ValidationDatabase(validation_finished)
        repository = _NoPendingRequestRepository()
        training_root = tmp_path / "runs"
        telemetry_calls = 0

        def intermittent_gpu_snapshot() -> GpuSnapshot:
            nonlocal telemetry_calls
            telemetry_calls += 1
            if telemetry_calls == 1:
                raise subprocess.CalledProcessError(1, ("nvidia-smi", "--query-gpu"))
            return _gpu_snapshot()

        supervisor = TrainingSupervisor(
            cast("Database", cast("object", database)),
            TrainingSettings(dataset_root=tmp_path, training_root=training_root),
            child_launcher=None,
            repository=cast("TrainingRepository", cast("object", repository)),
            gpu_snapshot_provider=intermittent_gpu_snapshot,
            dataset_validator=lambda root: _manifest(root),
        )

        assert not await supervisor.process_one()
        assert telemetry_calls == 1
        assert repository.peek_calls == 0
        unavailable = read_supervisor_status(
            training_root,
            source_fingerprint=repository.source_fingerprint,
            dataset_fingerprint="a" * 64,
        )
        assert unavailable.state == "unavailable"
        assert unavailable.reason == "training_gpu_unavailable"

        assert not await supervisor.process_one()
        assert telemetry_calls == 1
        assert repository.peek_calls == 0

        monkeypatch.setattr(supervisor, "_gpu_retry_after", 0.0)
        assert not await supervisor.process_one()
        assert telemetry_calls == 2
        assert repository.peek_calls == 1
        ready = read_supervisor_status(
            training_root,
            source_fingerprint=repository.source_fingerprint,
            dataset_fingerprint="a" * 64,
        )
        assert ready.state == "ready"
        assert ready.reason is None

    asyncio.run(scenario())
