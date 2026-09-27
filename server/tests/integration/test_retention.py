from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Event
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import anyio
import pytest
from anyio.to_thread import run_sync
from cryptography.fernet import Fernet
from sqlalchemy import delete, event, select

from gods_watching.appearances import ConservativeWriterBudget
from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.retention import (
    FailurePoint,
    InjectedRetentionFailure,
    RetentionService,
)
from gods_watching.storage import (
    Appearance,
    ApplicationSettings,
    Camera,
    CameraSession,
    CredentialCipher,
    CropGarbage,
    CropObjectStore,
    Database,
    StaleAppearanceVersionError,
    StorageRepository,
)
from gods_watching.storage.physical_usage import managed_crop_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from sqlalchemy.ext.asyncio import AsyncSession


_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class _TestSettings:
    retention_days: int = 7
    quota_bytes: int = 100_000_000_000


class _EmptyRows:
    def tuples(self) -> _EmptyRows:
        return self

    def all(self) -> list[tuple[object, ...]]:
        return []


class _AccountingSession:
    async def scalar(self, statement: object) -> object | None:
        if "application_settings" in str(statement).lower():
            return _TestSettings()
        return None

    async def execute(self, statement: object) -> _EmptyRows:
        del statement
        return _EmptyRows()


class _AccountingTransaction:
    async def __aenter__(self) -> _AccountingSession:
        return _AccountingSession()

    async def __aexit__(self, *_args: object) -> None:
        return None


class _AccountingDatabase:
    def transaction(self) -> _AccountingTransaction:
        return _AccountingTransaction()


class _AccountingStorage:
    async def application_relation_sizes(
        self,
        session: AsyncSession,
    ) -> Mapping[str, int]:
        del session
        return {"appearances": 0}


def _accounting_service(
    tmp_path: Path,
    *,
    writer_budget: ConservativeWriterBudget | None = None,
) -> RetentionService:
    return RetentionService(
        database=cast("Database", cast("object", _AccountingDatabase())),
        storage=cast("StorageRepository", cast("object", _AccountingStorage())),
        crop_store=CropObjectStore(tmp_path / "crops"),
        writer_budget=writer_budget,
    )


@pytest.mark.anyio
async def test_accounting_keeps_async_tasks_responsive_during_slow_scan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    budget = ConservativeWriterBudget(minimum_free_bytes=0)
    service = _accounting_service(tmp_path, writer_budget=budget)
    scan_started = Event()
    release_scan = Event()
    heartbeat_progressed = Event()
    scan_observed_heartbeat = Event()
    competing_writer_acquired = anyio.Event()

    def slow_scan(_root: Path) -> int:
        scan_started.set()
        _ = release_scan.wait(timeout=0.5)
        if heartbeat_progressed.is_set():
            scan_observed_heartbeat.set()
        return 0

    monkeypatch.setattr("gods_watching.retention.service.managed_crop_bytes", slow_scan)

    async def report_accounting() -> None:
        _ = await service.accounting()

    async def heartbeat() -> None:
        with anyio.fail_after(2):
            _ = await run_sync(scan_started.wait)
        heartbeat_progressed.set()
        release_scan.set()

    async def competing_writer() -> None:
        async with budget.writer_gate.hold():
            competing_writer_acquired.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(report_accounting)
        task_group.start_soon(heartbeat)
        task_group.start_soon(competing_writer)
        await anyio.lowlevel.checkpoint()
        assert not competing_writer_acquired.is_set()

    assert scan_observed_heartbeat.is_set()
    assert competing_writer_acquired.is_set()


@pytest.mark.anyio
async def test_no_change_sweep_scans_managed_files_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _accounting_service(tmp_path)
    real_managed_scan = managed_crop_bytes
    scans = 0

    def count_scans(root: Path) -> int:
        nonlocal scans
        scans += 1
        return real_managed_scan(root)

    monkeypatch.setattr("gods_watching.retention.service.managed_crop_bytes", count_scans)

    report = await service.sweep(now=_NOW)

    assert report.age_candidates == 0
    assert report.tombstoned == 0
    assert scans == 1


@pytest.mark.anyio
async def test_age_tombstone_forces_fresh_post_mutation_accounting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service = _accounting_service(tmp_path)
    remaining_file_sizes = [17]
    scans = 0
    candidate = object()

    def scan_current_files(_root: Path) -> int:
        nonlocal scans
        scans += 1
        return sum(remaining_file_sizes)

    async def one_expired_candidate(_now: datetime) -> tuple[None, tuple[object, ...]]:
        return None, (candidate,)

    async def tombstone(_candidate: object) -> bool:
        remaining_file_sizes.clear()
        return True

    async def no_pending_gc(_errors: list[str]) -> tuple[int, int]:
        return 0, 0

    monkeypatch.setattr("gods_watching.retention.service.managed_crop_bytes", scan_current_files)
    monkeypatch.setattr(service, "_age_candidates", one_expired_candidate)
    monkeypatch.setattr(service, "_tombstone", tombstone)
    monkeypatch.setattr(service, "_drain_gc", no_pending_gc)

    report = await service.sweep(now=_NOW)

    assert report.tombstoned == 1
    assert report.managed_bytes_before == 17
    assert report.physical_crop_bytes_after == 0
    assert report.managed_bytes_after == 0
    assert scans == 2


@dataclass
class _FailureInjector:
    point: FailurePoint

    def check(self, point: FailurePoint, object_key: str) -> None:
        if point is self.point:
            raise InjectedRetentionFailure(point, object_key)


@dataclass(frozen=True)
class _StorageAdapter:
    repository: StorageRepository
    relation_bytes: int

    async def application_relation_sizes(self, session: AsyncSession) -> Mapping[str, int]:
        del session
        return {"appearances": self.relation_bytes}

    async def tombstone_appearance(self, session: AsyncSession, appearance_id: UUID) -> None:
        await self.repository.tombstone_appearance(session, appearance_id)

    async def finalize_tombstoned_appearance(
        self,
        session: AsyncSession,
        *,
        appearance_id: UUID,
    ) -> bool:
        return await self.repository.finalize_tombstoned_appearance(
            session,
            appearance_id=appearance_id,
        )


@dataclass
class _Publisher:
    evicted: list[AppearanceId]

    def mark_quota_evicted(self, appearance_id: AppearanceId) -> None:
        self.evicted.append(appearance_id)


def _publication(  # noqa: PLR0913
    *,
    appearance_id: UUID,
    camera_id: UUID,
    session_id: UUID,
    crop_object_key: str,
    first_seen: datetime,
    track_id: int,
    ended_at: datetime | None,
    byte_size: int,
    representative_version: int = 1,
) -> AppearancePublication:
    return AppearancePublication(
        appearance_id=AppearanceId(appearance_id),
        camera_id=CameraId(camera_id),
        session_id=CameraSessionId(session_id),
        track_id=track_id,
        first_seen=first_seen,
        last_seen=first_seen,
        ended_at=ended_at,
        representative_version=representative_version,
        crop_object_key=crop_object_key,
        bounding_box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.9,
        crop_quality=10.0,
        byte_size=byte_size,
        embedded_at=first_seen,
        model_id="fixture-model",
        model_revision="fixture-revision",
        embedding=tuple(1.0 if index == 0 else 0.0 for index in range(512)),
    )


async def _seed(  # noqa: PLR0913
    database: Database,
    store: CropObjectStore,
    repository: StorageRepository,
    *,
    camera_name: str,
    first_seen: datetime,
    track_id: int,
    ended_at: datetime | None = None,
) -> tuple[UUID, UUID, str]:
    async with database.transaction() as session:
        camera = await repository.add_camera(
            session,
            name=camera_name,
            source_url=CameraCreateRequest.model_validate(
                {"name": camera_name, "source_url": "rtsp://fixture:8554/retention"}
            ).source_url,
        )
        camera_session = await repository.start_camera_session(
            session, camera.id, cause="retention_test"
        )
        crop = store.write(f"crop-{track_id}".encode())
        appearance_id = uuid4()
        publication = _publication(
            appearance_id=appearance_id,
            camera_id=camera.id,
            session_id=camera_session.id,
            crop_object_key=crop.object_key,
            first_seen=first_seen,
            track_id=track_id,
            ended_at=ended_at,
            byte_size=crop.byte_size,
        )
        _ = await repository.publish_appearance(session, publication)
    return appearance_id, camera_session.id, crop.object_key


async def _cleanup(
    database: Database,
    camera_session_id: UUID,
    appearance_id: UUID,
    crop_object_key: str,
) -> None:
    async with database.transaction() as session:
        camera_id = await session.scalar(
            select(CameraSession.camera_id).where(CameraSession.id == camera_session_id)
        )
        _ = await session.execute(
            delete(CropGarbage).where(CropGarbage.object_key == crop_object_key)
        )
        _ = await session.execute(delete(Appearance).where(Appearance.id == appearance_id))
        _ = await session.execute(
            delete(CameraSession).where(CameraSession.id == camera_session_id)
        )
        if camera_id is not None:
            _ = await session.execute(delete(Camera).where(Camera.id == camera_id))


@pytest.mark.anyio
async def test_age_sweep_hides_vector_and_reclaims_crop(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    camera_name = f"retention-age-{uuid4().hex}"
    appearance_id, camera_session_id, crop_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=camera_name,
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=1,
    )
    try:
        service = RetentionService(database=database, storage=repository, crop_store=store)
        report = await service.sweep(now=_NOW)
        assert report.age_candidates == 1
        assert report.tombstoned == 1
        assert not (store.root / crop_object_key).exists()
        async with database.transaction() as session:
            appearance = await session.get(Appearance, appearance_id)
            garbage = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == crop_object_key)
            )
        assert appearance is None
        assert garbage is None
    finally:
        await _cleanup(database, camera_session_id, appearance_id, crop_object_key)
        await database.close()


@pytest.mark.anyio
async def test_age_sweep_filters_scalar_candidates_in_sql(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    expired = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-scalar-old-{uuid4().hex}",
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=101,
    )
    recent = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-scalar-recent-{uuid4().hex}",
        first_seen=datetime(2026, 9, 6, tzinfo=UTC),
        track_id=102,
    )
    statements: list[str] = []

    def capture_sql(
        _connection: object,
        _cursor: object,
        statement: str,
        *_args: object,
    ) -> None:
        statements.append(statement.lower())

    try:
        service = RetentionService(database=database, storage=repository, crop_store=store)
        event.listen(database.engine.sync_engine, "before_cursor_execute", capture_sql)
        try:
            report = await service.sweep(now=_NOW)
        finally:
            event.remove(database.engine.sync_engine, "before_cursor_execute", capture_sql)

        candidate_queries = [
            statement
            for statement in statements
            if statement.lstrip().startswith("select")
            and "from appearances" in statement
            and "tombstoned_at is null" in statement
            and "order by appearances.first_seen" in statement
        ]
        assert len(candidate_queries) == 1
        candidate_query = candidate_queries[0]
        assert "first_seen <=" in candidate_query
        assert "embedding" not in candidate_query
        assert "camera_sessions" not in candidate_query
        assert report.age_candidates == 1
        assert report.tombstoned == 1
        async with database.transaction() as session:
            assert await session.get(Appearance, expired[0]) is None
            recent_row = await session.get(Appearance, recent[0])
        assert recent_row is not None
        assert recent_row.tombstoned_at is None
    finally:
        await _cleanup(database, expired[1], expired[0], expired[2])
        await _cleanup(database, recent[1], recent[0], recent[2])
        await database.close()


@pytest.mark.anyio
async def test_age_sweep_finalizes_completed_appearance_metadata_after_gc(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    camera_name = f"retention-finalize-{uuid4().hex}"
    appearance_id, camera_session_id, crop_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=camera_name,
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=10,
        ended_at=datetime(2026, 8, 1, 1, tzinfo=UTC),
    )
    try:
        service = RetentionService(database=database, storage=repository, crop_store=store)
        report = await service.sweep(now=_NOW)

        assert report.age_candidates == 1
        assert report.tombstoned == 1
        assert not (store.root / crop_object_key).exists()
        async with database.transaction() as session:
            assert await session.get(Appearance, appearance_id) is None
            assert (
                await session.scalar(
                    select(CropGarbage).where(CropGarbage.object_key == crop_object_key)
                )
                is None
            )
    finally:
        await _cleanup(database, camera_session_id, appearance_id, crop_object_key)
        await database.close()


@pytest.mark.anyio
async def test_old_representative_gc_keeps_live_current_appearance(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    appearance_id, camera_session_id, old_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-version-{uuid4().hex}",
        first_seen=datetime(2026, 9, 6, tzinfo=UTC),
        track_id=11,
    )
    new_crop = store.write(b"current-representative")
    try:
        async with database.transaction() as session:
            appearance = await session.get(Appearance, appearance_id)
            assert appearance is not None
            _ = await repository.publish_appearance(
                session,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=appearance.camera_id,
                    session_id=appearance.session_id,
                    crop_object_key=new_crop.object_key,
                    first_seen=appearance.first_seen,
                    track_id=appearance.track_id,
                    ended_at=appearance.ended_at,
                    byte_size=new_crop.byte_size,
                    representative_version=2,
                ),
            )

        service = RetentionService(database=database, storage=repository, crop_store=store)
        report = await service.sweep(now=_NOW)

        assert report.age_candidates == 0
        assert report.tombstoned == 0
        assert not (store.root / old_object_key).exists()
        assert (store.root / new_crop.object_key).exists()
        async with database.transaction() as session:
            current = await session.get(Appearance, appearance_id)
        assert current is not None
        assert current.tombstoned_at is None
        assert current.representative_version == 2
        assert current.crop_object_key == new_crop.object_key
    finally:
        await _cleanup(database, camera_session_id, appearance_id, old_object_key)
        async with database.transaction() as session:
            _ = await session.execute(
                delete(CropGarbage).where(CropGarbage.object_key == new_crop.object_key)
            )
        new_crop_path = store.root / new_crop.object_key
        new_crop_path.unlink(missing_ok=True)
        await database.close()


@pytest.mark.anyio
async def test_current_representative_gc_waits_for_old_outbox_before_finalization(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    appearance_id, camera_session_id, old_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-current-first-{uuid4().hex}",
        first_seen=datetime(2026, 9, 6, tzinfo=UTC),
        track_id=13,
    )
    new_crop = store.write(b"current-representative-after-upgrade")
    try:
        async with database.transaction() as session:
            appearance = await session.get(Appearance, appearance_id)
            assert appearance is not None
            _ = await repository.publish_appearance(
                session,
                _publication(
                    appearance_id=appearance_id,
                    camera_id=appearance.camera_id,
                    session_id=appearance.session_id,
                    crop_object_key=new_crop.object_key,
                    first_seen=appearance.first_seen,
                    track_id=appearance.track_id,
                    ended_at=appearance.ended_at,
                    byte_size=new_crop.byte_size,
                    representative_version=2,
                ),
            )
            await repository.tombstone_appearance(session, appearance_id)
        async with database.transaction() as session:
            old_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == old_object_key)
            )
            current_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == new_crop.object_key)
            )
            assert old_gc is not None
            assert current_gc is not None
            current_gc.enqueued_at = datetime(2000, 1, 1, tzinfo=UTC)
            old_gc.enqueued_at = datetime(2000, 1, 2, tzinfo=UTC)

        service = RetentionService(database=database, storage=repository, crop_store=store)
        first = await service.collect_one_gc()
        async with database.transaction() as session:
            after_current_gc = await session.get(Appearance, appearance_id)
            pending_old_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == old_object_key)
            )
        assert first is not None
        assert first.unlinked
        assert after_current_gc is not None
        assert after_current_gc.tombstoned_at is not None
        assert after_current_gc.embedding is None
        assert pending_old_gc is not None
        assert (store.root / old_object_key).exists()

        second = await service.collect_one_gc()
        async with database.transaction() as session:
            assert second is not None
            assert second.unlinked
            assert await session.get(Appearance, appearance_id) is None
            assert (
                await session.scalar(
                    select(CropGarbage).where(
                        CropGarbage.object_key.in_({old_object_key, new_crop.object_key})
                    )
                )
                is None
            )
        assert not (store.root / old_object_key).exists()
        assert not (store.root / new_crop.object_key).exists()
    finally:
        async with database.transaction() as session:
            _ = await session.execute(
                delete(CropGarbage).where(
                    CropGarbage.object_key.in_({old_object_key, new_crop.object_key})
                )
            )
        await _cleanup(database, camera_session_id, appearance_id, old_object_key)
        (store.root / new_crop.object_key).unlink(missing_ok=True)
        await database.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "order_versions",
    [(3, 1, 2), (1, 2, 3), (2, 1, 3)],
    ids=("current-first", "old-first", "middle-first"),
)
async def test_finalization_waits_for_all_owned_outbox_orderings(
    database_url: str,
    tmp_path: Path,
    order_versions: tuple[int, int, int],
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    first = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-order-{uuid4().hex}",
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=23,
    )
    crops = [first[2]]
    try:
        async with database.transaction() as session:
            appearance = await session.get(Appearance, first[0])
            assert appearance is not None
            for version in (2, 3):
                crop = store.write(f"representative-{version}".encode())
                crops.append(crop.object_key)
                _ = await repository.publish_appearance(
                    session,
                    _publication(
                        appearance_id=first[0],
                        camera_id=appearance.camera_id,
                        session_id=appearance.session_id,
                        crop_object_key=crop.object_key,
                        first_seen=appearance.first_seen,
                        track_id=appearance.track_id,
                        ended_at=appearance.ended_at,
                        byte_size=crop.byte_size,
                        representative_version=version,
                    ),
                )
            await repository.tombstone_appearance(session, first[0])
        async with database.transaction() as session:
            garbage = {
                row.object_key: row
                for row in await session.scalars(
                    select(CropGarbage).where(CropGarbage.object_key.in_(crops))
                )
            }
            assert len(garbage) == 3
            for position, version in enumerate(order_versions, start=1):
                garbage[crops[version - 1]].enqueued_at = datetime(2000, 1, position, tzinfo=UTC)

        service = RetentionService(database=database, storage=repository, crop_store=store)
        for index in range(len(order_versions)):
            result = await service.collect_one_gc()
            assert result is not None
            assert result.unlinked
            async with database.transaction() as session:
                persisted = await session.get(Appearance, first[0])
            if index < len(order_versions) - 1:
                assert persisted is not None
                assert persisted.tombstoned_at is not None
            else:
                assert persisted is None
    finally:
        async with database.transaction() as session:
            _ = await session.execute(delete(CropGarbage).where(CropGarbage.object_key.in_(crops)))
        await _cleanup(database, first[1], first[0], first[2])
        for crop_key in crops[1:]:
            (store.root / crop_key).unlink(missing_ok=True)
        await database.close()


@pytest.mark.anyio
async def test_gc_finalizes_one_tombstoned_appearance_while_unrelated_outbox_pending(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    first = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-owner-a-{uuid4().hex}",
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=20,
    )
    second = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-owner-b-{uuid4().hex}",
        first_seen=datetime(2026, 8, 2, tzinfo=UTC),
        track_id=21,
    )
    try:
        async with database.transaction() as session:
            await repository.tombstone_appearance(session, first[0])
            await repository.tombstone_appearance(session, second[0])
        async with database.transaction() as session:
            first_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == first[2])
            )
            second_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == second[2])
            )
            assert first_gc is not None
            assert second_gc is not None
            first_gc.enqueued_at = datetime(2000, 1, 1, tzinfo=UTC)
            second_gc.enqueued_at = datetime(2000, 1, 2, tzinfo=UTC)

        service = RetentionService(database=database, storage=repository, crop_store=store)
        result = await service.collect_one_gc()

        assert result is not None
        assert result.unlinked
        async with database.transaction() as session:
            assert await session.get(Appearance, first[0]) is None
            assert await session.get(Appearance, second[0]) is not None
            assert (
                await session.scalar(select(CropGarbage).where(CropGarbage.object_key == second[2]))
                is not None
            )
    finally:
        await _cleanup(database, first[1], first[0], first[2])
        await _cleanup(database, second[1], second[0], second[2])
        await database.close()


@pytest.mark.anyio
async def test_unowned_legacy_gc_does_not_authorize_metadata_finalization(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    appearance = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-legacy-{uuid4().hex}",
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=22,
    )
    legacy = store.write(b"legacy-unowned-crop")
    try:
        async with database.transaction() as session:
            await repository.tombstone_appearance(session, appearance[0])
            current_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == appearance[2])
            )
            assert current_gc is not None
            current_gc.enqueued_at = datetime(2000, 1, 2, tzinfo=UTC)
            session.add(CropGarbage(object_key=legacy.object_key, byte_size=legacy.byte_size))
            await session.flush()
            legacy_gc = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == legacy.object_key)
            )
            assert legacy_gc is not None
            legacy_gc.enqueued_at = datetime(2000, 1, 1, tzinfo=UTC)

        service = RetentionService(database=database, storage=repository, crop_store=store)
        result = await service.collect_one_gc()

        assert result is not None
        assert result.unlinked
        async with database.transaction() as session:
            row = await session.get(Appearance, appearance[0])
            assert row is not None
            assert row.tombstoned_at is not None
            assert (
                await session.scalar(
                    select(CropGarbage).where(CropGarbage.object_key == legacy.object_key)
                )
                is None
            )
            assert (
                await session.scalar(
                    select(CropGarbage).where(CropGarbage.object_key == appearance[2])
                )
                is not None
            )
    finally:
        await _cleanup(database, appearance[1], appearance[0], appearance[2])
        (store.root / legacy.object_key).unlink(missing_ok=True)
        await database.close()


@pytest.mark.anyio
async def test_finalized_appearance_rejects_stale_upgrade_publication(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    appearance_id, camera_session_id, old_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-stale-{uuid4().hex}",
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=12,
        ended_at=datetime(2026, 8, 1, 1, tzinfo=UTC),
    )
    stale_crop = store.write(b"stale-upgrade")
    try:
        service = RetentionService(database=database, storage=repository, crop_store=store)
        _ = await service.sweep(now=_NOW)
        async with database.transaction() as session:
            assert await session.get(Appearance, appearance_id) is None
            with pytest.raises(StaleAppearanceVersionError):
                _ = await repository.publish_appearance(
                    session,
                    _publication(
                        appearance_id=appearance_id,
                        camera_id=uuid4(),
                        session_id=camera_session_id,
                        crop_object_key=stale_crop.object_key,
                        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
                        track_id=12,
                        ended_at=datetime(2026, 8, 1, 1, tzinfo=UTC),
                        byte_size=stale_crop.byte_size,
                        representative_version=2,
                    ),
                )
    finally:
        await _cleanup(database, camera_session_id, appearance_id, old_object_key)
        stale_crop_path = store.root / stale_crop.object_key
        stale_crop_path.unlink(missing_ok=True)
        await database.close()


@pytest.mark.anyio
async def test_quota_sweep_evicts_oldest_active_track_and_suppresses_it(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    adapter = _StorageAdapter(repository, relation_bytes=100)
    storage = cast("StorageRepository", cast("object", adapter))
    store = CropObjectStore(tmp_path / "crops")
    publisher = _Publisher([])
    oldest = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-quota-old-{uuid4().hex}",
        first_seen=datetime(2026, 9, 5, tzinfo=UTC),
        track_id=3,
    )
    newest = await _seed(
        database,
        store,
        repository,
        camera_name=f"retention-quota-new-{uuid4().hex}",
        first_seen=datetime(2026, 9, 6, tzinfo=UTC),
        track_id=4,
    )
    async with database.transaction() as session:
        settings = await session.scalar(
            select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
        )
        assert settings is not None
        previous_days = settings.retention_days
        previous_quota = settings.quota_bytes
        settings.retention_days = 7
        settings.quota_bytes = 115
    try:
        service = RetentionService(
            database=database,
            storage=storage,
            crop_store=store,
            publisher=publisher,
        )
        report = await service.sweep(now=_NOW)
        assert report.age_candidates == 0
        assert report.quota_candidates == 1
        assert report.tombstoned == 1
        assert report.suppressed_tracks == 1
        assert not report.storage_full
        assert publisher.evicted == [AppearanceId(oldest[0])]
        assert not (store.root / oldest[2]).exists()
        assert (store.root / newest[2]).exists()
        async with database.transaction() as session:
            old_row = await session.get(Appearance, oldest[0])
            new_row = await session.get(Appearance, newest[0])
        assert old_row is None
        assert new_row is not None
        assert new_row.tombstoned_at is None
        assert new_row.embedding is not None
    finally:
        async with database.transaction() as session:
            settings = await session.scalar(
                select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
            )
            assert settings is not None
            settings.retention_days = previous_days
            settings.quota_bytes = previous_quota
        await _cleanup(database, oldest[1], oldest[0], oldest[2])
        await _cleanup(database, newest[1], newest[0], newest[2])
        await database.close()


@pytest.mark.anyio
async def test_gc_replays_after_tombstone_crash(
    database_url: str,
    tmp_path: Path,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    camera_name = f"retention-crash-{uuid4().hex}"
    appearance_id, camera_session_id, crop_object_key = await _seed(
        database,
        store,
        repository,
        camera_name=camera_name,
        first_seen=datetime(2026, 8, 1, tzinfo=UTC),
        track_id=2,
    )
    try:
        crashing = RetentionService(
            database=database,
            storage=repository,
            crop_store=store,
            failure_injector=_FailureInjector(FailurePoint.AFTER_TOMBSTONE),
        )
        with pytest.raises(InjectedRetentionFailure):
            _ = await crashing.sweep(now=_NOW)
        assert (store.root / crop_object_key).exists()
        async with database.transaction() as session:
            appearance = await session.get(Appearance, appearance_id)
            garbage = await session.scalar(
                select(CropGarbage).where(CropGarbage.object_key == crop_object_key)
            )
        assert appearance is not None
        assert appearance.tombstoned_at is not None
        assert appearance.embedding is None
        assert garbage is not None

        restarted = RetentionService(database=database, storage=repository, crop_store=store)
        report = await restarted.sweep(now=_NOW)
        assert report.unlinked == 1
        assert not (store.root / crop_object_key).exists()
        async with database.transaction() as session:
            assert await session.get(Appearance, appearance_id) is None
            assert (
                await session.scalar(
                    select(CropGarbage).where(CropGarbage.object_key == crop_object_key)
                )
                is None
            )
    finally:
        await _cleanup(database, camera_session_id, appearance_id, crop_object_key)
        await database.close()
