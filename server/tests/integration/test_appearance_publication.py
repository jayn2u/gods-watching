from __future__ import annotations

import asyncio
import threading
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast, final
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import anyio
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete, select

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    AppearancePublisher,
    BudgetLease,
    BudgetSnapshot,
    PublicationAck,
)
from gods_watching.appearances.queue import PendingEmbeddingQueue
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import (
    CropCandidate,
    GenerationBinding,
    PipelineHandoff,
    RgbCrop,
)
from gods_watching.inference.clip import ClipAdapter
from gods_watching.media.models import SourceGenerationId
from gods_watching.retention import RetentionService
from gods_watching.storage import (
    Appearance,
    ApplicationSettings,
    Camera,
    CameraSession,
    CredentialCipher,
    CropGarbage,
    CropObjectStore,
    Database,
    StorageRepository,
)
from gods_watching.storage.physical_usage import PhysicalUsageError, physical_crop_bytes
from gods_watching.tracking.models import (
    DetectorInputReference,
    LifecycleKind,
    LocalTrackId,
    ResetReason,
    TrackKey,
    TrackLifecycle,
    TrackObservation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pydantic import AnyUrl
    from sqlalchemy.ext.asyncio import AsyncSession


class SyntheticEmbeddingError(RuntimeError):
    pass


@final
class _ClipTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.image_calls = 0

    async def embed_image(self, image: bytes) -> tuple[float, ...]:
        self.image_calls += 1
        if self.fail:
            raise SyntheticEmbeddingError
        assert image
        return (1.0,) + (0.0,) * 511

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0,) + (0.0,) * 511


@final
class _DelayedClipTransport:
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def embed_image(self, image: bytes) -> tuple[float, ...]:
        if not image:
            raise SyntheticEmbeddingError
        self.started.set()
        await self.release.wait()
        return (1.0,) + (0.0,) * 511

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0,) + (0.0,) * 511


@final
class _Budget:
    def __init__(self) -> None:
        self.snapshots: list[BudgetSnapshot] = []

    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease:
        self.snapshots.append(snapshot)
        return BudgetLease(reserved_bytes=1)


@final
class _PauseOnceBudget:
    def __init__(self) -> None:
        self.started = anyio.Event()
        self.release = anyio.Event()
        self.calls = 0

    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease | None:
        del snapshot
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
            return None
        return BudgetLease(reserved_bytes=1)


@final
class _ReservationSession:
    def __init__(self) -> None:
        self._scalar_calls = 0

    async def scalar(self, statement: object) -> object:
        del statement
        self._scalar_calls += 1
        return 0 if self._scalar_calls == 1 else None


@final
class _ReservationTransaction:
    def __init__(self) -> None:
        self._session = _ReservationSession()

    async def __aenter__(self) -> _ReservationSession:
        return self._session

    async def __aexit__(self, *_: object) -> bool:
        return False


@final
class _ReservationDatabase:
    def transaction(self) -> _ReservationTransaction:
        return _ReservationTransaction()


@final
class _ReservationStorage:
    async def application_relation_sizes(self, session: AsyncSession) -> Mapping[str, int]:
        del session
        return {}


@final
class _RetentionStorageAdapter:
    def __init__(self, repository: StorageRepository, relation_bytes: int) -> None:
        self.repository = repository
        self.relation_bytes = relation_bytes

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


def _source(value: str) -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": value, "source_url": f"rtsp://fixture:8554/{value.lower()}"}
    ).source_url


def _handoff(  # noqa: PLR0913
    *,
    camera_id: CameraId,
    session_id: CameraSessionId,
    db_generation: UUID,
    source_generation: SourceGenerationId | None = None,
    track_id: int = 3,
    kind: LifecycleKind = LifecycleKind.START,
    camera_version: int = 1,
    observed_at: datetime | None = None,
    bounding_box: tuple[float, float, float, float] = (10.0, 20.0, 80.0, 180.0),
    include_candidate: bool = True,
) -> PipelineHandoff:
    source_generation = source_generation or SourceGenerationId(uuid4())
    key = TrackKey(camera_id, session_id, source_generation, LocalTrackId(track_id))
    observed_at = observed_at or datetime(2026, 9, 7, 2, 0, tzinfo=UTC)
    observation = TrackObservation(
        bounding_box=bounding_box,
        confidence=0.9,
        detector_input_reference=DetectorInputReference("frame-1"),
        ingress_utc=observed_at,
        ingress_monotonic=1.0,
        detector_result_monotonic=1.2,
    )
    candidate = CropCandidate(
        track_key=key,
        observation=observation,
        crop=RgbCrop(data=bytes(64 * 128 * 3), width=64, height=128),
        source_width=100,
        source_height=200,
    )
    return PipelineHandoff(
        generation=GenerationBinding(
            camera_id=camera_id,
            camera_session_id=session_id,
            db_generation_id=CameraGenerationId(db_generation),
            source_generation_id=source_generation,
            camera_version=camera_version,
        ),
        lifecycle=TrackLifecycle(
            kind=kind,
            track_key=key,
            source_generation_id=source_generation,
            scope_epoch=0,
            sequence=2 if kind is LifecycleKind.END else 1,
            observation=observation,
            first_candidate=observation,
            first_seen=observed_at,
            last_seen=observed_at,
            t_detect_monotonic=1.2,
            ended_at=observed_at if kind is LifecycleKind.END else None,
            ended_monotonic=3.0 if kind is LifecycleKind.END else None,
            end_reason=ResetReason.EXPLICIT if kind is LifecycleKind.END else None,
        ),
        candidate=None if kind is LifecycleKind.END or not include_candidate else candidate,
    )


@pytest.mark.anyio
async def test_slow_physical_crop_scan_does_not_block_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CropObjectStore(tmp_path / "crops")
    existing_crop = store.root / "existing.jpg"
    _ = existing_crop.write_bytes(b"saved crop")
    temporary_crop = store.root / "nested" / ".partial.tmp"
    _ = temporary_crop.parent.mkdir(parents=True, exist_ok=True)
    _ = temporary_crop.write_bytes(b"partial")
    _ = (store.root / "notes.txt").write_bytes(b"not counted")
    outside_crop = tmp_path / "outside.jpg"
    _ = outside_crop.write_bytes(b"outside")
    (store.root / "linked.jpg").symlink_to(outside_crop)
    outside_directory = tmp_path / "outside-directory"
    _ = outside_directory.mkdir()
    _ = (outside_directory / "escaped.jpg").write_bytes(b"outside directory")
    (store.root / "linked-directory").symlink_to(outside_directory, target_is_directory=True)
    budget = _Budget()
    publisher = AppearancePublisher(
        database=cast("Database", _ReservationDatabase()),
        storage=cast("StorageRepository", _ReservationStorage()),
        crop_store=store,
        clip=ClipAdapter(_ClipTransport()),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=budget,
    )
    physical_crop_bytes = publisher._physical_crop_bytes  # noqa: SLF001
    scan_started = threading.Event()
    scan_finished = threading.Event()
    heartbeat_during_scan = anyio.Event()
    event_loop = asyncio.get_running_loop()

    def record_heartbeat() -> None:
        if not scan_finished.is_set():
            heartbeat_during_scan.set()

    def slow_physical_crop_scan() -> int:
        scan_started.set()
        event_loop.call_soon_threadsafe(record_heartbeat)
        time.sleep(0.15)
        try:
            return physical_crop_bytes()
        finally:
            scan_finished.set()

    monkeypatch.setattr(publisher, "_physical_crop_bytes", slow_physical_crop_scan)
    lease = await publisher._reserve(new_crop_bytes=1)  # noqa: SLF001

    assert lease is not None
    try:
        assert heartbeat_during_scan.is_set()
        assert budget.snapshots[0].physical_crop_bytes == len(b"saved crop") + len(b"partial")
    finally:
        await lease.release()


@pytest.mark.anyio
async def test_physical_size_scan_failure_pauses_and_keeps_work_retryable(
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    transport = _ClipTransport()
    publisher = AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(transport),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=_Budget(),
        defer_publication=True,
    )
    consumer = AppearanceHandoffConsumer(publisher)
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication failed accounting {uuid4().hex}",
                source_url=_source(f"failed-accounting-{uuid4().hex}"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = camera.id
            session_id = camera_session.id

        initial = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
            )
        )
        assert initial.outcome.value == "queued"

        def failed_scan(_root: Path) -> int:
            raise PhysicalUsageError

        monkeypatch.setattr(
            "gods_watching.appearances.publication.physical_crop_bytes", failed_scan
        )
        paused = await consumer.drain_one()
        assert paused is not None
        assert paused.outcome.value == "paused"
        assert publisher.pending_embeddings == 1
        assert transport.image_calls == 0

        monkeypatch.setattr(
            "gods_watching.appearances.publication.physical_crop_bytes", physical_crop_bytes
        )
        retried = await consumer.drain_one()
        assert retried is not None
        assert retried.outcome.value == "published"
        assert retried.appearance_id == initial.appearance_id
        assert transport.image_calls == 1
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_deferred_initial_publication_persists_end_timestamp(
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    transport = _ClipTransport()
    publisher = AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(transport),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=_Budget(),
        defer_publication=True,
    )
    consumer = AppearanceHandoffConsumer(publisher)
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication deferred end {uuid4().hex}",
                source_url=_source(f"deferred-{uuid4().hex}"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = camera.id
            session_id = camera_session.id

        source_generation = SourceGenerationId(uuid4())
        start_time = datetime(2026, 9, 7, 2, 0, tzinfo=UTC)
        end_time = datetime(2026, 9, 7, 2, 5, tzinfo=UTC)
        initial = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                observed_at=start_time,
            )
        )
        ended = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                kind=LifecycleKind.END,
                observed_at=end_time,
            )
        )

        assert initial.outcome.value == "queued"
        assert ended.outcome.value == "queued"
        assert transport.image_calls == 0
        published = await consumer.drain_one()
        assert published is not None
        assert published.outcome.value == "published"
        async with database.transaction() as session:
            appearance = await session.scalar(
                select(Appearance).where(Appearance.camera_id == camera_id)
            )
        assert appearance is not None
        assert appearance.last_seen == end_time
        assert appearance.ended_at == end_time
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_slow_draining_does_not_block_other_handoffs_and_orders_end(  # noqa: PLR0915
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    transport = _DelayedClipTransport()
    publisher = AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(transport),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=_Budget(),
        defer_publication=True,
    )
    consumer = AppearanceHandoffConsumer(publisher)
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication deferred drain {uuid4().hex}",
                source_url=_source(f"deferred-drain-{uuid4().hex}"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = camera.id
            session_id = camera_session.id

        source_generation = SourceGenerationId(uuid4())
        first = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                track_id=31,
            )
        )
        assert first.outcome.value == "queued"
        drained: list[PublicationAck] = []
        ended: list[PublicationAck] = []

        async def drain_first() -> None:
            acknowledgement = await consumer.drain_one()
            assert acknowledgement is not None
            drained.append(acknowledgement)

        async def end_first() -> None:
            ended.append(
                await consumer.consume(
                    _handoff(
                        camera_id=CameraId(camera_id),
                        session_id=CameraSessionId(session_id),
                        db_generation=camera_session.generation_id,
                        source_generation=source_generation,
                        track_id=31,
                        kind=LifecycleKind.END,
                        observed_at=datetime(2026, 9, 7, 2, 5, tzinfo=UTC),
                    )
                )
            )

        async with anyio.create_task_group() as group:
            group.start_soon(drain_first)
            with anyio.fail_after(10):
                await transport.started.wait()
            group.start_soon(end_first)
            await anyio.lowlevel.checkpoint()
            try:
                with anyio.fail_after(2):
                    other = await consumer.consume(
                        _handoff(
                            camera_id=CameraId(camera_id),
                            session_id=CameraSessionId(session_id),
                            db_generation=camera_session.generation_id,
                            source_generation=source_generation,
                            track_id=32,
                        )
                    )
            finally:
                transport.release.set()

        assert other.outcome.value == "queued"
        assert drained[0].outcome.value == "published"
        assert ended[0].outcome.value == "ended"
        other_published = await consumer.drain_one()
        assert other_published is not None
        assert other_published.outcome.value == "published"
        async with database.transaction() as session:
            appearances = tuple(
                await session.scalars(
                    select(Appearance).where(Appearance.camera_id == camera_id)
                )
            )
        assert len(appearances) == 2
        ended_appearance = next(row for row in appearances if row.track_id == 31)
        assert ended_appearance.ended_at == datetime(2026, 9, 7, 2, 5, tzinfo=UTC)
    finally:
        transport.release.set()
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_cancelled_drain_restores_popped_work_ahead_of_queued_items(  # noqa: PLR0915
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    transport = _ClipTransport()
    publisher = AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(transport),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=_Budget(),
        defer_publication=True,
    )
    consumer = AppearanceHandoffConsumer(publisher)
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication cancelled drain {uuid4().hex}",
                source_url=_source(f"cancelled-drain-{uuid4().hex}"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = camera.id
            session_id = camera_session.id

        source_generation = SourceGenerationId(uuid4())
        first = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                track_id=41,
            )
        )
        second = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                track_id=42,
            )
        )
        assert first.outcome.value == "queued"
        assert second.outcome.value == "queued"

        pending = await publisher._take_next()  # noqa: SLF001
        assert pending is not None
        state = next(
            track_state
            for key, track_state in publisher._states.items()  # noqa: SLF001
            if key.track_id == 41
        )
        state.lock.acquire_nowait()

        async def cancel_before_track_lock() -> None:
            with anyio.CancelScope() as scope:
                scope.cancel()
                await publisher._publish_pending(pending)  # noqa: SLF001

        try:
            async with anyio.create_task_group() as group:
                group.start_soon(cancel_before_track_lock)
        finally:
            state.lock.release()

        assert publisher.pending_embeddings == 2
        assert transport.image_calls == 0
        retried = await consumer.drain_one()
        assert retried is not None
        assert retried.outcome.value == "published"
        assert retried.appearance_id == first.appearance_id
        assert publisher.pending_embeddings == 1
        following = await consumer.drain_one()
        assert following is not None
        assert following.outcome.value == "published"
        assert following.appearance_id == second.appearance_id
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_paused_inflight_publication_survives_queue_refill(
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    budget = _PauseOnceBudget()
    publisher = AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(_ClipTransport()),
        model_id="synthetic/clip",
        model_revision="fixture",
        writer_budget=budget,
        queue=PendingEmbeddingQueue(max_pending=1),
        defer_publication=True,
    )
    consumer = AppearanceHandoffConsumer(publisher)
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication paused refill {uuid4().hex}",
                source_url=_source(f"paused-refill-{uuid4().hex}"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = camera.id
            session_id = camera_session.id

        source_generation = SourceGenerationId(uuid4())
        first = await consumer.consume(
            _handoff(
                camera_id=CameraId(camera_id),
                session_id=CameraSessionId(session_id),
                db_generation=camera_session.generation_id,
                source_generation=source_generation,
                track_id=51,
            )
        )
        assert first.outcome.value == "queued"
        drained: list[PublicationAck | None] = []

        async def drain_first() -> None:
            drained.append(await consumer.drain_one())

        async with anyio.create_task_group() as group:
            group.start_soon(drain_first)
            with anyio.fail_after(10):
                await budget.started.wait()
            second = await consumer.consume(
                _handoff(
                    camera_id=CameraId(camera_id),
                    session_id=CameraSessionId(session_id),
                    db_generation=camera_session.generation_id,
                    source_generation=source_generation,
                    track_id=52,
                )
            )
            budget.release.set()

        assert second.outcome.value == "queued"
        assert drained[0] is not None
        assert drained[0].outcome.value == "paused"
        assert publisher.pending_embeddings == 2
        retried = await consumer.drain_one()
        assert retried is not None
        assert retried.appearance_id == first.appearance_id
        assert retried.outcome.value == "published"
        following = await consumer.drain_one()
        assert following is not None
        assert following.appearance_id == second.appearance_id
        assert following.outcome.value == "published"
    finally:
        budget.release.set()
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_publisher_commits_crop_and_vector_atomically(  # noqa: PLR0915
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera_id: UUID | None = None
    session_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session, name="Publication", source_url=_source("publication")
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = CameraId(camera.id)
            session_id = CameraSessionId(camera_session.id)
            db_generation = camera_session.generation_id
        handoff = _handoff(camera_id=camera_id, session_id=session_id, db_generation=db_generation)
        store = CropObjectStore(tmp_path / "crops")
        budget = _Budget()
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(_ClipTransport()),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=budget,
        )

        acknowledgement = await publisher.accept_handoff(handoff, now_monotonic=2.0)

        assert acknowledgement.outcome.value == "published"
        async with database.transaction() as session:
            appearance = await session.get(Appearance, UUID(str(acknowledgement.appearance_id)))
            assert appearance is not None
            assert appearance.first_seen == datetime(2026, 9, 7, 2, 0, tzinfo=UTC)
            assert appearance.embedding is not None
        assert budget.snapshots
        assert list((tmp_path / "crops").rglob("*.jpg"))

        stale = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                track_id=4,
                camera_version=2,
            ),
            now_monotonic=2.0,
        )
        assert stale.outcome.value == "stale_generation"

        failing = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(_ClipTransport(fail=True)),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=budget,
        )
        embedding_failure = await failing.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                track_id=5,
            ),
            now_monotonic=2.0,
        )
        assert embedding_failure.outcome.value == "embedding_failed"

        with patch.object(
            StorageRepository,
            "publish_appearance",
            new=AsyncMock(side_effect=RuntimeError("commit failure")),
        ):
            commit_failure = await publisher.accept_handoff(
                _handoff(
                    camera_id=camera_id,
                    session_id=session_id,
                    db_generation=db_generation,
                    track_id=6,
                ),
                now_monotonic=2.0,
            )
        assert commit_failure.outcome.value == "commit_failed"
        async with database.transaction() as session:
            absent = await session.scalar(
                select(Appearance).where(
                    Appearance.camera_id == camera_id, Appearance.track_id == 6
                )
            )
            assert absent is None

        orphan = store.write(b"orphan")
        temporary = store.root / "aa" / "bb" / f".{uuid4()}.tmp"
        _ = temporary.parent.mkdir(parents=True, exist_ok=True)
        _ = temporary.write_bytes(b"partial")
        reconciliation = await publisher.reconcile_orphans()
        assert reconciliation.removed_jpegs == 1
        assert reconciliation.removed_temps == 1
        assert not (store.root / orphan.object_key).exists()

        ended = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=handoff.generation.source_generation_id,
                kind=LifecycleKind.END,
            ),
            now_monotonic=5.0,
        )
        assert ended.outcome.value == "ended"
        async with database.transaction() as session:
            persisted = await session.scalar(
                select(Appearance).where(
                    Appearance.camera_id == camera_id, Appearance.track_id == 3
                )
            )
            assert persisted is not None
            assert persisted.ended_at is not None
    finally:
        if camera_id is not None and session_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.id == session_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_same_version_reconnect_rejects_late_old_publication(
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    cameras = CameraService(CameraRepository(repository))
    store = CropObjectStore(tmp_path / "crops")
    camera_id: CameraId | None = None
    try:
        async with database.transaction() as session:
            mutation = await cameras.create(
                session,
                CameraCreateRequest.model_validate(
                    {"name": "Publication reconnect", "source_url": "rtsp://fixture:8554/reconnect"}
                ),
            )
        activation = mutation.lifecycle.activation
        assert activation is not None
        camera_id = activation.camera_id
        source_generation = SourceGenerationId(uuid4())
        binding = GenerationBinding(
            camera_id=activation.camera_id,
            camera_session_id=activation.session_id,
            db_generation_id=activation.generation_id,
            source_generation_id=source_generation,
            camera_version=activation.version,
        )
        transport = _DelayedClipTransport()
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(transport),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=_Budget(),
        )
        acknowledgements: list[PublicationAck] = []

        async def publish() -> None:
            acknowledgements.append(
                await publisher.accept_handoff(
                    _handoff(
                        camera_id=activation.camera_id,
                        session_id=activation.session_id,
                        db_generation=activation.generation_id,
                        source_generation=source_generation,
                        track_id=31,
                    ),
                    now_monotonic=2.0,
                )
            )

        async with anyio.create_task_group() as group:
            group.start_soon(publish)
            with anyio.fail_after(10):
                await transport.started.wait()
            async with database.transaction() as session:
                _ = await cameras.reconnect(session, binding)
            transport.release.set()

        assert len(acknowledgements) == 1
        assert acknowledgements[0].outcome.value == "stale_generation"
        async with database.transaction() as session:
            camera = await session.get(Camera, UUID(str(activation.camera_id)))
            sessions = tuple(
                await session.scalars(
                    select(CameraSession)
                    .where(CameraSession.camera_id == UUID(str(activation.camera_id)))
                    .order_by(CameraSession.started_at, CameraSession.id)
                )
            )
            appearances = tuple(
                await session.scalars(
                    select(Appearance).where(
                        Appearance.camera_id == UUID(str(activation.camera_id))
                    )
                )
            )
        assert camera is not None
        assert camera.version == activation.version
        assert len(sessions) == 2
        assert sessions[0].ended_at is not None
        assert sessions[1].ended_at is None
        assert not appearances
        assert not list(store.root.rglob("*.jpg"))
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == UUID(str(camera_id)))
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == UUID(str(camera_id)))
                )
                _ = await session.execute(delete(Camera).where(Camera.id == UUID(str(camera_id))))
        await database.close()


@pytest.mark.anyio
async def test_end_after_reconnect_keeps_true_end_timestamp(
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    cameras = CameraService(CameraRepository(repository))
    store = CropObjectStore(tmp_path / "crops")
    camera_id: CameraId | None = None
    try:
        async with database.transaction() as session:
            mutation = await cameras.create(
                session,
                CameraCreateRequest.model_validate(
                    {"name": "Publication end", "source_url": "rtsp://fixture:8554/end"}
                ),
            )
        activation = mutation.lifecycle.activation
        assert activation is not None
        camera_id = activation.camera_id
        source_generation = SourceGenerationId(uuid4())
        binding = GenerationBinding(
            camera_id=activation.camera_id,
            camera_session_id=activation.session_id,
            db_generation_id=activation.generation_id,
            source_generation_id=source_generation,
            camera_version=activation.version,
        )
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(_ClipTransport()),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=_Budget(),
        )
        initial = await publisher.accept_handoff(
            _handoff(
                camera_id=activation.camera_id,
                session_id=activation.session_id,
                db_generation=activation.generation_id,
                source_generation=source_generation,
                track_id=32,
            ),
            now_monotonic=2.0,
        )
        assert initial.outcome.value == "published"
        async with database.transaction() as session:
            _ = await cameras.reconnect(session, binding)

        ended = await publisher.accept_handoff(
            _handoff(
                camera_id=activation.camera_id,
                session_id=activation.session_id,
                db_generation=activation.generation_id,
                source_generation=source_generation,
                track_id=32,
                kind=LifecycleKind.END,
            ),
            now_monotonic=5.0,
        )

        assert ended.outcome.value == "ended"
        async with database.transaction() as session:
            appearance = await session.get(Appearance, UUID(str(initial.appearance_id)))
        assert appearance is not None
        assert appearance.ended_at == datetime(2026, 9, 7, 2, 0, tzinfo=UTC)
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == UUID(str(camera_id)))
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == UUID(str(camera_id)))
                )
                _ = await session.execute(delete(Camera).where(Camera.id == UUID(str(camera_id))))
        await database.close()


@pytest.mark.anyio
async def test_late_update_after_end_does_not_advance_last_seen(
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    camera_id: CameraId | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session, name=f"Publication late update {uuid4().hex}", source_url=_source("late")
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = CameraId(camera.id)
            session_id = CameraSessionId(camera_session.id)
            db_generation = camera_session.generation_id
        source_generation = SourceGenerationId(uuid4())
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(_ClipTransport()),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=_Budget(),
        )
        started = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                observed_at=datetime(2026, 9, 7, 2, 0, tzinfo=UTC),
            )
        )
        ended = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                kind=LifecycleKind.END,
                observed_at=datetime(2026, 9, 7, 2, 1, tzinfo=UTC),
            )
        )
        assert started.outcome.value == "published"
        assert ended.outcome.value == "ended"
        async with database.transaction() as session:
            before = await session.get(Appearance, UUID(str(started.appearance_id)))
        assert before is not None

        # When: a restarted worker delivers a later update after the track already ended
        restarted_publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(_ClipTransport()),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=_Budget(),
        )
        late = await restarted_publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                kind=LifecycleKind.UPDATE,
                observed_at=datetime(2026, 9, 7, 2, 2, tzinfo=UTC),
            )
        )

        # Then: the closed appearance remains immutable and the late event is ignored
        assert late.outcome.value == "noop"
        async with database.transaction() as session:
            after = await session.get(Appearance, UUID(str(started.appearance_id)))
        assert after is not None
        assert after.last_seen == before.last_seen
        assert after.ended_at == before.ended_at
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()


@pytest.mark.anyio
async def test_retention_suppression_uses_stable_appearance_id(  # noqa: PLR0915
    database_url: str, tmp_path: Path
) -> None:
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    clip = _ClipTransport()
    budget = _Budget()
    camera_id: CameraId | None = None
    session_id: CameraSessionId | None = None
    previous_days: int | None = None
    previous_quota: int | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session,
                name=f"Publication quota-{uuid4().hex}",
                source_url=_source("publication-quota"),
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="fixture"
            )
            camera_id = CameraId(camera.id)
            session_id = CameraSessionId(camera_session.id)
            db_generation = camera_session.generation_id

        source_generation = SourceGenerationId(uuid4())
        assert source_generation != SourceGenerationId(db_generation)
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=store,
            clip=ClipAdapter(clip),
            model_id="synthetic/clip",
            model_revision="fixture",
            writer_budget=budget,
            monotonic_clock=lambda: 10.0,
        )
        consumer = AppearanceHandoffConsumer(publisher)
        old = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                track_id=3,
                observed_at=datetime(2026, 9, 2, 2, 0, tzinfo=UTC),
            ),
            now_monotonic=10.0,
        )
        unrelated = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                track_id=4,
                observed_at=datetime(2026, 9, 6, 2, 0, tzinfo=UTC),
            ),
            now_monotonic=10.0,
        )
        assert old.outcome.value == "published"
        assert unrelated.outcome.value == "published"
        assert clip.image_calls == 2

        crop_sizes = [path.stat().st_size for path in store.root.rglob("*.jpg")]
        assert len(crop_sizes) == 2
        async with database.transaction() as session:
            old_row = await session.get(Appearance, UUID(str(old.appearance_id)))
        assert old_row is not None
        old_crop = store.root / old_row.crop_object_key
        quota = int((max(crop_sizes) + 1) / 0.95) + 2
        async with database.transaction() as session:
            settings = await session.scalar(
                select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
            )
            assert settings is not None
            previous_days = settings.retention_days
            previous_quota = settings.quota_bytes
            settings.retention_days = 7
            settings.quota_bytes = quota

        publisher.writer_budget = None
        for track_id in (3, 4):
            paused = await publisher.accept_handoff(
                _handoff(
                    camera_id=camera_id,
                    session_id=session_id,
                    db_generation=db_generation,
                    source_generation=source_generation,
                    track_id=track_id,
                    kind=LifecycleKind.UPDATE,
                    observed_at=datetime(2026, 9, 2 if track_id == 3 else 9, 2, tzinfo=UTC),
                    bounding_box=(1.0, 1.0, 99.0, 199.0),
                ),
                now_monotonic=20.0,
            )
            assert paused.outcome.value in {"paused", "queued"}
        assert len(publisher.queue) == 2

        retention = RetentionService(
            database=database,
            storage=cast(
                "StorageRepository",
                cast("object", _RetentionStorageAdapter(repository, relation_bytes=0)),
            ),
            crop_store=store,
            publisher=publisher,
        )
        report = await retention.sweep(now=datetime(2026, 9, 7, 12, 0, tzinfo=UTC))
        assert report.quota_candidates == 1
        assert report.suppressed_tracks == 1
        assert len(publisher.queue) == 1

        suppressed = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                track_id=3,
                kind=LifecycleKind.UPDATE,
                observed_at=datetime(2026, 9, 2, 2, 0, tzinfo=UTC),
                bounding_box=(0.0, 0.0, 100.0, 200.0),
            ),
            now_monotonic=30.0,
        )
        assert suppressed.outcome.value == "suppressed"
        assert clip.image_calls == 2

        publisher.writer_budget = budget
        published_unrelated = await consumer.drain_one()
        assert published_unrelated is not None
        assert published_unrelated.outcome.value == "published"
        assert published_unrelated.appearance_id == unrelated.appearance_id
        assert clip.image_calls == 3
        assert consumer.stats.pending_embeddings == 0

        ended = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                track_id=3,
                kind=LifecycleKind.END,
                observed_at=datetime(2026, 9, 2, 2, 0, tzinfo=UTC),
            ),
            now_monotonic=30.0,
        )
        assert ended.outcome.value == "noop"
        after_end = await publisher.accept_handoff(
            _handoff(
                camera_id=camera_id,
                session_id=session_id,
                db_generation=db_generation,
                source_generation=source_generation,
                track_id=3,
                kind=LifecycleKind.UPDATE,
                include_candidate=False,
                observed_at=datetime(2026, 9, 2, 2, 0, tzinfo=UTC),
            ),
            now_monotonic=40.0,
        )
        assert after_end.outcome.value == "noop"
        assert not old_crop.exists()
    finally:
        if previous_days is not None and previous_quota is not None:
            async with database.transaction() as session:
                settings = await session.scalar(
                    select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
                )
                if settings is not None:
                    settings.retention_days = previous_days
                    settings.quota_bytes = previous_quota
        if camera_id is not None:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CropGarbage).where(
                        CropGarbage.appearance_id.in_(
                            select(Appearance.id).where(Appearance.camera_id == camera_id)
                        )
                    )
                )
                _ = await session.execute(
                    delete(Appearance).where(Appearance.camera_id == camera_id)
                )
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id == camera_id)
                )
                _ = await session.execute(delete(Camera).where(Camera.id == camera_id))
        await database.close()
