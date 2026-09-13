from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast, final
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import anyio
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete, select

from gods_watching.appearances import (
    AppearancePublisher,
    BudgetLease,
    BudgetSnapshot,
    PublicationAck,
)
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
        published_unrelated = await publisher.process_next()
        assert published_unrelated is not None
        assert published_unrelated.outcome.value == "published"
        assert published_unrelated.appearance_id == unrelated.appearance_id
        assert clip.image_calls == 3

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
