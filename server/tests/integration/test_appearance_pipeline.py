from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from typing import TYPE_CHECKING, final
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from PIL import Image
from sqlalchemy import delete

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    AppearancePublisher,
    BudgetLease,
    BudgetSnapshot,
)
from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.contracts.search import BrowseSearchRequest
from gods_watching.inference.clip import ClipAdapter
from gods_watching.inference.detector import Detection, DetectorRequest, DetectorResult
from gods_watching.ingest.models import DecodedFrame
from gods_watching.ingest.worker import IngestWorker, IngestWorkerConfiguration
from gods_watching.media.models import SourceGenerationId
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.storage import (
    Appearance,
    Camera,
    CameraSession,
    CredentialCipher,
    CropGarbage,
    CropObjectStore,
    Database,
    StorageRepository,
)
from gods_watching.tracking import DetectorInputReference

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import AnyUrl


@final
class _Budget:
    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease | None:
        return BudgetLease(reserved_bytes=snapshot.new_crop_bytes)


@final
class _ClipTransport:
    async def embed_image(self, image: bytes) -> tuple[float, ...]:
        with Image.open(BytesIO(image)) as decoded:
            assert decoded.format == "JPEG"
            assert decoded.mode == "RGB"
        return (1.0,) + (0.0,) * 511

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0,) + (0.0,) * 511


@final
class _Detector:
    async def detect(self, request: DetectorRequest) -> DetectorResult:
        return DetectorResult(detections=(Detection(80, 20, 160, 200, request.confidence, 0),))


def _source() -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": "Appearance pipeline", "source_url": "rtsp://fixture:8554/person"}
    ).source_url


def _frame(binding: GenerationBinding, *, sequence: int) -> DecodedFrame:
    width, height = 320, 240
    pixels = bytes((index + sequence) % 256 for index in range(width * height * 3))
    return DecodedFrame(
        source_generation_id=binding.source_generation_id,
        ingress_utc=datetime(2026, 9, 13, 1, 0, sequence, tzinfo=UTC),
        ingress_monotonic=float(sequence + 1),
        reference=DetectorInputReference(f"publication-{sequence}"),
        width=width,
        height=height,
        rgb_bytes=pixels,
        encoded_image=b"detector-input",
    )


@pytest.mark.anyio
async def test_ingest_start_is_searchable_before_track_exit(
    database_url: str, tmp_path: Path
) -> None:
    # Given: one real ingest worker connected to the appearance publication consumer.
    database = Database.connect(database_url)
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    crop_store = CropObjectStore(tmp_path / "crops")
    camera_id: UUID | None = None
    try:
        async with database.transaction() as session:
            camera = await repository.add_camera(
                session, name=f"Appearance pipeline {uuid4().hex}", source_url=_source()
            )
            camera_session = await repository.start_camera_session(
                session, camera.id, cause="integration"
            )
            camera_id = camera.id
        binding = GenerationBinding(
            camera_id=CameraId(camera.id),
            camera_session_id=CameraSessionId(camera_session.id),
            db_generation_id=CameraGenerationId(camera_session.generation_id),
            source_generation_id=SourceGenerationId(uuid4()),
            camera_version=camera.version,
        )
        publisher = AppearancePublisher(
            database=database,
            storage=repository,
            crop_store=crop_store,
            clip=ClipAdapter(_ClipTransport()),
            model_id="synthetic/clip",
            model_revision="pipeline-fixture",
            writer_budget=_Budget(),
            monotonic_clock=lambda: 10.0,
        )
        consumer = AppearanceHandoffConsumer(publisher)
        reconciliation = await consumer.start()
        worker = IngestWorker(
            IngestWorkerConfiguration(
                generation=binding,
                source_url="rtsp://fixture:8554/person",
                monotonic_clock=lambda: 10.0,
            ),
            _Detector(),
            consumer,
        )

        # When: ByteTrack confirms the person and emits its canonical START handoff.
        worker.receive(_frame(binding, sequence=0))
        starts = await worker.sample_once()

        # Then: the active appearance, vector, and retrievable JPEG are committed before END.
        assert reconciliation.removed_jpegs == 0
        assert len(starts) == 1
        assert consumer.last_ack is not None
        assert consumer.last_ack.outcome.value == "published"
        async with database.transaction() as session:
            response = await SearchService(SearchRepository()).search(
                session,
                BrowseSearchRequest(mode="browse", camera_ids=(CameraId(camera.id),), limit=10),
            )
            assert len(response.results) == 1
            appearance_id = UUID(str(response.results[0].appearance_id))
            crop = await AppearanceLookupService(SearchRepository(), crop_store).get_crop(
                session, appearance_id
            )
        assert response.results[0].ended_at is None
        assert crop.representative_version == 1
        with Image.open(BytesIO(crop.payload)) as decoded:
            assert decoded.format == "JPEG"
            assert decoded.size == (80, 180)
    finally:
        if camera_id is not None:
            async with database.transaction() as session:
                appearance_ids = tuple(
                    await session.scalars(
                        Appearance.__table__.select()
                        .with_only_columns(Appearance.id)
                        .where(Appearance.camera_id == camera_id)
                    )
                )
                if appearance_ids:
                    _ = await session.execute(
                        delete(CropGarbage).where(CropGarbage.appearance_id.in_(appearance_ids))
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
async def test_consumer_start_reconciles_interrupted_crop_write(
    database_url: str, tmp_path: Path
) -> None:
    # Given: an orphan JPEG left before its database transaction committed.
    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "crops")
    orphan = crop_store.write(b"interrupted-publication")
    publisher = AppearancePublisher(
        database=database,
        storage=StorageRepository(CredentialCipher(Fernet.generate_key())),
        crop_store=crop_store,
        clip=ClipAdapter(_ClipTransport()),
        model_id="synthetic/clip",
        model_revision="pipeline-fixture",
        writer_budget=_Budget(),
    )
    consumer = AppearanceHandoffConsumer(publisher)
    try:
        # When: the process-owned consumer performs startup recovery.
        report = await consumer.start()

        # Then: the unreferenced object is removed before new handoffs are accepted.
        assert report.removed_jpegs == 1
        assert not (crop_store.root / orphan.object_key).exists()
        assert await consumer.start() == report
    finally:
        await database.close()


@pytest.mark.anyio
async def test_consumer_start_leaves_unmanaged_paths_under_the_crop_root(
    database_url: str, tmp_path: Path
) -> None:
    # Given: generated orphans next to operator files and a crop-named symlink to outside data
    database = Database.connect(database_url)
    crop_store = CropObjectStore(tmp_path / "crops")
    orphan = crop_store.write(b"interrupted-publication")
    generated_temp = crop_store.root / orphan.object_key.rsplit("/", 1)[0] / f".{uuid4()}.tmp"
    _ = generated_temp.write_bytes(b"interrupted-write")
    outside = tmp_path / "outside.jpg"
    _ = outside.write_bytes(b"outside-data")
    link_parent = crop_store.root / "00" / "00"
    link_parent.mkdir(parents=True)
    link = link_parent / "00000000-0000-4000-8000-000000000000.jpg"
    link.symlink_to(outside)
    operator_jpeg = crop_store.root / "notes.jpg"
    _ = operator_jpeg.write_bytes(b"operator-file")
    operator_temp = link_parent / "partial.tmp"
    _ = operator_temp.write_bytes(b"operator-temp")
    publisher = AppearancePublisher(
        database=database,
        storage=StorageRepository(CredentialCipher(Fernet.generate_key())),
        crop_store=crop_store,
        clip=ClipAdapter(_ClipTransport()),
        model_id="synthetic/clip",
        model_revision="pipeline-fixture",
        writer_budget=_Budget(),
    )
    try:
        # When: the consumer performs startup recovery
        _ = await AppearanceHandoffConsumer(publisher).start()

        # Then: only generated objects are removed; unmanaged paths and link targets remain
        assert not (crop_store.root / orphan.object_key).exists()
        assert not generated_temp.exists()
        assert link.is_symlink()
        assert outside.read_bytes() == b"outside-data"
        assert operator_jpeg.read_bytes() == b"operator-file"
        assert operator_temp.read_bytes() == b"operator-temp"
    finally:
        await database.close()
