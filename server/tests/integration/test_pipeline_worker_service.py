from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, final, override
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
)
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.cameras import CameraCreateRequest, CameraPatchRequest
from gods_watching.contracts.identifiers import CameraId
from gods_watching.inference.clip import ClipAdapter
from gods_watching.inference.detector import Detection, DetectorRequest, DetectorResult
from gods_watching.ingest import IngestCoordinator, IngestWorker
from gods_watching.ingest.models import DecodedFrame
from gods_watching.pipeline_worker.service import PipelineWorker
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

    from gods_watching.contracts.pipeline import GenerationBinding, PipelineHandoff
    from gods_watching.ingest.worker import (
        DetectorPort,
        IngestWorkerConfiguration,
        PipelineHandoffConsumer,
    )


@final
class _Budget:
    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease | None:
        return BudgetLease(reserved_bytes=snapshot.new_crop_bytes)


@final
class _ClipTransport:
    async def embed_image(self, image: bytes) -> tuple[float, ...]:
        del image
        return (1.0,) + (0.0,) * 511

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return (1.0,) + (0.0,) * 511


@final
class _CountingConsumer:
    """Count END handoffs per track while delegating to real publication."""

    def __init__(self, delegate: AppearanceHandoffConsumer) -> None:
        self._delegate = delegate
        self.end_counts: dict[str, int] = {}

    async def __call__(self, handoff: PipelineHandoff) -> None:
        if str(handoff.lifecycle.kind) == "end":
            key = str(handoff.lifecycle.track_key)
            self.end_counts[key] = self.end_counts.get(key, 0) + 1
        await self._delegate(handoff)


@final
class _Detector:
    async def detect(self, request: DetectorRequest) -> DetectorResult:
        return DetectorResult(detections=(Detection(80, 20, 160, 200, request.confidence, 0),))


class _EarlyExitCoordinator(IngestCoordinator):
    def __init__(self) -> None:
        super().__init__()
        self.added: anyio.Event = anyio.Event()

    @override
    def add(self, worker: IngestWorker) -> None:
        super().add(worker)
        self.added.set()

    @override
    async def run(self, *, stop_event: anyio.Event) -> None:
        await stop_event.wait()


def _frame(binding: GenerationBinding, *, sequence: int) -> DecodedFrame:
    width, height = 320, 240
    return DecodedFrame(
        source_generation_id=binding.source_generation_id,
        ingress_utc=datetime(2026, 9, 13, 1, 0, sequence, tzinfo=UTC),
        ingress_monotonic=float(sequence + 1),
        reference=DetectorInputReference(f"pipeline-worker-{sequence}"),
        width=width,
        height=height,
        rgb_bytes=bytes((index + sequence) % 256 for index in range(width * height * 3)),
        encoded_image=b"detector-input",
    )


def _fixed_clock_worker(
    config: IngestWorkerConfiguration,
    detector: DetectorPort,
    consumer: PipelineHandoffConsumer,
) -> IngestWorker:
    return IngestWorker(replace(config, monotonic_clock=lambda: 10.0), detector, consumer)


async def _delete_camera_rows(database: Database, camera_id: UUID) -> None:
    async with database.transaction() as session:
        appearance_ids = tuple(
            await session.scalars(select(Appearance.id).where(Appearance.camera_id == camera_id))
        )
        if appearance_ids:
            _ = await session.execute(
                delete(CropGarbage).where(CropGarbage.appearance_id.in_(appearance_ids))
            )
        _ = await session.execute(delete(Appearance).where(Appearance.camera_id == camera_id))
        _ = await session.execute(delete(CameraSession).where(CameraSession.camera_id == camera_id))
        _ = await session.execute(delete(Camera).where(Camera.id == camera_id))


@pytest.mark.anyio
async def test_pipeline_worker_follows_camera_edits_and_publishes_on_the_current_version(
    database_url: str, tmp_path: Path
) -> None:
    # Given: an API-committed camera and a pipeline worker with fake detector and CLIP
    database = Database.connect(database_url)
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    cameras = CameraService(CameraRepository(storage))
    publisher = AppearancePublisher(
        database=database,
        storage=storage,
        crop_store=CropObjectStore(tmp_path / "crops"),
        clip=ClipAdapter(_ClipTransport()),
        model_id="openai/clip-vit-base-patch16",
        model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
        writer_budget=_Budget(),
    )
    counting = _CountingConsumer(AppearanceHandoffConsumer(publisher))
    coordinator = IngestCoordinator()
    pipeline = PipelineWorker(
        database=database,
        cameras=cameras,
        coordinator=coordinator,
        consumer=counting,
        detector=_Detector(),
        worker_factory=_fixed_clock_worker,
    )
    camera_id: UUID | None = None
    try:
        async with database.transaction() as session:
            created = await cameras.create(
                session,
                CameraCreateRequest.model_validate(
                    {
                        "name": f"pipeline worker {uuid4().hex[:8]}",
                        "source_url": "rtsp://fixture:8554/person",
                    }
                ),
            )
        camera_id = UUID(str(created.camera.camera_id))

        # When: the worker reconciles for the first time
        _ = await pipeline.reconcile_once()

        # Then: it runs the committed session and version
        (first,) = [w for w in coordinator.workers if w.camera_id == CameraId(camera_id)]
        activation = created.lifecycle.activation
        assert activation is not None
        assert first.generation.camera_session_id == activation.session_id
        assert first.generation.camera_version == created.camera.version

        # When: a threshold edit bumps the version and the worker reconciles again
        async with database.transaction() as session:
            tuned = await cameras.update(
                session,
                created.camera.camera_id,
                CameraPatchRequest.model_validate({"detection_threshold": 0.6}),
                expected_version=created.camera.version,
            )
        _ = await pipeline.reconcile_once()

        # Then: the replacement worker is bound to the new version and can publish
        (second,) = [w for w in coordinator.workers if w.camera_id == CameraId(camera_id)]
        assert second is not first
        assert second.generation.camera_version == tuned.camera.version
        assert second.threshold == 0.6
        second.receive(_frame(second.generation, sequence=0))
        _ = await second.sample_once()
        async with database.transaction() as session:
            published = tuple(
                await session.scalars(select(Appearance).where(Appearance.camera_id == camera_id))
            )
        assert len(published) == 1
        assert published[0].session_id == second.generation.camera_session_id

        # When: detection is disabled and the worker reconciles
        async with database.transaction() as session:
            _ = await cameras.update(
                session,
                created.camera.camera_id,
                CameraPatchRequest.model_validate({"detection_enabled": False}),
                expected_version=tuned.camera.version,
            )
        _ = await pipeline.reconcile_once()

        # Then: the camera no longer runs, its open track ended durably, and each
        # END reached publication exactly once
        assert counting.end_counts
        assert set(counting.end_counts.values()) == {1}
        assert all(w.camera_id != CameraId(camera_id) for w in coordinator.workers)
        async with database.transaction() as session:
            ended = await session.scalar(
                select(Appearance).where(Appearance.camera_id == camera_id)
            )
        assert ended is not None
        assert ended.ended_at is not None
    finally:
        if camera_id is not None:
            await _delete_camera_rows(database, camera_id)
        await database.close()


@pytest.mark.anyio
async def test_pipeline_worker_closes_workers_left_after_coordinator_exit(
    database_url: str,
) -> None:
    database = Database.connect(database_url)
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    cameras = CameraService(CameraRepository(storage))
    coordinator = _EarlyExitCoordinator()
    handoffs: list[PipelineHandoff] = []

    async def consume(handoff: PipelineHandoff) -> None:
        handoffs.append(handoff)

    pipeline = PipelineWorker(
        database=database,
        cameras=cameras,
        coordinator=coordinator,
        consumer=consume,
        detector=_Detector(),
        worker_factory=_fixed_clock_worker,
        poll_seconds=0.01,
    )
    camera_id: UUID | None = None
    try:
        async with database.transaction() as session:
            created = await cameras.create(
                session,
                CameraCreateRequest.model_validate(
                    {
                        "name": f"pipeline shutdown {uuid4().hex[:8]}",
                        "source_url": "rtsp://fixture:8554/person",
                    }
                ),
            )
        camera_id = UUID(str(created.camera.camera_id))
        stop = anyio.Event()

        async def run_pipeline() -> None:
            await pipeline.run(stop_event=stop)

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_pipeline)
            with anyio.fail_after(2.0):
                await coordinator.added.wait()
            worker = coordinator.workers[0]
            worker.receive(_frame(worker.generation, sequence=0))
            _ = await worker.sample_once()
            stop.set()

        assert [handoff.lifecycle.kind.value for handoff in handoffs] == [
            "start",
            "end",
        ]
    finally:
        if camera_id is not None:
            await _delete_camera_rows(database, camera_id)
        await database.close()
