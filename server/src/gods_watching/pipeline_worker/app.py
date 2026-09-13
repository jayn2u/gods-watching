"""Compose the pipeline worker process from its settings."""

from collections.abc import Awaitable, Callable
from pathlib import Path

import anyio

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    AppearancePublisher,
    ConservativeWriterBudget,
)
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.inference.detector import DetectorClient, TritonGrpcDetectorTransport
from gods_watching.ingest import IngestCoordinator
from gods_watching.retention import RetentionService
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

from .service import PipelineWorker
from .settings import PipelineWorkerSettings, locked_clip_model


def _coordinator_binding(
    coordinator: IngestCoordinator,
) -> Callable[[CameraId], Awaitable[GenerationBinding | None]]:
    async def active_binding(camera_id: CameraId) -> GenerationBinding | None:
        for worker in coordinator.workers:
            if worker.camera_id == camera_id:
                return worker.generation
        return None

    return active_binding


async def run_pipeline_worker(
    settings: PipelineWorkerSettings,
    *,
    lock_path: Path,
    stop_event: anyio.Event,
) -> None:
    """Run ingest, appearance publication, and retention until the stop event is set."""
    model_id, model_revision = locked_clip_model(lock_path)
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.camera_cipher_key.encode()))
    crop_store = CropObjectStore(settings.crops_root)
    budget = ConservativeWriterBudget()
    coordinator = IngestCoordinator()
    detector_transport = TritonGrpcDetectorTransport(url=settings.triton_grpc_url)
    try:
        async with TritonClipTransport(settings.triton_grpc_url) as clip_transport:
            publisher = AppearancePublisher(
                database=database,
                storage=storage,
                crop_store=crop_store,
                clip=ClipAdapter(clip_transport),
                model_id=model_id,
                model_revision=model_revision,
                writer_budget=budget,
                active_binding=_coordinator_binding(coordinator),
            )
            consumer = AppearanceHandoffConsumer(publisher)
            _ = await consumer.start()
            pipeline = PipelineWorker(
                database=database,
                cameras=CameraService(CameraRepository(storage)),
                coordinator=coordinator,
                consumer=consumer,
                detector=DetectorClient(transport=detector_transport),
                poll_seconds=settings.worker_poll_seconds,
            )
            retention = RetentionService(
                database=database,
                storage=storage,
                crop_store=crop_store,
                publisher=publisher,
                writer_budget=budget,
            )
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_run_pipeline, pipeline, stop_event)
                await retention.run_forever(stop_event)
    finally:
        await detector_transport.close()
        await database.close()


async def _run_pipeline(pipeline: PipelineWorker, stop_event: anyio.Event) -> None:
    await pipeline.run(stop_event=stop_event)


__all__ = ["run_pipeline_worker"]
