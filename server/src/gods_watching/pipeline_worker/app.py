"""Compose the pipeline worker and durable CLIP transition lifecycle."""

# ruff: noqa: TC001, TC003, PLR0913, SIM117, E501

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, final, override

import anyio

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    AppearancePublisher,
    ConservativeWriterBudget,
    PublicationOutcome,
)
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.inference.clip import ClipAdapter, ClipRuntimeManager, TritonClipTransport
from gods_watching.inference.detector import DetectorClient, TritonGrpcDetectorTransport
from gods_watching.ingest import IngestCoordinator
from gods_watching.live_detections import LiveDetectionPublisher
from gods_watching.model_selection import ClipModelRegistry
from gods_watching.model_selection.coordinator import TransitionCoordinator
from gods_watching.model_selection.models import TransitionRecoveryError
from gods_watching.model_selection.registry import ClipModelPackage
from gods_watching.model_selection.service import ModelSelectionService, PipelineLifecyclePort
from gods_watching.retention import RetentionService
from gods_watching.status import StatusReporter, WorkerStatusSnapshot
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

from .service import PipelineWorker
from .settings import PipelineWorkerSettings

if TYPE_CHECKING:
    from anyio.abc import TaskGroup


def _coordinator_binding(
    coordinator: IngestCoordinator,
) -> Callable[[CameraId], Awaitable[GenerationBinding | None]]:
    async def active_binding(camera_id: CameraId) -> GenerationBinding | None:
        for worker in coordinator.workers:
            if worker.camera_id == camera_id:
                return worker.generation
        return None

    return active_binding


@dataclass(slots=True)
class _Generation:
    """One fully joined pipeline/publication/retention cycle."""

    package: ClipModelPackage
    stop_event: anyio.Event
    done_event: anyio.Event
    coordinator: IngestCoordinator
    consumer: AppearanceHandoffConsumer
    retention: RetentionService
    pipeline: PipelineWorker
    live_detection: LiveDetectionPublisher
    clip_transport: TritonClipTransport


@dataclass(slots=True)
class _TransitionTaskControl:
    """Expose cancellation of the transition loop during worker shutdown."""

    scope: anyio.CancelScope | None = None

    def cancel(self) -> None:
        """Cancel an in-flight transition while preserving its durable job."""
        if self.scope is not None:
            self.scope.cancel()


@final
class _PipelineLifecycle(PipelineLifecyclePort):
    """Start and join generations under one worker-owned task group."""

    def __init__(
        self,
        *,
        settings: PipelineWorkerSettings,
        database: Database,
        storage: StorageRepository,
        crop_store: CropObjectStore,
        detector: DetectorClient,
        detector_transport: TritonGrpcDetectorTransport,
        task_group: TaskGroup,
    ) -> None:
        self._settings = settings
        self._database = database
        self._storage = storage
        self._crop_store = crop_store
        self._detector = detector
        self._detector_transport = detector_transport
        self._task_group = task_group
        self._generation: _Generation | None = None

    @property
    def generation(self) -> _Generation | None:
        """Return the currently running generation for status/debugging."""
        return self._generation

    @override
    async def start(self, package: ClipModelPackage) -> None:
        """Construct a fresh publisher/coordinator bound to one package."""
        if self._generation is not None:
            await self.stop_and_join()
        coordinator = IngestCoordinator()
        budget = ConservativeWriterBudget()
        clip_transport = TritonClipTransport(
            self._settings.triton_grpc_url,
            package=package,
        )
        publisher = AppearancePublisher(
            database=self._database,
            storage=self._storage,
            crop_store=self._crop_store,
            clip=ClipAdapter(clip_transport, package=package),
            model_id=package.model_id,
            model_revision=package.revision,
            writer_budget=budget,
            active_binding=_coordinator_binding(coordinator),
        )
        consumer = AppearanceHandoffConsumer(publisher)
        _ = await consumer.start()
        live_detection = LiveDetectionPublisher(self._database)
        pipeline = PipelineWorker(
            database=self._database,
            cameras=CameraService(CameraRepository(self._storage)),
            coordinator=coordinator,
            consumer=consumer,
            detector=self._detector,
            poll_seconds=self._settings.worker_poll_seconds,
            live_detection_sink=live_detection.publish,
        )
        retention = RetentionService(
            database=self._database,
            storage=self._storage,
            crop_store=self._crop_store,
            publisher=publisher,
            writer_budget=budget,
        )
        generation = _Generation(
            package=package,
            stop_event=anyio.Event(),
            done_event=anyio.Event(),
            coordinator=coordinator,
            consumer=consumer,
            retention=retention,
            pipeline=pipeline,
            live_detection=live_detection,
            clip_transport=clip_transport,
        )
        self._generation = generation
        self._task_group.start_soon(self._run_generation, generation)

    @override
    async def stop_and_join(self) -> None:
        """Signal one generation and wait for terminal handoffs to settle."""
        generation = self._generation
        if generation is None:
            return
        with anyio.CancelScope(shield=True):
            generation.stop_event.set()
            await generation.done_event.wait()
        if self._generation is generation:
            self._generation = None

    async def _run_generation(self, generation: _Generation) -> None:
        try:
            async with anyio.create_task_group() as task_group:
                task_group.start_soon(_run_pipeline, generation.pipeline, generation.stop_event)
                task_group.start_soon(generation.live_detection.run, generation.stop_event)
                task_group.start_soon(
                    _drain_publications,
                    generation.consumer,
                    generation.stop_event,
                    self._settings.worker_poll_seconds,
                )
                task_group.start_soon(
                    _report_status,
                    _StatusLoop(
                        database=self._database,
                        reporter=StatusReporter(generation.coordinator),
                        retention=generation.retention,
                        detector=self._detector_transport,
                        consumer=generation.consumer,
                    ),
                    generation.stop_event,
                    self._settings.worker_poll_seconds,
                )
                await generation.retention.run_forever(generation.stop_event)
                task_group.cancel_scope.cancel()
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await generation.clip_transport.__aexit__(None, None, None)
            finally:
                generation.done_event.set()


async def run_pipeline_worker(
    settings: PipelineWorkerSettings,
    *,
    lock_path: Path,
    stop_event: anyio.Event,
) -> None:
    """Run one process-owned pipeline and its durable transition coordinator."""
    del lock_path  # the durable active identity is the runtime source of truth
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.camera_cipher_key.encode()))
    crop_store = CropObjectStore(settings.crops_root)
    detector_transport = TritonGrpcDetectorTransport(url=settings.triton_grpc_url)
    detector = DetectorClient(transport=detector_transport)
    registry = ClipModelRegistry()
    from gods_watching.model_selection.assets import PreparedModelCatalog  # noqa: PLC0415

    prepared = PreparedModelCatalog(
        registry,
        settings.model_lock_path,
        assets_root=settings.model_assets_root,
    )
    coordinator = TransitionCoordinator(database)
    selection = ModelSelectionService(
        database=database,
        registry=registry,
        prepared=prepared,
        coordinator=coordinator,
        storage=storage,
    )
    runtime = ClipRuntimeManager(settings.triton_grpc_url)
    try:
        async with coordinator.worker_ownership():
            async with runtime:
                async with anyio.create_task_group() as task_group:
                    lifecycle = _PipelineLifecycle(
                        settings=settings,
                        database=database,
                        storage=storage,
                        crop_store=crop_store,
                        detector=detector,
                        detector_transport=detector_transport,
                        task_group=task_group,
                    )
                    _ = await selection.recover_startup(runtime=runtime, pipeline=lifecycle)
                    transition_done = anyio.Event()
                    transition_control = _TransitionTaskControl()
                    task_group.start_soon(
                        _run_model_transitions,
                        selection,
                        lifecycle,
                        crop_store,
                        runtime,
                        settings,
                        stop_event,
                        transition_done,
                        transition_control,
                    )
                    await stop_event.wait()
                    transition_control.cancel()
                    with anyio.CancelScope(shield=True):
                        await transition_done.wait()
                    await lifecycle.stop_and_join()
                    task_group.cancel_scope.cancel()
    finally:
        await detector_transport.close()
        await database.close()


async def _run_model_transitions(
    selection: ModelSelectionService,
    lifecycle: _PipelineLifecycle,
    crop_store: CropObjectStore,
    runtime: ClipRuntimeManager,
    settings: PipelineWorkerSettings,
    stop_event: anyio.Event,
    done_event: anyio.Event,
    control: _TransitionTaskControl,
) -> None:
    with anyio.CancelScope() as scope:
        control.scope = scope
        try:
            while not stop_event.is_set():
                try:
                    transition_transport = TritonClipTransport(settings.triton_grpc_url)
                    try:
                        _ = await selection.run_pending(
                            crop_store=crop_store,
                            runtime=runtime,
                            clip_factory=lambda package, transport=transition_transport: ClipAdapter(
                                transport,
                                package=package,
                            ),
                            pipeline=lifecycle,
                        )
                    finally:
                        with anyio.CancelScope(shield=True):
                            await transition_transport.__aexit__(None, None, None)
                except TransitionRecoveryError as error:
                    if (
                        "durable identity unavailable" in str(error)
                        or "does not match transition endpoints" in str(error)
                    ):
                        raise
                # Pace both idle polling and rolling_back retry.  A failed
                # restore remains visible in maintenance and must not hammer
                # Triton/DB.
                with anyio.move_on_after(settings.worker_poll_seconds):
                    await stop_event.wait()
        finally:
            done_event.set()


async def _run_pipeline(pipeline: PipelineWorker, stop_event: anyio.Event) -> None:
    await pipeline.run(stop_event=stop_event)


async def _drain_publications(
    consumer: AppearanceHandoffConsumer,
    stop_event: anyio.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        acknowledgement = await consumer.drain_one()
        if acknowledgement is None or acknowledgement.outcome is PublicationOutcome.PAUSED:
            with anyio.move_on_after(interval_seconds):
                await stop_event.wait()


@dataclass(frozen=True, slots=True)
class _StatusLoop:
    database: Database
    reporter: StatusReporter
    retention: RetentionService
    detector: TritonGrpcDetectorTransport
    consumer: AppearanceHandoffConsumer


async def _report_status(
    loop: _StatusLoop,
    stop_event: anyio.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        accounting = await loop.retention.accounting()
        publication = loop.consumer.stats
        async with loop.database.transaction() as session:
            await loop.reporter.report(
                session,
                snapshot=WorkerStatusSnapshot(
                    observed_at=datetime.now(UTC),
                    inference_ready=await loop.detector.ready(),
                    persistence_paused=(
                        accounting.cleanup_required or accounting.filesystem_guard_active
                    ),
                    storage_managed_bytes=accounting.managed_bytes,
                    storage_quota_bytes=accounting.quota_bytes,
                    indexing_queue_depth=publication.pending_embeddings,
                    last_searchable_latency_seconds=(publication.last_searchable_latency_seconds),
                ),
            )
        with anyio.move_on_after(interval_seconds):
            await stop_event.wait()


__all__ = ["run_pipeline_worker"]
