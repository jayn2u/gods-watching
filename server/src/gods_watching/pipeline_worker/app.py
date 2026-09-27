"""Compose the pipeline worker and durable CLIP transition lifecycle."""

# ruff: noqa: TC001, TC003, PLR0913, SIM117

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final, final, override

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
from gods_watching.model_selection.coordinator import TransitionCoordinator
from gods_watching.model_selection.models import TransitionRecoveryError, TransitionResult
from gods_watching.model_selection.registry import (
    ClipModelPackage,
    ClipModelRegistry,
    load_clip_registry,
)
from gods_watching.model_selection.service import ModelSelectionService, PipelineLifecyclePort
from gods_watching.model_selection.transition_observer import TransitionObserver
from gods_watching.retention import RetentionService, StorageAccounting
from gods_watching.status import StatusReporter, WorkerStatusSnapshot
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

from .service import PipelineWorker
from .settings import PipelineWorkerSettings

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

    from gods_watching.model_selection.service import PreparedModelCatalogPort

_ACCOUNTING_REFRESH_INTERVAL_SECONDS: Final = 10.0
_ACCOUNTING_STALE_AFTER_SECONDS: Final = 30.0
_PUBLICATION_SHUTDOWN_DRAIN_SECONDS: Final = 6.0
_LOGGER: Final = logging.getLogger(__name__)


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
            defer_publication=True,
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
            with anyio.move_on_after(_PUBLICATION_SHUTDOWN_DRAIN_SECONDS) as scope:
                await generation.done_event.wait()
            if scope.cancelled_caught:
                pending = generation.consumer.stats.pending_embeddings
                _LOGGER.warning(
                    "publication drain incomplete (pending_publications=%d)",
                    pending,
                )
                message = "publication drain incomplete; refusing model/pipeline restart"
                raise TransitionRecoveryError(message)
        if self._generation is generation:
            self._generation = None

    async def _run_generation(self, generation: _Generation) -> None:
        try:
            async with anyio.create_task_group() as task_group:
                pipeline_done = anyio.Event()
                publications_done = anyio.Event()
                status_stop_event = anyio.Event()
                task_group.start_soon(
                    _run_pipeline,
                    generation.pipeline,
                    generation.stop_event,
                    pipeline_done,
                )
                task_group.start_soon(generation.live_detection.run, generation.stop_event)
                task_group.start_soon(
                    _drain_publications,
                    generation.consumer,
                    generation.stop_event,
                    self._settings.worker_poll_seconds,
                    pipeline_done,
                    publications_done,
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
                    status_stop_event,
                    self._settings.worker_poll_seconds,
                )
                await generation.retention.run_forever(generation.stop_event)
                drained = await _wait_for_shutdown_drain(
                    pipeline_done,
                    publications_done,
                    generation.consumer,
                )
                if not drained:
                    await publications_done.wait()
                status_stop_event.set()
                task_group.cancel_scope.cancel()
        finally:
            try:
                with anyio.CancelScope(shield=True):
                    await generation.clip_transport.__aexit__(None, None, None)
            finally:
                generation.done_event.set()


@dataclass(frozen=True, slots=True)
class OneShotTransition:
    """Explicitly supplied production components without a polling task."""

    selection: ModelSelectionService
    lifecycle: _PipelineLifecycle
    runtime: ClipRuntimeManager
    crop_store: CropObjectStore
    triton_grpc_url: str

    async def run_pending(
        self,
        *,
        observer: TransitionObserver,
        restart_probe: Callable[[], Awaitable[None]] | None = None,
    ) -> TransitionResult | None:
        """Run one queued transition, optionally inspect restart, then join generation."""
        transport = TritonClipTransport(self.triton_grpc_url)
        try:
            result = await self.selection.run_pending(
                crop_store=self.crop_store,
                runtime=self.runtime,
                clip_factory=lambda package: ClipAdapter(transport, package=package),
                pipeline=self.lifecycle,
                observer=observer,
            )
            if result is not None and result.activated and restart_probe is not None:
                try:
                    await restart_probe()
                except BaseException:
                    observer.complete = False
                    raise
            return result
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    await self.lifecycle.stop_and_join()
                except BaseException:
                    observer.complete = False
                    raise
                finally:
                    try:
                        await transport.__aexit__(None, None, None)
                    except BaseException:
                        observer.complete = False
                        raise


def compose_one_shot_transition(
    *,
    settings: PipelineWorkerSettings,
    database: Database,
    storage: StorageRepository,
    crop_store: CropObjectStore,
    detector_transport: TritonGrpcDetectorTransport,
    detector: DetectorClient,
    registry: ClipModelRegistry,
    prepared: PreparedModelCatalogPort,
    coordinator: TransitionCoordinator,
    runtime: ClipRuntimeManager,
    task_group: TaskGroup,
    quality_policy_path: Path,
) -> OneShotTransition:
    """Compose the worker's transition from explicit caller-owned resources."""
    selection = ModelSelectionService(
        database=database,
        registry=registry,
        prepared=prepared,
        coordinator=coordinator,
        storage=storage,
        imported_assets_root=settings.model_assets_root / "imported",
        quality_policy_path=quality_policy_path,
        quality_evidence_root=settings.model_assets_root / "quality-evidence",
        preflight_assets_root=settings.model_assets_root,
        preflight_crop_store=crop_store,
    )
    lifecycle = _PipelineLifecycle(
        settings=settings,
        database=database,
        storage=storage,
        crop_store=crop_store,
        detector=detector,
        detector_transport=detector_transport,
        task_group=task_group,
    )
    return OneShotTransition(selection, lifecycle, runtime, crop_store, settings.triton_grpc_url)


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
    registry = load_clip_registry(settings.model_assets_root)
    from gods_watching.model_selection.assets import PreparedModelCatalog  # noqa: PLC0415

    prepared = PreparedModelCatalog(
        registry,
        settings.model_lock_path,
        assets_root=settings.model_assets_root,
    )
    coordinator = TransitionCoordinator(database)
    runtime = ClipRuntimeManager(settings.triton_grpc_url)
    try:
        async with coordinator.worker_ownership():
            async with runtime:
                async with anyio.create_task_group() as task_group:
                    composition = compose_one_shot_transition(
                        settings=settings,
                        database=database,
                        storage=storage,
                        crop_store=crop_store,
                        detector=detector,
                        detector_transport=detector_transport,
                        registry=registry,
                        prepared=prepared,
                        coordinator=coordinator,
                        runtime=runtime,
                        task_group=task_group,
                        quality_policy_path=Path(
                            "/opt/gods-watching/assets/retrieval-quality-policy.json"
                        ),
                    )
                    selection = composition.selection
                    lifecycle = composition.lifecycle
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
                            clip_factory=lambda package,
                            transport=transition_transport: ClipAdapter(
                                transport,
                                package=package,
                            ),
                            pipeline=lifecycle,
                        )
                    finally:
                        with anyio.CancelScope(shield=True):
                            await transition_transport.__aexit__(None, None, None)
                except TransitionRecoveryError as error:
                    if "durable identity unavailable" in str(
                        error
                    ) or "does not match transition endpoints" in str(error):
                        raise
                # Pace both idle polling and rolling_back retry.  A failed
                # restore remains visible in maintenance and must not hammer
                # Triton/DB.
                with anyio.move_on_after(settings.worker_poll_seconds):
                    await stop_event.wait()
        finally:
            done_event.set()


async def _run_pipeline(
    pipeline: PipelineWorker,
    stop_event: anyio.Event,
    done_event: anyio.Event,
) -> None:
    try:
        await pipeline.run(stop_event=stop_event)
    finally:
        done_event.set()


async def _drain_publications(
    consumer: AppearanceHandoffConsumer,
    stop_event: anyio.Event,
    interval_seconds: float,
    pipeline_done: anyio.Event | None = None,
    drainer_done: anyio.Event | None = None,
) -> None:
    pipeline_completion = pipeline_done or stop_event
    try:
        while True:
            if (
                stop_event.is_set()
                and pipeline_completion.is_set()
                and consumer.stats.pending_embeddings == 0
            ):
                return
            acknowledgement = await consumer.drain_one()
            if acknowledgement is None or acknowledgement.outcome is PublicationOutcome.PAUSED:
                if stop_event.is_set():
                    await anyio.sleep(interval_seconds)
                else:
                    with anyio.move_on_after(interval_seconds):
                        await stop_event.wait()
    finally:
        if drainer_done is not None:
            drainer_done.set()


async def _wait_for_shutdown_drain(
    pipeline_done: anyio.Event,
    drainer_done: anyio.Event,
    consumer: AppearanceHandoffConsumer,
) -> bool:
    with anyio.move_on_after(_PUBLICATION_SHUTDOWN_DRAIN_SECONDS) as scope:
        await pipeline_done.wait()
        await drainer_done.wait()
    if not scope.cancelled_caught:
        return True
    _LOGGER.warning(
        "shutdown drain deadline: pipeline_done=%s drainer_done=%s pending_publications=%d",
        pipeline_done.is_set(),
        drainer_done.is_set(),
        consumer.stats.pending_embeddings,
    )
    return False


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
    latest_accounting: StorageAccounting | None = None
    last_accounting_success: float | None = None
    accounting_failed = False

    async def refresh_accounting() -> None:
        nonlocal accounting_failed, last_accounting_success, latest_accounting
        while not stop_event.is_set():
            try:
                latest_accounting = await loop.retention.accounting()
                last_accounting_success = monotonic()
                accounting_failed = False
            except Exception:
                accounting_failed = True
                _LOGGER.exception("retention accounting refresh failed")
            with anyio.move_on_after(_ACCOUNTING_REFRESH_INTERVAL_SECONDS) as scope:
                await stop_event.wait()
            if not scope.cancelled_caught:
                return

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(refresh_accounting)
        try:
            while not stop_event.is_set():
                accounting = latest_accounting
                accounting_fresh = (
                    accounting is not None
                    and last_accounting_success is not None
                    and monotonic() - last_accounting_success <= _ACCOUNTING_STALE_AFTER_SECONDS
                )
                if not accounting_fresh:
                    accounting = None
                publication = loop.consumer.stats
                async with loop.database.transaction() as session:
                    await loop.reporter.report(
                        session,
                        snapshot=WorkerStatusSnapshot(
                            observed_at=datetime.now(UTC),
                            inference_ready=await loop.detector.ready(),
                            persistence_paused=(
                                accounting_failed
                                or accounting is None
                                or accounting.cleanup_required
                                or accounting.filesystem_guard_active
                            ),
                            storage_managed_bytes=(
                                0 if accounting is None else accounting.managed_bytes
                            ),
                            storage_quota_bytes=0 if accounting is None else accounting.quota_bytes,
                            indexing_queue_depth=publication.pending_embeddings,
                            last_searchable_latency_seconds=(
                                publication.last_searchable_latency_seconds
                            ),
                        ),
                    )
                with anyio.move_on_after(interval_seconds):
                    await stop_event.wait()
        finally:
            task_group.cancel_scope.cancel()


__all__ = ["run_pipeline_worker"]
