"""Run ingest workers for committed camera sessions in a process separate from the API."""

from collections.abc import Callable
from typing import Final
from uuid import uuid4

import anyio

from gods_watching.cameras import CameraService
from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.ingest import IngestCoordinator, IngestWorker, IngestWorkerConfiguration
from gods_watching.ingest.worker import DetectorPort, PipelineHandoffConsumer
from gods_watching.media.models import SourceGenerationId
from gods_watching.storage import Database

from .desired import load_desired_cameras
from .reconcile import DesiredCamera, ReconcilePlan, RunningCamera, plan_reconcile

WorkerFactory = Callable[
    [IngestWorkerConfiguration, DetectorPort, PipelineHandoffConsumer], IngestWorker
]
_DEFAULT_POLL_SECONDS: Final = 1.0


class PipelineWorker:
    """Follow the API's committed camera sessions and keep one ingest worker per session."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        database: Database,
        cameras: CameraService,
        coordinator: IngestCoordinator,
        consumer: PipelineHandoffConsumer,
        detector: DetectorPort,
        worker_factory: WorkerFactory = IngestWorker,
        poll_seconds: float = _DEFAULT_POLL_SECONDS,
        detector_deadline_seconds: float = 2.0,
    ) -> None:
        """Bind the database, ingest coordinator, and shared publication consumer."""
        self._database: Database = database
        self._cameras: CameraService = cameras
        self._coordinator: IngestCoordinator = coordinator
        self._consumer: PipelineHandoffConsumer = consumer
        self._detector: DetectorPort = detector
        self._worker_factory: WorkerFactory = worker_factory
        self._poll_seconds: float = poll_seconds
        self._detector_deadline_seconds: float = detector_deadline_seconds

    async def active_binding(self, camera_id: CameraId) -> GenerationBinding | None:
        """Return the generation the live worker owns, for publication fencing."""
        for worker in self._coordinator.workers:
            if worker.camera_id == camera_id:
                return worker.generation
        return None

    async def reconcile_once(self) -> ReconcilePlan:
        """Stop, replace, and start workers so they match committed camera sessions."""
        async with self._database.transaction() as session:
            desired = await load_desired_cameras(session, self._cameras.repository)
        failed = set(self._coordinator.failed_cameras)
        running = {
            worker.camera_id: _running(worker)
            for worker in self._coordinator.workers
            if worker.camera_id not in failed
        }
        for camera_id in failed:
            await self._stop(camera_id)
        plan = plan_reconcile(desired, running)
        for camera_id in plan.stop:
            await self._stop(camera_id)
        for camera in plan.start:
            self._coordinator.add(self._new_worker(camera))
        return plan

    async def run(self, *, stop_event: anyio.Event) -> None:
        """Run decode/detection and poll committed sessions until stopped."""
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(self._run_coordinator, stop_event)
            while not stop_event.is_set():
                _ = await self.reconcile_once()
                with anyio.move_on_after(self._poll_seconds):
                    await stop_event.wait()

    async def _run_coordinator(self, stop_event: anyio.Event) -> None:
        await self._coordinator.run(stop_event=stop_event)

    async def _stop(self, camera_id: CameraId) -> None:
        for handoff in await self._coordinator.remove(camera_id):
            await self._consumer(handoff)

    def _new_worker(self, camera: DesiredCamera) -> IngestWorker:
        binding = GenerationBinding(
            camera_id=camera.camera_id,
            camera_session_id=camera.session_id,
            db_generation_id=camera.generation_id,
            source_generation_id=SourceGenerationId(uuid4()),
            camera_version=camera.version,
        )
        return self._worker_factory(
            IngestWorkerConfiguration(
                generation=binding,
                source_url=camera.source_url,
                threshold=camera.threshold,
                detector_deadline_seconds=self._detector_deadline_seconds,
                generation_reconnect=self._reconnect,
            ),
            self._detector,
            self._consumer,
        )

    async def _reconnect(self, previous: GenerationBinding) -> GenerationBinding:
        # A mismatch (the API already replaced the session) raises out of the decoder;
        # the coordinator records the failure and the next pass rebinds the camera.
        async with self._database.transaction() as session:
            request = await self._cameras.reconnect(session, previous)
        return GenerationBinding(
            camera_id=request.camera_id,
            camera_session_id=request.session_id,
            db_generation_id=request.generation_id,
            source_generation_id=SourceGenerationId(uuid4()),
            camera_version=request.version,
        )


def _running(worker: IngestWorker) -> RunningCamera:
    generation = worker.generation
    return RunningCamera(
        camera_id=generation.camera_id,
        version=generation.camera_version,
        session_id=generation.camera_session_id,
        generation_id=generation.db_generation_id,
        threshold=worker.threshold,
    )


__all__ = ["PipelineWorker", "WorkerFactory"]
