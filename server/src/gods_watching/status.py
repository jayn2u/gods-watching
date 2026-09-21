"""Persist worker telemetry and compose authenticated operational status."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, Literal, final

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.status import CameraStatus, StatusResponse
from gods_watching.storage import Camera, CameraRuntimeStatus, WorkerRuntimeStatus

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.ingest import IngestCoordinator

_STALE_AFTER: Final = timedelta(seconds=2)
_OFFLINE_AFTER: Final = timedelta(seconds=10)
_WORKER_OFFLINE_AFTER: Final = timedelta(seconds=5)


@final
class StatusReporter:
    """Write one replaceable runtime snapshot from the worker process."""

    def __init__(self, coordinator: IngestCoordinator) -> None:
        """Bind the reporter to the live worker coordinator."""
        self._coordinator: IngestCoordinator = coordinator

    async def report(
        self,
        session: AsyncSession,
        *,
        snapshot: WorkerStatusSnapshot,
    ) -> None:
        """Upsert live workers and remove snapshots for workers no longer active."""
        active_ids: set[CameraId] = set()
        for worker in self._coordinator.workers:
            active_ids.add(worker.camera_id)
            stats = worker.stats
            last_ingest_at = (
                None
                if stats.frame_age_seconds is None
                else snapshot.observed_at - timedelta(seconds=stats.frame_age_seconds)
            )
            statement = insert(CameraRuntimeStatus).values(
                camera_id=worker.camera_id,
                camera_session_id=worker.generation.camera_session_id,
                updated_at=snapshot.observed_at,
                last_ingest_at=last_ingest_at,
                actual_framerate=stats.actual_framerate,
                detector_framerate=stats.detector_framerate,
                dropped_frames=stats.dropped_frames,
                detector_requests=stats.detector_requests,
                detector_results=stats.detector_results,
                last_error=stats.last_sanitized_error,
            )
            _ = await session.execute(
                statement.on_conflict_do_update(
                    index_elements=[CameraRuntimeStatus.camera_id],
                    set_={
                        "updated_at": statement.excluded.updated_at,
                        "camera_session_id": statement.excluded.camera_session_id,
                        "last_ingest_at": statement.excluded.last_ingest_at,
                        "actual_framerate": statement.excluded.actual_framerate,
                        "detector_framerate": statement.excluded.detector_framerate,
                        "dropped_frames": statement.excluded.dropped_frames,
                        "detector_requests": statement.excluded.detector_requests,
                        "detector_results": statement.excluded.detector_results,
                        "last_error": statement.excluded.last_error,
                    },
                )
            )
        cleanup = delete(CameraRuntimeStatus)
        if active_ids:
            cleanup = cleanup.where(CameraRuntimeStatus.camera_id.not_in(active_ids))
        _ = await session.execute(cleanup)
        worker_statement = insert(WorkerRuntimeStatus).values(
            singleton=True,
            updated_at=snapshot.observed_at,
            inference_ready=snapshot.inference_ready,
            persistence_paused=snapshot.persistence_paused,
            storage_managed_bytes=snapshot.storage_managed_bytes,
            storage_quota_bytes=snapshot.storage_quota_bytes,
            indexing_queue_depth=snapshot.indexing_queue_depth,
            last_searchable_latency_seconds=snapshot.last_searchable_latency_seconds,
        )
        _ = await session.execute(
            worker_statement.on_conflict_do_update(
                index_elements=[WorkerRuntimeStatus.singleton],
                set_={
                    "updated_at": worker_statement.excluded.updated_at,
                    "inference_ready": worker_statement.excluded.inference_ready,
                    "persistence_paused": worker_statement.excluded.persistence_paused,
                    "storage_managed_bytes": worker_statement.excluded.storage_managed_bytes,
                    "storage_quota_bytes": worker_statement.excluded.storage_quota_bytes,
                    "indexing_queue_depth": worker_statement.excluded.indexing_queue_depth,
                    "last_searchable_latency_seconds": (
                        worker_statement.excluded.last_searchable_latency_seconds
                    ),
                },
            )
        )


@dataclass(frozen=True, slots=True)
class WorkerStatusSnapshot:
    """Carry one process-wide status observation across the persistence boundary."""

    observed_at: datetime
    inference_ready: bool
    persistence_paused: bool
    storage_managed_bytes: int
    storage_quota_bytes: int
    indexing_queue_depth: int
    last_searchable_latency_seconds: float | None


async def read_status(
    session: AsyncSession, *, observed_at: datetime | None = None
) -> StatusResponse:
    """Compose camera and process readiness without treating missing telemetry as healthy."""
    now = observed_at or datetime.now(UTC)
    worker = await session.get(WorkerRuntimeStatus, True)
    rows = (
        await session.execute(
            select(Camera, CameraRuntimeStatus)
            .outerjoin(CameraRuntimeStatus, CameraRuntimeStatus.camera_id == Camera.id)
            .where(Camera.deleted_at.is_(None))
            .order_by(Camera.name.asc(), Camera.id.asc())
        )
    ).tuples()
    worker_fresh = worker is not None and now - worker.updated_at <= _WORKER_OFFLINE_AFTER
    cameras = tuple(
        _camera_status(camera, runtime, now=now, worker_fresh=worker_fresh)
        for camera, runtime in rows
    )
    if worker is None:
        state: Literal["ready", "degraded", "starting"] = "starting"
    elif (
        not worker_fresh
        or not worker.inference_ready
        or worker.persistence_paused
        or any(camera.state in {"stale", "offline"} for camera in cameras)
    ):
        state = "degraded"
    else:
        state = "ready"
    return StatusResponse(
        state=state,
        cameras=cameras,
        inference_ready=worker_fresh and worker is not None and worker.inference_ready,
        persistence_paused=worker is not None and worker.persistence_paused,
        worker_updated_at=None if worker is None else worker.updated_at,
        storage_managed_bytes=0 if worker is None else worker.storage_managed_bytes,
        storage_quota_bytes=0 if worker is None else worker.storage_quota_bytes,
        indexing_queue_depth=0 if worker is None else worker.indexing_queue_depth,
        last_searchable_latency_seconds=(
            None if worker is None else worker.last_searchable_latency_seconds
        ),
    )


def _camera_status(
    camera: Camera,
    runtime: CameraRuntimeStatus | None,
    *,
    now: datetime,
    worker_fresh: bool,
) -> CameraStatus:
    if not camera.detection_enabled:
        state: Literal["online", "stale", "offline", "disabled"] = "disabled"
    elif (
        not worker_fresh
        or runtime is None
        or runtime.last_ingest_at is None
        or now - runtime.last_ingest_at > _OFFLINE_AFTER
    ):
        state = "offline"
    elif now - runtime.last_ingest_at > _STALE_AFTER:
        state = "stale"
    else:
        state = "online"
    return CameraStatus(
        camera_id=CameraId(camera.id),
        camera_session_id=(
            None
            if runtime is None or runtime.camera_session_id is None
            else CameraSessionId(runtime.camera_session_id)
        ),
        state=state,
        last_ingest_at=None if runtime is None else runtime.last_ingest_at,
        frame_age_seconds=(
            None
            if runtime is None or runtime.last_ingest_at is None
            else max(0.0, (now - runtime.last_ingest_at).total_seconds())
        ),
        actual_framerate=0.0 if runtime is None else runtime.actual_framerate,
        detector_framerate=0.0 if runtime is None else runtime.detector_framerate,
        dropped_frames=0 if runtime is None else runtime.dropped_frames,
        detector_requests=0 if runtime is None else runtime.detector_requests,
        detector_results=0 if runtime is None else runtime.detector_results,
        last_error=None if runtime is None else runtime.last_error,
    )


__all__ = ["StatusReporter", "WorkerStatusSnapshot", "read_status"]
