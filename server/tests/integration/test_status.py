from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from pydantic import AnyUrl
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.status import read_status
from gods_watching.storage import (
    Camera,
    CameraRuntimeStatus,
    CredentialCipher,
    StorageRepository,
    WorkerRuntimeStatus,
)


async def _camera(session: AsyncSession, *, enabled: bool = True) -> Camera:
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    return await repository.add_camera(
        session,
        name=f"status-{uuid4()}",
        source_url=AnyUrl("rtsp://camera.local/live"),
        detection_enabled=enabled,
    )


@pytest.mark.anyio
async def test_status_is_ready_when_worker_and_frame_are_fresh(session: AsyncSession) -> None:
    # Given: a recent worker heartbeat and a camera ingesting a recent frame
    now = datetime(2026, 9, 20, 12, tzinfo=UTC)
    camera = await _camera(session)
    session.add_all(
        [
            WorkerRuntimeStatus(
                singleton=True,
                updated_at=now,
                inference_ready=True,
                persistence_paused=False,
                storage_managed_bytes=512,
                storage_quota_bytes=1024,
                indexing_queue_depth=2,
                last_searchable_latency_seconds=1.25,
            ),
            CameraRuntimeStatus(
                camera_id=camera.id,
                camera_session_id=None,
                updated_at=now,
                last_ingest_at=now - timedelta(seconds=1),
                actual_framerate=25.0,
                detector_framerate=4.9,
                dropped_frames=3,
                detector_requests=8,
                detector_results=8,
                last_error=None,
            ),
        ]
    )
    await session.flush()

    # When: the API composes its public status response
    response = await read_status(session, observed_at=now)

    # Then: it reports the observed metrics without degrading the service
    assert response.state == "ready"
    assert response.inference_ready is True
    assert response.storage_managed_bytes == 512
    assert response.indexing_queue_depth == 2
    assert response.last_searchable_latency_seconds == 1.25
    assert response.cameras[0].state == "online"
    assert response.cameras[0].actual_framerate == 25.0
    assert response.cameras[0].detector_framerate == 4.9
    assert response.cameras[0].frame_age_seconds == 1.0
    assert response.cameras[0].dropped_frames == 3


@pytest.mark.anyio
async def test_status_degrades_when_worker_heartbeat_expires(session: AsyncSession) -> None:
    # Given: a configured camera whose worker heartbeat is older than the liveness bound
    now = datetime(2026, 9, 20, 12, tzinfo=UTC)
    camera = await _camera(session)
    session.add(
        WorkerRuntimeStatus(
            singleton=True,
            updated_at=now - timedelta(seconds=6),
            inference_ready=True,
            persistence_paused=False,
            storage_managed_bytes=0,
            storage_quota_bytes=1024,
            indexing_queue_depth=0,
            last_searchable_latency_seconds=None,
        )
    )
    await session.flush()

    # When: status is read after the heartbeat deadline
    response = await read_status(session, observed_at=now)

    # Then: the process and camera are not presented as healthy
    assert response.state == "degraded"
    assert response.inference_ready is False
    assert response.cameras[0].camera_id == camera.id
    assert response.cameras[0].state == "offline"
