"""Real PostgreSQL coverage for latest live detector snapshots."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.cameras import CameraActivationRequest, CameraRepository, CameraService
from gods_watching.contracts.cameras import CameraCreateRequest, CameraPatchRequest
from gods_watching.live_detections import (
    LiveDetectionBox,
    LiveDetectionRepository,
    LiveDetectionSnapshot,
)
from gods_watching.media.models import SourceGenerationId
from gods_watching.storage import CredentialCipher, StorageRepository


def _service() -> CameraService:
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    return CameraService(CameraRepository(storage))


def _snapshot(
    activation: CameraActivationRequest,
    *,
    frame_at: datetime,
    boxes: tuple[LiveDetectionBox, ...],
) -> LiveDetectionSnapshot:
    return LiveDetectionSnapshot(
        camera_id=activation.camera_id,
        camera_session_id=activation.session_id,
        db_generation_id=activation.generation_id,
        source_generation_id=SourceGenerationId(uuid4()),
        frame_at=frame_at,
        width=640,
        height=480,
        boxes=boxes,
    )


@pytest.mark.anyio
async def test_latest_repository_reads_current_boxes_expires_them_and_fences_reconnect(
    session: AsyncSession,
) -> None:
    service = _service()
    created = await service.create(
        session,
        CameraCreateRequest.model_validate(
            {"name": f"overlay-{uuid4()}", "source_url": "rtsp://fixture:8554/live"}
        ),
    )
    first = created.lifecycle.activation
    assert first is not None
    frame_at = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = _snapshot(
        first,
        frame_at=frame_at,
        boxes=(LiveDetectionBox(10, 20, 110, 220, 0.91),),
    )
    repository = LiveDetectionRepository()
    assert await repository.publish(session, snapshot) is True

    current = await repository.read(session, first.camera_id, now=frame_at + timedelta(seconds=0.5))
    assert current.camera_session_id == first.session_id
    assert current.frame_age_seconds == 0.5
    assert len(current.boxes) == 1
    stale = await repository.read(session, first.camera_id, now=frame_at + timedelta(seconds=1.1))
    assert stale.boxes == ()
    assert stale.frame_age_seconds == 1.1

    disabled = await service.update(
        session,
        first.camera_id,
        CameraPatchRequest(detection_enabled=False),
        expected_version=created.camera.version,
    )
    enabled = await service.update(
        session,
        first.camera_id,
        CameraPatchRequest(detection_enabled=True),
        expected_version=disabled.camera.version,
    )
    replacement = enabled.lifecycle.activation
    assert replacement is not None
    assert replacement.session_id != first.session_id
    assert await repository.publish(session, snapshot) is False

    fenced = await repository.read(session, first.camera_id, now=frame_at + timedelta(seconds=0.1))
    assert fenced.camera_session_id == replacement.session_id
    assert fenced.frame_at is None
    assert fenced.boxes == ()

    empty_snapshot = _snapshot(replacement, frame_at=frame_at, boxes=())
    assert await repository.publish(session, empty_snapshot) is True
    empty = await repository.read(session, first.camera_id, now=frame_at + timedelta(seconds=0.1))
    assert empty.boxes == ()
