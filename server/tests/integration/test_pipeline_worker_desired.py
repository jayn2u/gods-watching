from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import delete

from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.contracts.cameras import CameraCreateRequest, CameraPatchRequest
from gods_watching.contracts.identifiers import CameraId
from gods_watching.pipeline_worker.desired import load_desired_cameras
from gods_watching.storage import (
    Camera,
    CameraSession,
    CredentialCipher,
    Database,
    StorageRepository,
)

if TYPE_CHECKING:
    from gods_watching.cameras import CameraMutation


def _create_request(label: str, *, detection_enabled: bool = True) -> CameraCreateRequest:
    return CameraCreateRequest.model_validate(
        {
            "name": f"desired {label} {uuid4().hex[:8]}",
            "source_url": f"rtsp://user:secret@fixture:8554/{label}",
            "detection_enabled": detection_enabled,
        }
    )


def _session_id(mutation: CameraMutation) -> UUID:
    activation = mutation.lifecycle.activation
    assert activation is not None
    return activation.session_id


@pytest.mark.anyio
async def test_desired_cameras_follow_committed_detection_sessions(database_url: str) -> None:
    # Given: cameras created, disabled, deleted, re-sourced, and re-thresholded by the API
    database = Database.connect(database_url)
    service = CameraService(
        CameraRepository(StorageRepository(CredentialCipher(Fernet.generate_key())))
    )
    created: list[UUID] = []
    try:
        async with database.transaction() as session:
            enabled = await service.create(session, _create_request("enabled"))
            disabled = await service.create(
                session, _create_request("disabled", detection_enabled=False)
            )
            deleted = await service.create(session, _create_request("deleted"))
            edited = await service.create(session, _create_request("edited"))
            tuned = await service.create(session, _create_request("tuned"))
        created.extend(
            UUID(str(mutation.camera.camera_id))
            for mutation in (enabled, disabled, deleted, edited, tuned)
        )
        async with database.transaction() as session:
            _ = await service.delete(
                session, deleted.camera.camera_id, expected_version=deleted.camera.version
            )
            re_sourced = await service.update(
                session,
                edited.camera.camera_id,
                CameraPatchRequest.model_validate(
                    {"source_url": "rtsp://user:secret@fixture:8554/edited-v2"}
                ),
                expected_version=edited.camera.version,
            )
            _ = await service.update(
                session,
                tuned.camera.camera_id,
                CameraPatchRequest.model_validate({"detection_threshold": 0.7}),
                expected_version=tuned.camera.version,
            )

        # When: the pipeline worker reads the committed desired state
        async with database.transaction() as session:
            loaded = await load_desired_cameras(session, service.repository)
        desired = {
            camera_id: camera for camera_id, camera in loaded.items() if camera_id in created
        }

        # Then: only detection-enabled, undeleted cameras with an active session remain
        assert set(desired) == {
            CameraId(enabled.camera.camera_id),
            CameraId(edited.camera.camera_id),
            CameraId(tuned.camera.camera_id),
        }
        enabled_camera = desired[CameraId(enabled.camera.camera_id)]
        assert enabled_camera.session_id == _session_id(enabled)
        assert enabled_camera.source_url == "rtsp://user:secret@fixture:8554/enabled"
        edited_camera = desired[CameraId(edited.camera.camera_id)]
        assert edited_camera.session_id == _session_id(re_sourced)
        assert edited_camera.session_id != _session_id(edited)
        assert edited_camera.source_url == "rtsp://user:secret@fixture:8554/edited-v2"
        tuned_camera = desired[CameraId(tuned.camera.camera_id)]
        assert tuned_camera.session_id == _session_id(tuned)
        assert tuned_camera.threshold == 0.7
        assert tuned_camera.version == tuned.camera.version + 1
    finally:
        if created:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id.in_(created))
                )
                _ = await session.execute(delete(Camera).where(Camera.id.in_(created)))
        await database.close()


@pytest.mark.anyio
async def test_undecryptable_camera_is_skipped_without_blocking_others(database_url: str) -> None:
    # Given: one camera encrypted under a different key next to a readable camera
    database = Database.connect(database_url)
    service = CameraService(
        CameraRepository(StorageRepository(CredentialCipher(Fernet.generate_key())))
    )
    foreign = CameraService(
        CameraRepository(StorageRepository(CredentialCipher(Fernet.generate_key())))
    )
    created: list[UUID] = []
    try:
        async with database.transaction() as session:
            readable = await service.create(session, _create_request("readable"))
            unreadable = await foreign.create(session, _create_request("foreign-key"))
        created.extend(UUID(str(mutation.camera.camera_id)) for mutation in (readable, unreadable))

        # When: the worker loads desired cameras with its own key
        async with database.transaction() as session:
            loaded = await load_desired_cameras(session, service.repository)

        # Then: the readable camera still runs and the undecryptable one is left out
        assert CameraId(readable.camera.camera_id) in loaded
        assert CameraId(unreadable.camera.camera_id) not in loaded
    finally:
        if created:
            async with database.transaction() as session:
                _ = await session.execute(
                    delete(CameraSession).where(CameraSession.camera_id.in_(created))
                )
                _ = await session.execute(delete(Camera).where(Camera.id.in_(created)))
        await database.close()
