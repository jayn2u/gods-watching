from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.storage import (
    Appearance,
    CredentialCipher,
    CropGarbage,
    CropObjectStore,
    StaleAppearanceVersionError,
    StorageRepository,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import AnyUrl
    from sqlalchemy.ext.asyncio import AsyncSession

_OBSERVED_AT: Final = datetime(2026, 9, 6, 12, 30, tzinfo=UTC)


def _rtsp(value: str) -> AnyUrl:
    return CameraCreateRequest.model_validate(
        {"name": "fixture", "source_url": value}
    ).source_url


def _publication(
    *,
    appearance_id: UUID,
    camera_id: UUID,
    session_id: UUID,
    version: int,
    object_key: str,
) -> AppearancePublication:
    return AppearancePublication(
        appearance_id=AppearanceId(appearance_id),
        camera_id=CameraId(camera_id),
        session_id=CameraSessionId(session_id),
        track_id=7,
        first_seen=_OBSERVED_AT,
        last_seen=_OBSERVED_AT,
        ended_at=None,
        representative_version=version,
        crop_object_key=object_key,
        bounding_box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.91,
        crop_quality=42.5,
        byte_size=13,
        embedded_at=_OBSERVED_AT,
        model_id="openai/clip-vit-base-patch16",
        model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
        embedding=tuple(1.0 if index == 0 else 0.0 for index in range(512)),
    )


@pytest.mark.anyio
async def test_persist_camera_encrypts_credentials_and_enforces_unique_name(
    session: AsyncSession,
) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    source = _rtsp("rtsp://operator:camera-secret@fixture:8554/lobby")

    # When
    camera = await repository.add_camera(session, name="Lobby", source_url=source)
    await session.flush()

    # Then
    assert camera.source_ciphertext != str(source).encode()
    assert b"camera-secret" not in camera.source_ciphertext
    assert await repository.get_camera_source(session, camera.id) == str(source)
    with pytest.raises(IntegrityError):
        _ = await repository.add_camera(session, name="Lobby", source_url=source)


@pytest.mark.anyio
async def test_persist_appearance_round_trips_utc_vector_and_generated_crop(
    session: AsyncSession,
    tmp_path: Path,
) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    store = CropObjectStore(tmp_path / "crops")
    camera = await repository.add_camera(
        session,
        name="Entrance",
        source_url=_rtsp("rtsp://fixture:8554/entrance"),
    )
    camera_session = await repository.start_camera_session(session, camera.id, cause="reconnect")
    crop = store.write(b"jpeg-fixture")
    publication = _publication(
        appearance_id=uuid4(),
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key=crop.object_key,
    )

    # When
    _ = await repository.publish_appearance(session, publication)
    await session.flush()
    persisted = await session.scalar(
        select(Appearance).where(Appearance.id == publication.appearance_id)
    )

    # Then
    assert persisted is not None
    assert persisted.first_seen == _OBSERVED_AT
    assert persisted.embedding is not None
    assert persisted.embedding.startswith("[1.0,0.0,")
    assert store.read(crop.object_key) == b"jpeg-fixture"


@pytest.mark.anyio
async def test_rollback_does_not_persist_partial_appearance(session: AsyncSession) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera = await repository.add_camera(
        session,
        name="Rollback camera",
        source_url=_rtsp("rtsp://fixture:8554/rollback"),
    )
    camera_session = await repository.start_camera_session(session, camera.id, cause="source_edit")
    appearance_id = uuid4()
    publication = _publication(
        appearance_id=appearance_id,
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key="aa/bb/00000000-0000-4000-8000-000000000001.jpg",
    )
    savepoint = await session.begin_nested()

    # When
    _ = await repository.publish_appearance(session, publication)
    await savepoint.rollback()

    # Then
    assert await session.scalar(select(func.count()).select_from(Appearance)) == 0


@pytest.mark.anyio
async def test_stale_version_cannot_overwrite_committed_revision(session: AsyncSession) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera = await repository.add_camera(
        session,
        name="CAS camera",
        source_url=_rtsp("rtsp://fixture:8554/cas"),
    )
    camera_session = await repository.start_camera_session(session, camera.id, cause="enabled")
    appearance_id = uuid4()
    initial = _publication(
        appearance_id=appearance_id,
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key="aa/bb/00000000-0000-4000-8000-000000000002.jpg",
    )
    _ = await repository.publish_appearance(session, initial)
    await session.flush()

    # When / Then
    with pytest.raises(StaleAppearanceVersionError):
        _ = await repository.publish_appearance(session, initial)
    persisted = await session.get(Appearance, appearance_id)
    assert persisted is not None
    assert persisted.representative_version == 1


def test_traversal_is_rejected_without_writing_outside_root(tmp_path: Path) -> None:
    # Given
    store = CropObjectStore(tmp_path / "crops")
    outside = tmp_path / "outside.jpg"

    # When / Then
    with pytest.raises(ValueError, match="crop object key"):
        _ = store.read("../outside.jpg")
    assert not outside.exists()


def test_symlink_swap_cannot_redirect_crop_write_outside_root(tmp_path: Path) -> None:
    # Given
    root = tmp_path / "crops"
    outside = tmp_path / "outside"
    store = CropObjectStore(root)
    stable_id = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
    expected_parent = root / "aa" / "aa"
    expected_parent.mkdir(parents=True)
    (outside / "aa").mkdir(parents=True)
    original_open = os.open
    swapped = False

    def racing_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        is_path_write = str(path).endswith(".tmp")
        is_directory_walk = path == "aa" and dir_fd is not None
        if not swapped and (is_path_write or is_directory_walk):
            _ = (root / "aa").rename(root / "displaced")
            _ = (root / "aa").symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    # When / Then
    with (
        patch("gods_watching.storage.crops.uuid4", return_value=stable_id),
        patch("gods_watching.storage.crops.os.open", side_effect=racing_open),
        pytest.raises(ValueError, match="crop object key"),
    ):
        _ = store.write(b"must-not-escape")
    assert swapped
    assert not any(outside.rglob("*.jpg"))


@pytest.mark.anyio
async def test_tombstone_removes_vector_and_enqueues_crop_gc(session: AsyncSession) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera = await repository.add_camera(
        session,
        name="Delete camera",
        source_url=_rtsp("rtsp://fixture:8554/delete"),
    )
    camera_session = await repository.start_camera_session(session, camera.id, cause="loop")
    appearance_id = uuid4()
    publication = _publication(
        appearance_id=appearance_id,
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key="aa/bb/00000000-0000-4000-8000-000000000003.jpg",
    )
    _ = await repository.publish_appearance(session, publication)

    # When
    await repository.tombstone_appearance(session, appearance_id)
    await session.flush()

    # Then
    appearance = await session.get(Appearance, appearance_id)
    assert appearance is not None
    assert appearance.tombstoned_at is not None
    assert appearance.embedding is None
    assert await session.scalar(select(func.count()).select_from(CropGarbage)) == 1


@pytest.mark.anyio
async def test_persist_reports_positive_physical_relation_size(session: AsyncSession) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))

    # When
    sizes = await repository.application_relation_sizes(session)

    # Then
    assert "appearances" in sizes
    assert sizes["appearances"] > 0


@pytest.mark.anyio
async def test_schema_rejects_duplicate_track_identity(session: AsyncSession) -> None:
    # Given
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    camera = await repository.add_camera(
        session,
        name="Unique camera",
        source_url=_rtsp("rtsp://fixture:8554/unique"),
    )
    camera_session = await repository.start_camera_session(session, camera.id, cause="loop")
    first = _publication(
        appearance_id=uuid4(),
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key="aa/bb/00000000-0000-4000-8000-000000000004.jpg",
    )
    duplicate = _publication(
        appearance_id=uuid4(),
        camera_id=camera.id,
        session_id=camera_session.id,
        version=1,
        object_key="aa/bb/00000000-0000-4000-8000-000000000005.jpg",
    )
    _ = await repository.publish_appearance(session, first)
    await session.flush()

    # When / Then
    with pytest.raises(IntegrityError):
        _ = await repository.publish_appearance(session, duplicate)
