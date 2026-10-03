"""Transactional storage contracts for camera-management events."""

from typing import cast
from uuid import UUID, uuid4

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import CheckConstraint, DateTime, String, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import TextClause

from gods_watching.cameras import (
    CameraRepository,
    CameraService,
    ParsedRtspSource,
    ProbeFailureCode,
    ProbeResult,
    RtspProbeError,
    StaleCameraVersionError,
)
from gods_watching.cameras.service import SourceProbePort
from gods_watching.contracts.cameras import CameraCreateRequest, CameraPatchRequest
from gods_watching.storage import Base, Camera, CameraEvent, CredentialCipher, StorageRepository


def test_event_schema_contract() -> None:
    """Map only the approved identity, timestamp, type, camera and name fields."""
    metadata = Base.metadata

    # Given: application mappings before migrations are applied
    assert "camera_events" in metadata.tables
    table = metadata.tables["camera_events"]

    # Then: the event table contains exactly the durable, non-secret projection
    assert set(table.columns.keys()) == {
        "id",
        "occurred_at",
        "event_type",
        "camera_id",
        "camera_name",
    }
    assert table.c.id.primary_key
    assert table.c.id.type.python_type is int
    assert table.c.id.identity is not None
    assert isinstance(table.c.occurred_at.type, DateTime)
    assert table.c.occurred_at.type.timezone is True
    assert table.c.occurred_at.server_default is not None
    assert table.c.event_type.nullable is False
    assert table.c.camera_id.nullable is False
    assert table.c.camera_id.type.python_type.__name__ == "UUID"
    assert isinstance(table.c.camera_name.type, String)
    assert table.c.camera_name.type.length == 80
    assert table.c.camera_name.nullable is False
    assert not table.foreign_keys

    normalized: set[str] = set()
    for constraint in table.constraints:
        if isinstance(constraint, CheckConstraint):
            sqltext: object = cast("object", constraint.sqltext)
            assert isinstance(sqltext, TextClause)
            normalized.add(" ".join(str(sqltext).lower().split()))
    assert any(
        "camera.created" in sql and "camera.updated" in sql and "camera.deleted" in sql
        for sql in normalized
    )
    assert any("char_length(camera_name) between 1 and 80" in sql for sql in normalized)


def _service(*, source_probe: SourceProbePort | None = None) -> CameraService:
    """Build the real camera service with an isolated credential key."""
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    return CameraService(CameraRepository(storage), source_probe=source_probe)


def _create_request(name: str) -> CameraCreateRequest:
    """Build a valid synthetic camera request with recognizable secret material."""
    return CameraCreateRequest.model_validate(
        {"name": name, "source_url": "rtsp://operator:synthetic-secret@fixture:8554/live"}
    )


async def _events(session: AsyncSession, camera_id: UUID) -> tuple[CameraEvent, ...]:
    """Read retained events for one camera in event-ID order."""
    rows = await session.scalars(
        select(CameraEvent).where(CameraEvent.camera_id == camera_id).order_by(CameraEvent.id)
    )
    return tuple(rows.all())


@pytest.mark.anyio
async def test_create_update_delete_record_ordered_events(engine: AsyncEngine) -> None:
    """Capture one safe name snapshot for each committed camera mutation."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = _service()
    camera_name = f"events-{uuid4()}"
    changed_name = f"{camera_name}-renamed"

    async with factory.begin() as session:
        created = await service.create(session, _create_request(camera_name))
    async with factory.begin() as session:
        changed = await service.update(
            session,
            created.camera.camera_id,
            CameraPatchRequest(name=changed_name),
            expected_version=created.camera.version,
        )
    async with factory.begin() as session:
        deleted = await service.delete(
            session,
            created.camera.camera_id,
            expected_version=changed.camera.version,
        )

    async with factory.begin() as session:
        events = await _events(session, UUID(str(created.camera.camera_id)))
        archived_camera = await session.scalar(
            select(Camera).where(Camera.id == UUID(str(created.camera.camera_id)))
        )

    assert [event.event_type for event in events] == [
        "camera.created",
        "camera.updated",
        "camera.deleted",
    ]
    assert [event.camera_name for event in events] == [camera_name, changed_name, changed_name]
    assert all(event.camera_id == UUID(str(created.camera.camera_id)) for event in events)
    assert [event.id for event in events] == sorted(event.id for event in events)
    assert all(event.occurred_at.tzinfo is not None for event in events)
    assert "synthetic-secret" not in repr(events)
    assert deleted.camera.deleted_at is not None
    assert archived_camera is not None
    assert archived_camera.deleted_at is not None


@pytest.mark.anyio
async def test_noop_and_stale_update_do_not_record(engine: AsyncEngine) -> None:
    """Record only updates that pass the version fence and change persisted state."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = _service()
    name = f"no-op-{uuid4()}"
    async with factory.begin() as session:
        created = await service.create(session, _create_request(name))

    async with factory.begin() as session:
        _ = await service.update(
            session,
            created.camera.camera_id,
            CameraPatchRequest(name=name),
            expected_version=created.camera.version,
        )
    with pytest.raises(StaleCameraVersionError):
        async with factory.begin() as session:
            _ = await service.update(
                session,
                created.camera.camera_id,
                CameraPatchRequest(name=f"{name}-stale"),
                expected_version=created.camera.version - 1,
            )

    async with factory.begin() as session:
        events = await _events(session, UUID(str(created.camera.camera_id)))
    assert [event.event_type for event in events] == ["camera.created"]


@pytest.mark.anyio
async def test_failed_probe_does_not_record(engine: AsyncEngine) -> None:
    """A source probe failure leaves the durable event history unchanged."""

    class _FailedProbe:
        async def probe(self, source: ParsedRtspSource) -> ProbeResult:
            del source
            raise RtspProbeError(
                code=ProbeFailureCode.UNREACHABLE,
                detail="synthetic source is unavailable",
            )

    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = _service()
    failing_service = _service(source_probe=_FailedProbe())
    name = f"probe-failure-{uuid4()}"
    async with factory.begin() as session:
        created = await service.create(session, _create_request(name))

    with pytest.raises(RtspProbeError):
        async with factory.begin() as session:
            _ = await failing_service.update(
                session,
                created.camera.camera_id,
                CameraPatchRequest.model_validate(
                    {"source_url": "rtsp://operator:new-secret@offline:8554/live"}
                ),
                expected_version=created.camera.version,
            )

    async with factory.begin() as session:
        events = await _events(session, UUID(str(created.camera.camera_id)))
    assert [event.event_type for event in events] == ["camera.created"]


@pytest.mark.anyio
async def test_mutation_and_event_rollback_together(engine: AsyncEngine) -> None:
    """Rolling back a camera create also rolls back its event row."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = _service()
    name = f"rollback-{uuid4()}"

    async with factory() as session:
        transaction = await session.begin()
        try:
            created = await service.create(session, _create_request(name))
            event_rows = await session.scalars(
                select(CameraEvent).where(CameraEvent.camera_id == created.camera.camera_id)
            )
            observed_event_count = len(event_rows.all())
            camera_id = UUID(str(created.camera.camera_id))
        finally:
            await transaction.rollback()

    async with factory.begin() as session:
        camera = await session.scalar(select(Camera).where(Camera.name == name))
        events = await session.scalars(
            select(CameraEvent).where(CameraEvent.camera_id == camera_id)
        )
    assert observed_event_count == 1
    assert camera is None
    assert tuple(events.all()) == ()


@pytest.mark.anyio
async def test_event_insert_failure_rolls_back_camera(engine: AsyncEngine) -> None:
    """A rejected history insert aborts the enclosing camera mutation."""
    factory = async_sessionmaker(engine, expire_on_commit=False)
    service = _service()
    name = "event-write-failure"
    async with engine.begin() as connection:
        _ = await connection.execute(
            text(
                """
                ALTER TABLE camera_events
                ADD CONSTRAINT ck_camera_events_test_reject_name
                CHECK (camera_name <> 'event-write-failure')
                """
            )
        )
    insert_failed = False
    try:
        async with factory() as session:
            transaction = await session.begin()
            try:
                _ = await service.create(session, _create_request(name))
            except IntegrityError:
                insert_failed = True
            finally:
                if transaction.is_active:
                    await transaction.rollback()
    finally:
        async with engine.begin() as connection:
            _ = await connection.execute(
                text("ALTER TABLE camera_events DROP CONSTRAINT ck_camera_events_test_reject_name")
            )

    async with factory.begin() as session:
        camera = await session.scalar(select(Camera).where(Camera.name == name))
        events = await session.scalars(select(CameraEvent).where(CameraEvent.camera_name == name))
    assert insert_failed
    assert camera is None
    assert tuple(events.all()) == ()
