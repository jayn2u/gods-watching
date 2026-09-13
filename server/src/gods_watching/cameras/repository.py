"""Camera-specific durable persistence built on the Task 4 mappings."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, override
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.storage import Camera, CameraSession, StorageRepository


class CameraServiceError(RuntimeError):
    """Base class for typed camera persistence failures."""


class CameraNameConflictError(CameraServiceError):
    """Report a name that is already retained by another camera row."""

    name: str

    def __init__(self, *, name: str) -> None:
        """Retain the conflicting name for structured handling."""
        self.name = name
        super().__init__(name)

    @override
    def __str__(self) -> str:
        return "camera name is already in use"


class CameraRecordNotFoundError(CameraServiceError):
    """Report a camera identifier with no persisted row."""

    camera_id: UUID

    def __init__(self, *, camera_id: UUID) -> None:
        """Retain the missing camera identifier for structured handling."""
        self.camera_id = camera_id
        super().__init__(camera_id)

    @override
    def __str__(self) -> str:
        return "camera was not found"


class CameraDeletedError(CameraServiceError):
    """Report a mutation against a soft-deleted camera."""

    camera_id: UUID

    def __init__(self, *, camera_id: UUID) -> None:
        """Retain the deleted camera identifier for structured handling."""
        self.camera_id = camera_id
        super().__init__(camera_id)

    @override
    def __str__(self) -> str:
        return "camera is deleted"


class StaleCameraVersionError(CameraServiceError):
    """Report an optimistic version fence mismatch."""

    expected: int
    actual: int

    def __init__(self, *, expected: int, actual: int) -> None:
        """Retain both sides of the optimistic version comparison."""
        self.expected = expected
        self.actual = actual
        super().__init__(expected, actual)

    @override
    def __str__(self) -> str:
        return "camera version is stale"


_ACTIVE_SESSION: Final = CameraSession.ended_at.is_(None)


@dataclass(frozen=True, slots=True)
class CameraRepository:
    """Provide row and source-generation operations for the camera service."""

    storage: StorageRepository

    async def ensure_name_available(
        self,
        session: AsyncSession,
        name: str,
        *,
        exclude_camera_id: UUID | None = None,
    ) -> None:
        """Check the database-wide camera-name uniqueness contract."""
        statement = select(Camera.id).where(Camera.name == name)
        if exclude_camera_id is not None:
            statement = statement.where(Camera.id != exclude_camera_id)
        if await session.scalar(statement) is not None:
            raise CameraNameConflictError(name=name)

    async def create(
        self,
        session: AsyncSession,
        request: CameraCreateRequest,
    ) -> Camera:
        """Persist one encrypted camera row after the name check."""
        await self.ensure_name_available(session, request.name)
        return await self.storage.add_camera(
            session,
            name=request.name,
            source_url=request.source_url,
            detection_enabled=request.detection_enabled,
            detection_threshold=request.detection_threshold,
        )

    async def get(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        for_update: bool = False,
        include_deleted: bool = False,
    ) -> Camera:
        """Load one camera row with optional lock and tombstone visibility."""
        statement = select(Camera).where(Camera.id == camera_id)
        if for_update:
            statement = statement.with_for_update()
        camera = await session.scalar(statement)
        if camera is None:
            raise CameraRecordNotFoundError(camera_id=camera_id)
        if camera.deleted_at is not None and not include_deleted:
            raise CameraDeletedError(camera_id=camera_id)
        return camera

    async def list(
        self,
        session: AsyncSession,
        *,
        include_deleted: bool = False,
    ) -> tuple[Camera, ...]:
        """List camera rows in stable creation order."""
        statement = select(Camera).order_by(Camera.created_at, Camera.id)
        if not include_deleted:
            statement = statement.where(Camera.deleted_at.is_(None))
        return tuple((await session.scalars(statement)).all())

    async def source(self, session: AsyncSession, camera: Camera) -> str:
        """Decrypt a source only for the trusted service boundary."""
        return await self.storage.get_camera_source(session, camera.id)

    async def active_session(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        for_update: bool = False,
    ) -> CameraSession | None:
        """Load the current unended source generation for one camera."""
        statement = (
            select(CameraSession)
            .where(CameraSession.camera_id == camera_id, _ACTIVE_SESSION)
            .order_by(CameraSession.started_at.desc(), CameraSession.id.desc())
        )
        if for_update:
            statement = statement.with_for_update()
        return await session.scalar(statement)

    async def end_session(self, session: AsyncSession, camera_session: CameraSession) -> None:
        """End one source generation without touching its historical appearance rows."""
        if camera_session.ended_at is None:
            camera_session.ended_at = datetime.now(UTC)
            await session.flush()

    async def start_session(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        cause: str,
    ) -> CameraSession:
        """Start one new durable source generation after old work has ended."""
        camera_session = await self.storage.start_camera_session(session, camera_id, cause=cause)
        await session.flush()
        return camera_session
