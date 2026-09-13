"""Transactional repository operations for cameras and appearance revisions."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import final, override
from uuid import UUID

from pydantic import AnyUrl
from sqlalchemy import BigInteger, cast, func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.appearances import AppearancePublication

from .crypto import CredentialCipher
from .models import Appearance, Camera, CameraSession, CropGarbage


@final
class CameraNotFoundError(Exception):
    """Report a missing camera repository row."""

    def __init__(self, camera_id: UUID) -> None:
        """Retain the typed missing camera identifier."""
        self.camera_id = camera_id
        super().__init__(camera_id)

    @override
    def __str__(self) -> str:
        """Return the missing camera identifier."""
        return f"camera {self.camera_id} not found"


@final
class AppearanceNotFoundError(Exception):
    """Report a missing appearance repository row."""

    def __init__(self, appearance_id: UUID) -> None:
        """Retain the typed missing appearance identifier."""
        self.appearance_id = appearance_id
        super().__init__(appearance_id)

    @override
    def __str__(self) -> str:
        """Return the missing appearance identifier."""
        return f"appearance {self.appearance_id} not found"


@final
class StaleAppearanceVersionError(Exception):
    """Report a publication that cannot advance the committed version by one."""

    def __init__(
        self,
        appearance_id: UUID,
        attempted_version: int,
        committed_version: int | None,
    ) -> None:
        """Retain typed attempted and committed revision state."""
        self.appearance_id = appearance_id
        self.attempted_version = attempted_version
        self.committed_version = committed_version
        super().__init__(appearance_id, attempted_version, committed_version)

    @override
    def __str__(self) -> str:
        """Return attempted and committed versions."""
        return (
            f"appearance {self.appearance_id} version {self.attempted_version} is stale; "
            f"committed version is {self.committed_version}"
        )


@dataclass(frozen=True, slots=True)
class StorageRepository:
    """Apply storage mutations within a caller-owned SQLAlchemy transaction."""

    credential_cipher: CredentialCipher

    async def add_camera(
        self,
        session: AsyncSession,
        *,
        name: str,
        source_url: AnyUrl,
        detection_enabled: bool = True,
        detection_threshold: float = 0.5,
    ) -> Camera:
        """Persist an encrypted RTSP camera configuration."""
        host = source_url.unicode_host()
        if host is None:
            raise AssertionError
        camera = Camera(
            name=name,
            source_ciphertext=self.credential_cipher.encrypt(str(source_url)),
            source_host=host,
            source_port=source_url.port,
            detection_enabled=detection_enabled,
            detection_threshold=detection_threshold,
        )
        session.add(camera)
        await session.flush()
        return camera

    async def get_camera_source(self, session: AsyncSession, camera_id: UUID) -> str:
        """Decrypt a camera source for trusted server-side use."""
        camera = await session.get(Camera, camera_id)
        if camera is None:
            raise CameraNotFoundError(camera_id=camera_id)
        return self.credential_cipher.decrypt(camera.source_ciphertext)

    async def start_camera_session(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        cause: str,
    ) -> CameraSession:
        """Start a new camera-scoped track identity generation."""
        if await session.get(Camera, camera_id) is None:
            raise CameraNotFoundError(camera_id=camera_id)
        camera_session = CameraSession(camera_id=camera_id, cause=cause)
        session.add(camera_session)
        await session.flush()
        return camera_session

    async def publish_appearance(
        self,
        session: AsyncSession,
        publication: AppearancePublication,
    ) -> Appearance:
        """Insert version one or atomically advance the committed revision."""
        appearance_id = UUID(str(publication.appearance_id))
        statement = select(Appearance).where(Appearance.id == appearance_id).with_for_update()
        existing = await session.scalar(statement)
        if existing is None:
            if publication.representative_version != 1:
                raise StaleAppearanceVersionError(
                    appearance_id=appearance_id,
                    attempted_version=publication.representative_version,
                    committed_version=None,
                )
            appearance = self._new_appearance(publication)
            session.add(appearance)
            await session.flush()
            return appearance
        if (
            existing.tombstoned_at is not None
            or publication.representative_version != existing.representative_version + 1
        ):
            raise StaleAppearanceVersionError(
                appearance_id=appearance_id,
                attempted_version=publication.representative_version,
                committed_version=existing.representative_version,
            )
        self._enqueue_crop_garbage(session, existing)
        self._replace_appearance(existing, publication)
        await session.flush()
        return existing

    async def tombstone_appearance(self, session: AsyncSession, appearance_id: UUID) -> None:
        """Hide a representative vector and enqueue its crop for retryable GC."""
        statement = select(Appearance).where(Appearance.id == appearance_id).with_for_update()
        appearance = await session.scalar(statement)
        if appearance is None:
            raise AppearanceNotFoundError(appearance_id=appearance_id)
        if appearance.tombstoned_at is None:
            self._enqueue_crop_garbage(session, appearance)
            appearance.tombstoned_at = datetime.now(UTC)
            appearance.embedding = None
            await session.flush()

    async def finalize_tombstoned_appearance(
        self,
        session: AsyncSession,
        *,
        appearance_id: UUID,
    ) -> bool:
        """Delete one tombstoned appearance after its crop outbox drains."""
        appearance = await session.scalar(
            select(Appearance).where(Appearance.id == appearance_id).with_for_update()
        )
        if appearance is None or appearance.tombstoned_at is None:
            return False
        pending = await session.scalar(
            select(CropGarbage.id)
            .where(CropGarbage.appearance_id == appearance_id)
            .limit(1)
            .with_for_update()
        )
        if pending is not None:
            return False
        await session.delete(appearance)
        return True

    async def application_relation_sizes(self, session: AsyncSession) -> Mapping[str, int]:
        """Report physical table and index bytes used by application relations."""
        relation_names = (
            "cameras",
            "camera_sessions",
            "appearances",
            "crop_gc",
            "sessions",
            "settings",
            "operator_credentials",
        )
        sizes: dict[str, int] = {}
        for relation_name in relation_names:
            expression = cast(
                func.pg_total_relation_size(literal_column(f"'{relation_name}'::regclass")),
                BigInteger,
            )
            relation_size = await session.scalar(select(expression))
            if relation_size is None:
                raise AssertionError
            sizes[relation_name] = relation_size
        return sizes

    @staticmethod
    def _enqueue_crop_garbage(session: AsyncSession, appearance: Appearance) -> None:
        session.add(
            CropGarbage(
                appearance_id=appearance.id,
                object_key=appearance.crop_object_key,
                byte_size=appearance.byte_size,
            )
        )

    @staticmethod
    def _new_appearance(publication: AppearancePublication) -> Appearance:
        box = publication.bounding_box
        return Appearance(
            id=UUID(str(publication.appearance_id)),
            camera_id=UUID(str(publication.camera_id)),
            session_id=UUID(str(publication.session_id)),
            track_id=publication.track_id,
            first_seen=publication.first_seen,
            last_seen=publication.last_seen,
            ended_at=publication.ended_at,
            representative_version=publication.representative_version,
            crop_object_key=publication.crop_object_key,
            x_min=box.x_min,
            y_min=box.y_min,
            x_max=box.x_max,
            y_max=box.y_max,
            source_width=publication.source_width,
            source_height=publication.source_height,
            detector_confidence=publication.detector_confidence,
            crop_quality=publication.crop_quality,
            byte_size=publication.byte_size,
            embedded_at=publication.embedded_at,
            model_id=publication.model_id,
            model_revision=publication.model_revision,
            embedding="[" + ",".join(str(item) for item in publication.embedding) + "]",
        )

    @staticmethod
    def _replace_appearance(
        appearance: Appearance,
        publication: AppearancePublication,
    ) -> None:
        box = publication.bounding_box
        appearance.last_seen = publication.last_seen
        appearance.ended_at = publication.ended_at
        appearance.representative_version = publication.representative_version
        appearance.crop_object_key = publication.crop_object_key
        appearance.x_min = box.x_min
        appearance.y_min = box.y_min
        appearance.x_max = box.x_max
        appearance.y_max = box.y_max
        appearance.source_width = publication.source_width
        appearance.source_height = publication.source_height
        appearance.detector_confidence = publication.detector_confidence
        appearance.crop_quality = publication.crop_quality
        appearance.byte_size = publication.byte_size
        appearance.embedded_at = publication.embedded_at
        appearance.model_id = publication.model_id
        appearance.model_revision = publication.model_revision
        appearance.embedding = "[" + ",".join(str(item) for item in publication.embedding) + "]"
