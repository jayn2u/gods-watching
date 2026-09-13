"""Current-pointer detail and crop reads for authenticated route composition."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.appearances import AppearanceResponse
from gods_watching.storage import CropObjectStore

from .errors import CropChangedDuringReadError, CropUnavailableError
from .repository import SearchCandidate, SearchRepository
from .service import appearance_response


@dataclass(frozen=True, slots=True)
class CropPayload:
    """Return crop bytes together with the current object metadata."""

    appearance: AppearanceResponse
    payload: bytes
    object_key: str
    representative_version: int


@dataclass(frozen=True, slots=True)
class AppearanceLookupService:
    """Revalidate a committed appearance pointer before and after local object reads."""

    repository: SearchRepository
    crop_store: CropObjectStore

    async def get_detail(
        self,
        session: AsyncSession,
        appearance_id: UUID,
    ) -> AppearanceResponse:
        """Return the current visible detail row, including archived camera labels."""
        row = await self.repository.get_visible(session, appearance_id)
        if row is None:
            raise CropUnavailableError(appearance_id=appearance_id)
        appearance, camera = row
        return appearance_response(SearchCandidate(appearance, camera, None))

    async def get_crop(
        self,
        session: AsyncSession,
        appearance_id: UUID,
    ) -> CropPayload:
        """Read only the object key and version currently committed by PostgreSQL."""
        initial = await self.repository.get_visible(session, appearance_id)
        if initial is None:
            raise CropUnavailableError(appearance_id=appearance_id)
        appearance, camera = initial
        object_key = appearance.crop_object_key
        version = appearance.representative_version
        try:
            payload = self.crop_store.read(object_key)
        except (FileNotFoundError, OSError, ValueError) as error:
            raise CropUnavailableError(appearance_id=appearance_id) from error
        current = await self.repository.get_visible(session, appearance_id)
        if (
            current is None
            or current[0].crop_object_key != object_key
            or current[0].representative_version != version
        ):
            raise CropChangedDuringReadError(
                appearance_id=appearance_id,
                representative_version=version,
            )
        return CropPayload(
            appearance=appearance_response(SearchCandidate(appearance, camera, None)),
            payload=payload,
            object_key=object_key,
            representative_version=version,
        )
