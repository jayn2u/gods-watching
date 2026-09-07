"""Durable retention, quota, and wall-slot settings service."""

from dataclasses import dataclass
from typing import Final, override
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.identifiers import CameraId
from gods_watching.contracts.settings import SettingsPatchRequest, SettingsResponse
from gods_watching.storage import ApplicationSettings

DEFAULT_RETENTION_DAYS: Final = 7
DEFAULT_QUOTA_BYTES: Final = 100_000_000_000
WALL_SLOT_COUNT: Final = 4
DEFAULT_WALL_SLOTS: Final[tuple[CameraId | None, ...]] = (None, None, None, None)
_EMPTY_SLOT: Final = UUID(int=0)
MANAGED_QUOTA_LABEL: Final = (
    "physical crop files including pending garbage plus PostgreSQL application "
    "relations and indexes"
)
OPERATIONAL_STORAGE_LABEL: Final = "model assets, bounded logs, and PostgreSQL WAL"


class SettingsServiceError(RuntimeError):
    """Base class for typed settings persistence failures."""


class SettingsNoChangesError(SettingsServiceError):
    """Report an empty settings patch."""

    @override
    def __str__(self) -> str:
        return "at least one settings field is required"


class SettingsStorageError(SettingsServiceError):
    """Report settings data that cannot satisfy the four-slot contract."""

    @override
    def __str__(self) -> str:
        return "stored settings do not satisfy the wall-slot contract"


@dataclass(frozen=True, slots=True)
class QuotaAccountingContract:
    """Describe quota accounting without implementing retention or eviction."""

    managed_label: str = MANAGED_QUOTA_LABEL
    operational_label: str = OPERATIONAL_STORAGE_LABEL
    eviction_owner: str = "retention service (Task 13)"


@dataclass(frozen=True, slots=True)
class SettingsService:
    """Read and update the singleton settings row using caller-owned transactions."""

    accounting: QuotaAccountingContract = QuotaAccountingContract()

    async def get(self, session: AsyncSession) -> SettingsResponse:
        """Return persisted settings and materialize approved wall-slot defaults."""
        row = await self._row(session)
        slots = _wall_slots(row)
        storage_slots = _storage_slots(slots)
        if row.wall_slot_ids != storage_slots:
            row.wall_slot_ids = storage_slots
            await session.flush()
        return _response(row, slots)

    async def update(
        self,
        session: AsyncSession,
        patch: SettingsPatchRequest,
    ) -> SettingsResponse:
        """Validate positive settings and persist one atomic patch."""
        if (
            patch.retention_days is None
            and patch.quota_bytes is None
            and patch.wall_slot_ids is None
        ):
            raise SettingsNoChangesError
        row = await self._row(session)
        current = _wall_slots(row)
        if patch.retention_days is not None:
            row.retention_days = patch.retention_days
        if patch.quota_bytes is not None:
            row.quota_bytes = patch.quota_bytes
        if patch.wall_slot_ids is not None:
            slots = _validate_slots(patch.wall_slot_ids)
            row.wall_slot_ids = _storage_slots(slots)
        else:
            slots = current
        await session.flush()
        return _response(row, slots)

    async def _row(self, session: AsyncSession) -> ApplicationSettings:
        row = await session.scalar(
            select(ApplicationSettings)
            .where(ApplicationSettings.singleton.is_(True))
            .with_for_update()
        )
        if row is None:
            row = ApplicationSettings(
                singleton=True,
                retention_days=DEFAULT_RETENTION_DAYS,
                quota_bytes=DEFAULT_QUOTA_BYTES,
                wall_slot_ids=_storage_slots(DEFAULT_WALL_SLOTS),
            )
            session.add(row)
            await session.flush()
        return row


def _wall_slots(row: ApplicationSettings) -> tuple[CameraId | None, ...]:
    if not row.wall_slot_ids:
        return DEFAULT_WALL_SLOTS
    if len(row.wall_slot_ids) != WALL_SLOT_COUNT:
        raise SettingsStorageError
    return tuple(None if slot == _EMPTY_SLOT else CameraId(slot) for slot in row.wall_slot_ids)


def _storage_slots(slots: tuple[CameraId | None, ...]) -> list[UUID]:
    return [slot if slot is not None else _EMPTY_SLOT for slot in slots]


def _validate_slots(
    slots: tuple[CameraId | None, CameraId | None, CameraId | None, CameraId | None],
) -> tuple[CameraId | None, ...]:
    identifiers = tuple(slot for slot in slots if slot is not None)
    if len(set(identifiers)) != len(identifiers):
        raise SettingsStorageError
    return slots


def _response(
    row: ApplicationSettings,
    slots: tuple[CameraId | None, ...],
) -> SettingsResponse:
    if len(slots) != WALL_SLOT_COUNT:
        raise SettingsStorageError
    return SettingsResponse(
        retention_days=row.retention_days,
        quota_bytes=row.quota_bytes,
        wall_slot_ids=(slots[0], slots[1], slots[2], slots[3]),
    )
