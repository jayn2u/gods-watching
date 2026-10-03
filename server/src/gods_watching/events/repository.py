"""Write immutable camera events in the caller's transaction."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.events import EventExportFilters
from gods_watching.storage import CameraEvent


class EventRepository:
    """Add one event without owning transaction or connection lifecycle."""

    @staticmethod
    def statement(
        filters: EventExportFilters,
    ) -> Select[tuple[int, datetime, str, UUID, str]]:
        """Select only the five export fields in stable event-ID order."""
        statement = select(
            CameraEvent.id,
            CameraEvent.occurred_at,
            CameraEvent.event_type,
            CameraEvent.camera_id,
            CameraEvent.camera_name,
        ).order_by(CameraEvent.id)
        if filters.camera_id is not None:
            statement = statement.where(CameraEvent.camera_id == filters.camera_id)
        if filters.since is not None:
            statement = statement.where(CameraEvent.occurred_at >= filters.since)
        if filters.until is not None:
            statement = statement.where(CameraEvent.occurred_at < filters.until)
        return statement

    @staticmethod
    async def record(
        session: AsyncSession,
        *,
        event_type: str,
        camera_id: UUID,
        camera_name: str,
    ) -> CameraEvent:
        """Add and flush a safe event projection in the supplied transaction."""
        event = CameraEvent(
            event_type=event_type,
            camera_id=camera_id,
            camera_name=camera_name,
        )
        session.add(event)
        await session.flush()
        return event
