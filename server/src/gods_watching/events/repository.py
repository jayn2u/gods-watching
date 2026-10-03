"""Write immutable camera events in the caller's transaction."""

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.storage import CameraEvent


class EventRepository:
    """Add one event without owning transaction or connection lifecycle."""

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
