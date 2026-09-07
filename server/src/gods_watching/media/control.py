"""Camera source activation and generation fencing."""

from typing import Protocol, final

from gods_watching.contracts.identifiers import CameraId

from .generation import SourceGenerationCoordinator
from .models import MediaPath, SourceGenerationId


class MediaControlGateway(Protocol):
    """Expose only the MediaMTX path mutations needed by camera supervision."""

    async def upsert_path(self, path: MediaPath, source_url: str) -> None:
        """Create or replace one RTSP/TCP path."""
        ...

    async def delete_path(self, path: MediaPath) -> None:
        """Delete one configured path."""
        ...


@final
class MediaControlAdapter:
    """Fence source configuration changes with ordered generation events."""

    def __init__(
        self,
        *,
        gateway: MediaControlGateway,
        generations: SourceGenerationCoordinator,
    ) -> None:
        """Bind private control and downstream generation coordination."""
        self._gateway = gateway
        self._generations = generations

    async def activate(
        self,
        camera_id: CameraId,
        source_url: str,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId:
        """Cancel old work before applying a source and starting its generation."""
        return await self._generations.replace(
            camera_id,
            lambda: self._gateway.upsert_path(camera_path(camera_id), source_url),
            publish_events=publish_events,
        )

    async def activate_source(
        self,
        camera_id: CameraId,
        source_url: str,
        *,
        publish_started: bool = True,
    ) -> SourceGenerationId:
        """Return a source identity while optionally deferring START publication."""
        return await self.activate(
            camera_id,
            source_url,
            publish_events=publish_started,
        )

    async def deactivate(
        self,
        camera_id: CameraId,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        """Delete a path and end any active generation."""
        return await self._generations.deactivate(
            camera_id,
            lambda: self._gateway.delete_path(camera_path(camera_id)),
            publish_events=publish_events,
        )

    async def end_source(
        self,
        camera_id: CameraId,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        """End one source and optionally suppress the legacy unbound events."""
        return await self.deactivate(camera_id, publish_events=publish_events)

    async def source_lost(
        self,
        camera_id: CameraId,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        """End work when MediaMTX reports that a source became unavailable."""
        return await self._generations.source_lost(
            camera_id,
            publish_events=publish_events,
        )

    async def active_generation(self, camera_id: CameraId) -> SourceGenerationId | None:
        """Return the current source generation without changing live media."""
        return await self._generations.active_generation(camera_id)


def camera_path(camera_id: CameraId) -> MediaPath:
    """Derive the private gateway path without source credentials."""
    return MediaPath(f"camera/{camera_id}")
