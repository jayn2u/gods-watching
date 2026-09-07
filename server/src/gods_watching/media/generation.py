"""Source-generation event coordination."""

from collections.abc import Awaitable, Callable
from typing import Protocol, final
from uuid import uuid4

import anyio

from gods_watching.contracts.identifiers import CameraId

from .models import GenerationEvent, GenerationEventKind, SourceGenerationId


class GenerationEventSink(Protocol):
    """Consume ordered source-generation lifecycle events."""

    async def publish(self, event: GenerationEvent) -> None:
        """Publish one ordered event to downstream supervision."""
        ...


@final
class SourceGenerationCoordinator:
    """End old camera work before a replacement generation can start."""

    def __init__(self, *, sink: GenerationEventSink) -> None:
        """Create an empty camera-generation registry."""
        self._sink = sink
        self._active: dict[CameraId, SourceGenerationId] = {}
        self._locks: dict[CameraId, anyio.Lock] = {}

    async def begin(self, camera_id: CameraId) -> SourceGenerationId:
        """Start a fresh generation, ending any prior generation first."""
        async with self._lock_for(camera_id):
            await self._end_active(camera_id)
            return await self._start_generation(camera_id)

    async def replace(
        self,
        camera_id: CameraId,
        activate: Callable[[], Awaitable[None]],
    ) -> SourceGenerationId:
        """End old work, apply a source, and publish its new generation atomically."""
        async with self._lock_for(camera_id):
            await self._end_active(camera_id)
            await activate()
            return await self._start_generation(camera_id)

    async def deactivate(
        self,
        camera_id: CameraId,
        remove: Callable[[], Awaitable[None]],
    ) -> SourceGenerationId | None:
        """End old work and remove its source while holding the camera fence."""
        async with self._lock_for(camera_id):
            ended = await self._source_lost(camera_id)
            await remove()
            return ended

    async def source_lost(self, camera_id: CameraId) -> SourceGenerationId | None:
        """Publish source loss and terminal generation events once."""
        async with self._lock_for(camera_id):
            return await self._source_lost(camera_id)

    def _lock_for(self, camera_id: CameraId) -> anyio.Lock:
        return self._locks.setdefault(camera_id, anyio.Lock())

    async def _start_generation(self, camera_id: CameraId) -> SourceGenerationId:
        generation_id = SourceGenerationId(uuid4())
        self._active[camera_id] = generation_id
        await self._sink.publish(
            GenerationEvent(
                camera_id=camera_id,
                generation_id=generation_id,
                kind=GenerationEventKind.STARTED,
            )
        )
        return generation_id

    async def _end_active(self, camera_id: CameraId) -> None:
        generation_id = self._active.pop(camera_id, None)
        if generation_id is None:
            return
        await self._sink.publish(
            GenerationEvent(
                camera_id=camera_id,
                generation_id=generation_id,
                kind=GenerationEventKind.ENDED,
            )
        )

    async def _source_lost(self, camera_id: CameraId) -> SourceGenerationId | None:
        generation_id = self._active.pop(camera_id, None)
        if generation_id is None:
            return None
        await self._sink.publish(
            GenerationEvent(
                camera_id=camera_id,
                generation_id=generation_id,
                kind=GenerationEventKind.SOURCE_LOST,
            )
        )
        await self._sink.publish(
            GenerationEvent(
                camera_id=camera_id,
                generation_id=generation_id,
                kind=GenerationEventKind.ENDED,
            )
        )
        return generation_id
