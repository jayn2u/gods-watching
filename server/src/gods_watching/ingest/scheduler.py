"""Fair bounded scheduling for one detector request per camera."""

from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Final, override

import anyio

from gods_watching.contracts.identifiers import CameraId

_MIN_TARGET_FPS: Final = 0.1


@dataclass(frozen=True, slots=True)
class SchedulerConfigurationError(ValueError):
    """Describe an invalid detector sampling rate."""

    target_fps: float

    @override
    def __str__(self) -> str:
        return f"target_fps must be finite and positive: {self.target_fps}"


@dataclass(frozen=True, slots=True)
class UnknownCameraError(KeyError):
    """Describe an operation for a camera absent from the scheduler."""

    camera_id: CameraId

    @override
    def __str__(self) -> str:
        return f"camera is not registered: {self.camera_id}"


@dataclass(slots=True)
class _ScheduleState:
    next_due: float
    in_flight: bool = False


class FairRoundRobinScheduler:
    """Rotate due cameras at a fixed target rate while enforcing single-flight RPCs."""

    def __init__(self, *, target_fps: float = 5.0) -> None:
        """Create a scheduler with one bounded due slot per registered camera."""
        if not isfinite(target_fps) or target_fps < _MIN_TARGET_FPS:
            raise SchedulerConfigurationError(target_fps=target_fps)
        self._period_seconds: float = 1.0 / target_fps
        self._states: dict[CameraId, _ScheduleState] = {}
        self._order: list[CameraId] = []
        self._cursor: int = 0

    @property
    def period_seconds(self) -> float:
        """Return the target interval between accepted samples for one camera."""
        return self._period_seconds

    @property
    def in_flight_count(self) -> int:
        """Return the number of cameras with an outstanding detector request."""
        return sum(state.in_flight for state in self._states.values())

    def register(self, camera_id: CameraId) -> None:
        """Add one camera to the fair rotation, idempotently."""
        if camera_id in self._states:
            return
        self._states[camera_id] = _ScheduleState(next_due=0.0)
        self._order.append(camera_id)

    def unregister(self, camera_id: CameraId) -> None:
        """Remove one camera and release any scheduler-owned flight slot."""
        _ = self._require(camera_id)
        del self._states[camera_id]
        index = self._order.index(camera_id)
        _ = self._order.pop(index)
        if self._order:
            self._cursor %= len(self._order)
        else:
            self._cursor = 0

    def next_due(self, *, now_monotonic: float) -> CameraId | None:
        """Return the next due non-inflight camera in round-robin order."""
        if not self._order:
            return None
        for offset in range(len(self._order)):
            index = (self._cursor + offset) % len(self._order)
            camera_id = self._order[index]
            state = self._states[camera_id]
            if not state.in_flight and now_monotonic >= state.next_due:
                self._cursor = (index + 1) % len(self._order)
                return camera_id
        return None

    def mark_dispatched(self, camera_id: CameraId, *, now_monotonic: float) -> None:
        """Reserve one detector request and schedule the camera's next due time."""
        state = self._require(camera_id)
        if state.in_flight:
            return
        state.in_flight = True
        state.next_due = now_monotonic + self._period_seconds

    def complete(self, camera_id: CameraId) -> None:
        """Release a camera's single-flight slot after success, error, or cancellation."""
        self._require(camera_id).in_flight = False

    async def run(
        self,
        dispatch: Callable[[CameraId], Awaitable[None]],
        *,
        stop_event: anyio.Event,
    ) -> None:
        """Run the fair loop until its caller signals cancellation."""
        async with anyio.create_task_group() as task_group:
            while not stop_event.is_set():
                now = monotonic()
                camera_id = self.next_due(now_monotonic=now)
                if camera_id is None:
                    await anyio.sleep(min(self._period_seconds / 4.0, 0.05))
                    continue
                self.mark_dispatched(camera_id, now_monotonic=now)
                task_group.start_soon(self._dispatch_one, dispatch, camera_id)

    async def _dispatch_one(
        self,
        dispatch: Callable[[CameraId], Awaitable[None]],
        camera_id: CameraId,
    ) -> None:
        try:
            await dispatch(camera_id)
        finally:
            with suppress(UnknownCameraError):
                self.complete(camera_id)

    def _require(self, camera_id: CameraId) -> _ScheduleState:
        state = self._states.get(camera_id)
        if state is None:
            raise UnknownCameraError(camera_id=camera_id)
        return state


__all__ = [
    "FairRoundRobinScheduler",
    "SchedulerConfigurationError",
    "UnknownCameraError",
]
