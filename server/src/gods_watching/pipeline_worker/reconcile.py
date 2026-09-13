"""Plan worker changes from committed camera sessions without touching I/O."""

from collections.abc import Mapping
from dataclasses import dataclass

from gods_watching.cameras import CameraGenerationId
from gods_watching.contracts.identifiers import CameraId, CameraSessionId


@dataclass(frozen=True, slots=True)
class DesiredCamera:
    """Describe one camera whose committed session should be decoded and detected."""

    camera_id: CameraId
    version: int
    session_id: CameraSessionId
    generation_id: CameraGenerationId
    source_url: str
    threshold: float


@dataclass(frozen=True, slots=True)
class RunningCamera:
    """Describe the generation and threshold a live ingest worker currently owns."""

    camera_id: CameraId
    version: int
    session_id: CameraSessionId
    generation_id: CameraGenerationId
    threshold: float


@dataclass(frozen=True, slots=True)
class ReconcilePlan:
    """Order stops before starts so a replaced camera never runs two generations."""

    stop: tuple[CameraId, ...]
    start: tuple[DesiredCamera, ...]


def plan_reconcile(
    desired: Mapping[CameraId, DesiredCamera],
    running: Mapping[CameraId, RunningCamera],
) -> ReconcilePlan:
    """Compare committed sessions with live workers and return stops and starts."""
    stop: list[CameraId] = []
    start: list[DesiredCamera] = []
    for camera_id in sorted(desired.keys() | running.keys(), key=str):
        wanted = desired.get(camera_id)
        current = running.get(camera_id)
        if wanted is None:
            stop.append(camera_id)
            continue
        if current is None:
            start.append(wanted)
            continue
        # Publication fences handoffs on the exact camera version, so any committed edit
        # (threshold or rename included) needs a worker bound to the new version.
        if (
            wanted.session_id != current.session_id
            or wanted.generation_id != current.generation_id
            or wanted.version != current.version
            or wanted.threshold != current.threshold
        ):
            stop.append(camera_id)
            start.append(wanted)
    return ReconcilePlan(stop=tuple(stop), start=tuple(start))


__all__ = ["DesiredCamera", "ReconcilePlan", "RunningCamera", "plan_reconcile"]
