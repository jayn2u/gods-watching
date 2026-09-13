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
    thresholds: tuple[tuple[CameraId, float], ...]


def plan_reconcile(
    desired: Mapping[CameraId, DesiredCamera],
    running: Mapping[CameraId, RunningCamera],
) -> ReconcilePlan:
    """Compare committed sessions with live workers and return the minimal changes."""
    stop: list[CameraId] = []
    start: list[DesiredCamera] = []
    thresholds: list[tuple[CameraId, float]] = []
    for camera_id in sorted(desired.keys() | running.keys(), key=str):
        wanted = desired.get(camera_id)
        current = running.get(camera_id)
        if wanted is None:
            stop.append(camera_id)
            continue
        if current is None:
            start.append(wanted)
            continue
        if wanted.session_id != current.session_id or wanted.generation_id != current.generation_id:
            stop.append(camera_id)
            start.append(wanted)
            continue
        if wanted.threshold != current.threshold:
            thresholds.append((camera_id, wanted.threshold))
    return ReconcilePlan(stop=tuple(stop), start=tuple(start), thresholds=tuple(thresholds))


__all__ = ["DesiredCamera", "ReconcilePlan", "RunningCamera", "plan_reconcile"]
