"""Latest-only detector snapshots shared by the worker and authenticated API."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import TYPE_CHECKING, cast, final, override

import anyio
from sqlalchemy import select

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.live import DetectionBox, LiveDetectionResponse
from gods_watching.storage.models import Camera, CameraDetectionLatest, CameraSession

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.cameras.lifecycle import CameraGenerationId
    from gods_watching.inference.detector import DetectorResult
    from gods_watching.media.models import SourceGenerationId
    from gods_watching.storage import Database

LIVE_DETECTION_MAX_AGE_SECONDS = 1.0
LIVE_DETECTION_MAX_BOXES = 300


@dataclass(frozen=True, slots=True)
class LiveDetectionInputError(ValueError):
    """Describe malformed live detector state or writer configuration."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class LiveDetectionBox:
    """Carry one validated source-resolution overlay box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float


@dataclass(frozen=True, slots=True)
class LiveDetectionSnapshot:
    """Carry one detector result across the bounded worker-to-database handoff."""

    camera_id: CameraId
    camera_session_id: CameraSessionId
    db_generation_id: CameraGenerationId
    source_generation_id: SourceGenerationId
    frame_at: datetime
    width: int
    height: int
    boxes: tuple[LiveDetectionBox, ...]

    def __post_init__(self) -> None:
        """Reject malformed timestamps, dimensions, or overlay geometry."""
        if self.frame_at.tzinfo is None or self.frame_at.utcoffset() is None:
            raise LiveDetectionInputError(detail="live detection frame_at must include a timezone")
        if self.width <= 0 or self.height <= 0:
            raise LiveDetectionInputError(detail="live detection dimensions must be positive")
        if len(self.boxes) > LIVE_DETECTION_MAX_BOXES:
            raise LiveDetectionInputError(detail="live detection box count exceeds 300")
        for box in self.boxes:
            if (
                not all(isfinite(value) for value in (box.x1, box.y1, box.x2, box.y2))
                or not isfinite(box.confidence)
                or not 0.0 <= box.confidence <= 1.0
                or not 0.0 <= box.x1 < box.x2 <= self.width
                or not 0.0 <= box.y1 < box.y2 <= self.height
            ):
                raise LiveDetectionInputError(
                    detail="live detection box is outside the source frame"
                )
        object.__setattr__(self, "frame_at", self.frame_at.astimezone(UTC))


def snapshot_from_result(  # noqa: PLR0913 - fields are the persisted overlay boundary
    *,
    camera_id: CameraId,
    camera_session_id: CameraSessionId,
    db_generation_id: CameraGenerationId,
    source_generation_id: SourceGenerationId,
    frame_at: datetime,
    width: int,
    height: int,
    result: DetectorResult,
    threshold: float,
) -> LiveDetectionSnapshot:
    """Filter, clamp, and bound raw detector boxes for immediate overlays."""
    boxes: list[LiveDetectionBox] = []
    for detection in result.detections:
        if detection.class_id != 0 or not isfinite(detection.confidence):
            continue
        if detection.confidence < threshold:
            continue
        x1, y1, x2, y2 = detection.xyxy
        if not all(isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        bounded = LiveDetectionBox(
            x1=max(0.0, min(float(width), x1)),
            y1=max(0.0, min(float(height), y1)),
            x2=max(0.0, min(float(width), x2)),
            y2=max(0.0, min(float(height), y2)),
            confidence=float(detection.confidence),
        )
        if bounded.x1 >= bounded.x2 or bounded.y1 >= bounded.y2:
            continue
        boxes.append(bounded)
        if len(boxes) == LIVE_DETECTION_MAX_BOXES:
            break
    return LiveDetectionSnapshot(
        camera_id=camera_id,
        camera_session_id=camera_session_id,
        db_generation_id=db_generation_id,
        source_generation_id=source_generation_id,
        frame_at=frame_at,
        width=width,
        height=height,
        boxes=tuple(boxes),
    )


@final
class LiveDetectionCameraNotFoundError(LookupError):
    """Report a missing or deleted camera without exposing storage details."""


@final
class LiveDetectionRepository:
    """Read and replace the one latest detector snapshot per camera."""

    async def publish(self, session: AsyncSession, snapshot: LiveDetectionSnapshot) -> bool:
        """Persist a snapshot only when its session is still active and generation-matched."""
        active = await session.scalar(
            select(CameraSession).where(
                CameraSession.id == snapshot.camera_session_id,
                CameraSession.camera_id == snapshot.camera_id,
                CameraSession.generation_id == snapshot.db_generation_id,
                CameraSession.ended_at.is_(None),
            )
        )
        if active is None:
            return False
        current = await session.scalar(
            select(CameraDetectionLatest)
            .where(CameraDetectionLatest.camera_id == snapshot.camera_id)
            .with_for_update()
        )
        if current is not None and current.camera_session_id != snapshot.camera_session_id:
            current_session = await session.scalar(
                select(CameraSession.id).where(
                    CameraSession.id == current.camera_session_id,
                    CameraSession.ended_at.is_(None),
                )
            )
            if current_session is not None:
                return False
        values = {
            "camera_id": snapshot.camera_id,
            "camera_session_id": snapshot.camera_session_id,
            "db_generation_id": snapshot.db_generation_id,
            "source_generation_id": snapshot.source_generation_id,
            "frame_at": snapshot.frame_at,
            "width": snapshot.width,
            "height": snapshot.height,
            "boxes": [
                {
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "confidence": box.confidence,
                }
                for box in snapshot.boxes
            ],
        }
        if current is None:
            session.add(CameraDetectionLatest(**values))
        else:
            for key, value in values.items():
                if key != "camera_id":
                    setattr(current, key, value)
        await session.flush()
        return True

    async def read(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        now: datetime | None = None,
    ) -> LiveDetectionResponse:
        """Return a fresh current-generation result, fencing stale or disabled rows."""
        camera = await session.scalar(
            select(Camera).where(Camera.id == camera_id, Camera.deleted_at.is_(None))
        )
        if camera is None:
            raise LiveDetectionCameraNotFoundError
        active = await session.scalar(
            select(CameraSession)
            .where(CameraSession.camera_id == camera_id, CameraSession.ended_at.is_(None))
            .order_by(CameraSession.started_at.desc(), CameraSession.id.desc())
        )
        current_session_id = None if active is None else CameraSessionId(active.id)
        current = await session.scalar(
            select(CameraDetectionLatest).where(CameraDetectionLatest.camera_id == camera_id)
        )
        if (
            current is None
            or active is None
            or current.camera_session_id != active.id
            or current.db_generation_id != active.generation_id
        ):
            return _empty_response(CameraId(camera_id), current_session_id)
        observed_at = now or datetime.now(UTC)
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise LiveDetectionInputError(detail="live detection now must include a timezone")
        age = max(0.0, (observed_at.astimezone(UTC) - current.frame_at).total_seconds())
        boxes = () if age > LIVE_DETECTION_MAX_AGE_SECONDS else _response_boxes(current.boxes)
        return LiveDetectionResponse(
            camera_id=CameraId(camera_id),
            camera_session_id=current_session_id,
            frame_at=current.frame_at,
            frame_age_seconds=age,
            width=current.width,
            height=current.height,
            boxes=boxes,
        )


def _empty_response(
    camera_id: CameraId,
    camera_session_id: CameraSessionId | None,
) -> LiveDetectionResponse:
    return LiveDetectionResponse(
        camera_id=camera_id,
        camera_session_id=camera_session_id,
        frame_at=None,
        frame_age_seconds=None,
        width=None,
        height=None,
        boxes=(),
    )


def _response_boxes(value: object) -> tuple[DetectionBox, ...]:
    if not isinstance(value, list):
        return ()
    boxes: list[DetectionBox] = []
    items = cast("list[object]", value)
    for item in items[:LIVE_DETECTION_MAX_BOXES]:
        if not isinstance(item, dict):
            continue
        try:
            boxes.append(
                DetectionBox.model_validate(cast("dict[str, object]", item))
            )
        except ValueError:
            continue
    return tuple(boxes)


LiveDetectionSink = Callable[[LiveDetectionSnapshot], Awaitable[None]]


@final
class LiveDetectionPublisher:
    """Keep one pending snapshot per camera and persist it in a short background loop."""

    def __init__(self, database: Database, *, interval_seconds: float = 0.1) -> None:
        """Bind the shared database and a bounded write cadence."""
        if interval_seconds <= 0.0 or not isfinite(interval_seconds):
            raise LiveDetectionInputError(
                detail="live detection publisher interval must be positive and finite"
            )
        self._database = database
        self._interval_seconds = interval_seconds
        self._repository = LiveDetectionRepository()
        self._pending: dict[CameraId, LiveDetectionSnapshot] = {}
        self._lock = anyio.Lock()

    async def publish(self, snapshot: LiveDetectionSnapshot) -> None:
        """Replace an older pending result without waiting on PostgreSQL."""
        async with self._lock:
            self._pending[snapshot.camera_id] = snapshot

    async def flush_once(self) -> None:
        """Attempt pending writes while isolating database failures per camera."""
        async with self._lock:
            snapshots = tuple(self._pending.values())
            for snapshot in snapshots:
                _ = self._pending.pop(snapshot.camera_id, None)
        for snapshot in snapshots:
            try:
                async with self._database.transaction() as session:
                    _ = await self._repository.publish(session, snapshot)
            except Exception:  # noqa: BLE001 - live overlays must never stop ingest
                async with self._lock:
                    _ = self._pending.setdefault(snapshot.camera_id, snapshot)

    async def run(self, stop_event: anyio.Event) -> None:
        """Persist latest snapshots until the pipeline worker stops."""
        while not stop_event.is_set():
            await self.flush_once()
            with anyio.move_on_after(self._interval_seconds):
                _ = await stop_event.wait()


__all__ = [
    "LIVE_DETECTION_MAX_AGE_SECONDS",
    "LIVE_DETECTION_MAX_BOXES",
    "LiveDetectionBox",
    "LiveDetectionCameraNotFoundError",
    "LiveDetectionInputError",
    "LiveDetectionPublisher",
    "LiveDetectionRepository",
    "LiveDetectionSink",
    "LiveDetectionSnapshot",
    "snapshot_from_result",
]
