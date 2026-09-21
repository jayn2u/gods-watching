"""Operational status response contracts."""

from typing import Literal

from .base import ContractModel
from .identifiers import CameraId, CameraSessionId
from .primitives import UtcDatetime


class CameraStatus(ContractModel):
    """Report one camera's observed ingest and detection state."""

    camera_id: CameraId
    camera_session_id: CameraSessionId | None
    state: Literal["online", "stale", "offline", "disabled"]
    last_ingest_at: UtcDatetime | None
    frame_age_seconds: float | None
    actual_framerate: float
    detector_framerate: float
    dropped_frames: int
    detector_requests: int
    detector_results: int
    last_error: str | None


class StatusResponse(ContractModel):
    """Report service readiness without claiming unverified health."""

    state: Literal["ready", "degraded", "starting"]
    cameras: tuple[CameraStatus, ...]
    inference_ready: bool
    persistence_paused: bool
    worker_updated_at: UtcDatetime | None
    storage_managed_bytes: int
    storage_quota_bytes: int
    indexing_queue_depth: int
    last_searchable_latency_seconds: float | None
