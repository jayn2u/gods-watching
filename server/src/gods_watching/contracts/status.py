"""Operational status response contracts."""

from typing import Literal

from .base import ContractModel
from .identifiers import CameraId
from .primitives import UtcDatetime


class CameraStatus(ContractModel):
    """Report one camera's observed ingest and detection state."""

    camera_id: CameraId
    state: Literal["online", "stale", "offline", "disabled"]
    last_ingest_at: UtcDatetime | None


class StatusResponse(ContractModel):
    """Report service readiness without claiming unverified health."""

    state: Literal["ready", "degraded", "starting"]
    cameras: tuple[CameraStatus, ...]
    inference_ready: bool
    persistence_paused: bool
