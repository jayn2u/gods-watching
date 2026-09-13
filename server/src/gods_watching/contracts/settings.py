"""Operator settings request and response contracts."""

from typing import Annotated

from pydantic import Field

from .base import ContractModel
from .identifiers import CameraId

RetentionDays = Annotated[int, Field(ge=1)]
QuotaBytes = Annotated[int, Field(gt=0)]
WallSlots = tuple[CameraId | None, CameraId | None, CameraId | None, CameraId | None]


class SettingsPatchRequest(ContractModel):
    """Change global retention, quota, or durable wall placement."""

    retention_days: RetentionDays | None = None
    quota_bytes: QuotaBytes | None = None
    wall_slot_ids: WallSlots | None = None


class SettingsResponse(ContractModel):
    """Expose current global operator settings."""

    retention_days: RetentionDays
    quota_bytes: QuotaBytes
    wall_slot_ids: WallSlots
