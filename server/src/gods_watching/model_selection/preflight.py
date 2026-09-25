"""Measured duration estimate for a manual model transition."""

# ruff: noqa: TC001, TC002, TC003, PLC0415, TRY003, EM101

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.model_selection.registry import ClipModelPackage
from gods_watching.storage import CropObjectStore
from gods_watching.storage.models import Appearance

MAX_SWITCH_SECONDS: Final[int] = 900
MIN_SAMPLE_COUNT: Final[int] = 16
MEASUREMENT_MAX_AGE: Final = timedelta(hours=24)


@dataclass(frozen=True, slots=True)
class SwitchPreflight:
    """Current retained-corpus estimate for one target package."""

    target_model_id: str
    retained_count: int
    estimated_missing_count: int
    measured_crops_per_second: float | None
    estimated_seconds: float | None
    max_seconds: int
    eligible: bool
    reason: str | None


def estimate_switch(
    retained_count: int,
    measured_crops_per_second: float | None,
    estimated_missing_count: int,
    *,
    target_model_id: str,
) -> SwitchPreflight:
    """Bound a switch by the measured rate for available retained crops."""
    if retained_count < 0 or not 0 <= estimated_missing_count <= retained_count:
        msg = "retained and missing counts are inconsistent"
        raise ValueError(msg)
    valid_rate = (
        measured_crops_per_second is not None
        and math.isfinite(measured_crops_per_second)
        and measured_crops_per_second > 0
    )

    if not valid_rate:
        return SwitchPreflight(
            target_model_id=target_model_id,
            retained_count=retained_count,
            estimated_missing_count=estimated_missing_count,
            measured_crops_per_second=None,
            estimated_seconds=None,
            max_seconds=MAX_SWITCH_SECONDS,
            eligible=False,
            reason="throughput_unavailable",
        )
    if measured_crops_per_second is None:
        raise ValueError("missing measured rate")
    estimated_seconds = (retained_count - estimated_missing_count) / measured_crops_per_second
    eligible = math.isfinite(estimated_seconds) and estimated_seconds <= MAX_SWITCH_SECONDS
    return SwitchPreflight(
        target_model_id=target_model_id,
        retained_count=retained_count,
        estimated_missing_count=estimated_missing_count,
        measured_crops_per_second=measured_crops_per_second,
        estimated_seconds=estimated_seconds if math.isfinite(estimated_seconds) else None,
        max_seconds=MAX_SWITCH_SECONDS,
        eligible=eligible,
        reason=None if eligible else "estimate_exceeds_limit",
    )


def measurement_path(assets_root: Path, package: ClipModelPackage) -> Path:
    """Name a record by immutable package identity without trusting model IDs as paths."""
    digest = hashlib.sha256(f"{package.model_id}\0{package.revision}".encode()).hexdigest()
    return assets_root / "switch-throughput" / f"{digest}.json"


def measured_rate(
    assets_root: Path, package: ClipModelPackage, *, now: datetime | None = None
) -> float | None:
    """Accept only a fresh exact-package measurement on the deployment GPU."""
    try:
        manifest = json.loads((assets_root / "prepared-manifest.json").read_text())
        record = json.loads(measurement_path(assets_root, package).read_text())
        device = manifest["cuda_device"]
        at = datetime.fromisoformat(record["measured_at"])
        elapsed = (now or datetime.now(UTC)) - at
        count = record["sample_count"]
        seconds = record["measured_seconds"]
        if (
            record["model_id"] != package.model_id
            or record["revision"] != package.revision
            or record["dimension"] != package.dimension
            or record["device"] != device
            or not device
            or record["detector_resident"] is not True
            or at.tzinfo is None
            or not timedelta(0) <= elapsed <= MEASUREMENT_MAX_AGE
            or type(count) is not int
            or count < MIN_SAMPLE_COUNT
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
        ):
            return None
        return count / seconds
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


async def scan_retained(
    session: AsyncSession, crop_store: CropObjectStore
) -> tuple[int, int]:
    """Count retained appearances and crops that staging will skip."""
    keys = (await session.scalars(
        select(Appearance.crop_object_key).where(Appearance.tombstoned_at.is_(None))
    )).all()
    missing = 0
    for key in keys:
        try:
            from io import BytesIO

            with Image.open(BytesIO(crop_store.read(key))) as image:
                image.verify()
        except (OSError, ValueError, UnidentifiedImageError):
            missing += 1
    return len(keys), missing
