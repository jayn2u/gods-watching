"""Measured duration estimate for a manual model transition."""

# ruff: noqa: TC001, TC002, TRY003, EM101

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
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
RUNTIME_SOURCE_FILES: Final = (
    "preflight.py",
    "service.py",
    "rehearsal.py",
    "repository.py",
    "coordinator.py",
    "transition_observer.py",
    "rehearsal_stack.py",
)


def runtime_code_sha256() -> str | None:
    """Fingerprint installed model-transition runtime sources in a fixed order."""
    digest = hashlib.sha256()
    try:
        for name in RUNTIME_SOURCE_FILES:
            data = (Path(__file__).parent / name).read_bytes()
            digest.update(name.encode("ascii") + b"\0" + len(data).to_bytes(8, "big") + data)
    except OSError:
        return None
    return digest.hexdigest()


async def retained_corpus_sha256(session: AsyncSession, crop_store: CropObjectStore) -> str | None:
    """Bind proof to the ordered live retained rows and their exact crop bytes."""
    digest = hashlib.sha256()
    try:
        rows = await session.stream(
            select(Appearance.id, Appearance.crop_object_key)
            .where(Appearance.tombstoned_at.is_(None))
            .order_by(Appearance.id)
        )
        async for batch in rows.partitions(128):
            for appearance_id, key in batch:
                payload = await asyncio.to_thread(crop_store.read, key)
                identity = f"{appearance_id}\0{key}".encode()
                digest.update(len(identity).to_bytes(8, "big") + identity)
                digest.update(len(payload).to_bytes(8, "big") + payload)
    except (OSError, ValueError, TypeError):
        return None
    return digest.hexdigest()


async def scan_retained_snapshot(
    session: AsyncSession, crop_store: CropObjectStore
) -> tuple[int, int, str | None]:
    """Count, validate, and fingerprint the same ordered retained corpus pass."""
    digest = hashlib.sha256()
    count = missing = 0
    try:
        rows = await session.stream(
            select(Appearance.id, Appearance.crop_object_key)
            .where(Appearance.tombstoned_at.is_(None))
            .order_by(Appearance.id)
        )
        async for batch in rows.partitions(128):
            for appearance_id, key in batch:
                count += 1
                identity = f"{appearance_id}\0{key}".encode()
                digest.update(len(identity).to_bytes(8, "big") + identity)
                try:
                    payload = await asyncio.to_thread(crop_store.read, key)
                except (FileNotFoundError, ValueError):
                    missing += 1
                    digest.update(b"missing\0")
                    continue
                digest.update(len(payload).to_bytes(8, "big") + payload)
                try:
                    with Image.open(BytesIO(payload)) as image:
                        image.load()
                except (OSError, ValueError, UnidentifiedImageError):
                    missing += 1
    except (OSError, ValueError, TypeError):
        return count, missing, None
    return count, missing, digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SwitchPreflight:
    """Current retained-corpus estimate for one target package."""

    target_model_id: str
    retained_count: int
    estimated_missing_count: int
    measured_crops_per_second: float | None
    measured_fixed_seconds: float | None
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
    measured_fixed_seconds: float | None = None,
) -> SwitchPreflight:
    """Bound a switch by full-path crop rate and measured fixed overhead."""
    if retained_count < 0 or not 0 <= estimated_missing_count <= retained_count:
        msg = "retained and missing counts are inconsistent"
        raise ValueError(msg)
    valid_rate = (
        measured_crops_per_second is not None
        and math.isfinite(measured_crops_per_second)
        and measured_crops_per_second > 0
        and measured_fixed_seconds is not None
        and math.isfinite(measured_fixed_seconds)
        and measured_fixed_seconds >= 0
    )

    if not valid_rate:
        return SwitchPreflight(
            target_model_id=target_model_id,
            retained_count=retained_count,
            estimated_missing_count=estimated_missing_count,
            measured_crops_per_second=None,
            measured_fixed_seconds=None,
            estimated_seconds=None,
            max_seconds=MAX_SWITCH_SECONDS,
            eligible=False,
            reason="throughput_unavailable",
        )
    if measured_crops_per_second is None:
        raise ValueError("missing measured rate")
    if measured_fixed_seconds is None:
        raise ValueError("missing measured overhead")
    estimated_seconds = (
        measured_fixed_seconds
        + (retained_count - estimated_missing_count) / measured_crops_per_second
    )
    eligible = math.isfinite(estimated_seconds) and estimated_seconds <= MAX_SWITCH_SECONDS
    return SwitchPreflight(
        target_model_id=target_model_id,
        retained_count=retained_count,
        estimated_missing_count=estimated_missing_count,
        measured_crops_per_second=measured_crops_per_second,
        measured_fixed_seconds=measured_fixed_seconds,
        estimated_seconds=estimated_seconds if math.isfinite(estimated_seconds) else None,
        max_seconds=MAX_SWITCH_SECONDS,
        eligible=eligible,
        reason=None if eligible else "estimate_exceeds_limit",
    )


def measurement_path(assets_root: Path, package: ClipModelPackage) -> Path:
    """Name a diagnostic embedding-only benchmark by immutable identity."""
    digest = hashlib.sha256(f"{package.model_id}\0{package.revision}".encode()).hexdigest()
    return assets_root / "switch-throughput" / f"{digest}.json"


def rehearsal_path(assets_root: Path, package: ClipModelPackage) -> Path:
    """Name an independent full-transition rehearsal proof."""
    digest = hashlib.sha256(f"{package.model_id}\0{package.revision}".encode()).hexdigest()
    return assets_root / "switch-rehearsal" / f"{digest}.json"


def measured_rehearsal(
    assets_root: Path,
    package: ClipModelPackage,
    *,
    corpus_sha256: str | None = None,
    now: datetime | None = None,
) -> tuple[float, float] | None:
    """Accept only a fresh full-path proof on the exact deployment GPU UUID."""
    try:
        manifest = json.loads((assets_root / "prepared-manifest.json").read_text())
        record = json.loads(rehearsal_path(assets_root, package).read_text())
        device = manifest["cuda_device"]
        device_uuid = manifest["cuda_device_uuid"]
        at = datetime.fromisoformat(record["measured_at"])
        elapsed = (now or datetime.now(UTC)) - at
        count = record["sample_count"]
        seconds = record["measured_seconds"]
        fixed_seconds = record["measured_fixed_seconds"]
        if (
            record["kind"] != "full_transition_rehearsal_v1"
            or record["model_id"] != package.model_id
            or record["revision"] != package.revision
            or record["dimension"] != package.dimension
            or record["device"] != device
            or record["device_uuid"] != device_uuid
            or not device
            or not isinstance(device_uuid, str)
            or not device_uuid.startswith("GPU-")
            or record["detector_resident"] is not True
            or record["triton_rpc_measured"] is not True
            or record["database_staging_measured"] is not True
            or record["activation_measured"] is not True
            or record["pipeline_restart_measured"] is not True
            or not corpus_sha256
            or record.get("retained_corpus_sha256") != corpus_sha256
            or not runtime_code_sha256()
            or record.get("runtime_code_sha256") != runtime_code_sha256()
            or at.tzinfo is None
            or not timedelta(0) <= elapsed <= MEASUREMENT_MAX_AGE
            or type(count) is not int
            or count < MIN_SAMPLE_COUNT
            or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)
            or seconds <= 0
            or not isinstance(fixed_seconds, (int, float))
            or not math.isfinite(fixed_seconds)
            or fixed_seconds < 0
        ):
            return None
        return count / seconds, float(fixed_seconds)
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        return None


async def scan_retained(session: AsyncSession, crop_store: CropObjectStore) -> tuple[int, int]:
    """Count retained appearances and crops that staging will skip."""
    rows = await session.stream_scalars(
        select(Appearance.crop_object_key).where(Appearance.tombstoned_at.is_(None))
    )
    retained = 0
    missing = 0
    async for keys in rows.partitions(128):
        retained += len(keys)
        missing += await asyncio.to_thread(_count_missing, crop_store, keys)
    return retained, missing


def _count_missing(crop_store: CropObjectStore, keys: list[str]) -> int:
    """Decode a bounded crop batch outside the API event loop."""
    missing = 0
    for key in keys:
        try:
            payload = crop_store.read(key)
        except (FileNotFoundError, ValueError):
            missing += 1
            continue
        # Other object-store I/O errors abort apply, as they do during staging.
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
        except (OSError, ValueError, UnidentifiedImageError):
            missing += 1
    return missing
