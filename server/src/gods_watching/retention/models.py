"""Typed retention configuration, accounting, and failure boundaries."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, final, override

from gods_watching.contracts.identifiers import AppearanceId

DEFAULT_RETENTION_DAYS = 7
DEFAULT_QUOTA_BYTES = 100_000_000_000
DEFAULT_CLEANUP_THRESHOLD = 0.95
DEFAULT_MINIMUM_FREE_BYTES = 5 * 1024**3
DEFAULT_SWEEP_INTERVAL_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class RetentionSettings:
    """Represent positive retention values after boundary normalization."""

    retention_days: int
    quota_bytes: int
    cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES
    sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS

    @classmethod
    def from_values(
        cls,
        *,
        retention_days: int | None,
        quota_bytes: int | None,
        cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD,
        minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES,
        sweep_interval_seconds: float = DEFAULT_SWEEP_INTERVAL_SECONDS,
    ) -> "RetentionSettings":
        """Clamp positive settings and preserve safe scheduler boundaries."""
        days = DEFAULT_RETENTION_DAYS if retention_days is None else retention_days
        quota = DEFAULT_QUOTA_BYTES if quota_bytes is None else quota_bytes
        return cls(
            retention_days=max(1, days),
            quota_bytes=max(1, quota),
            cleanup_threshold=min(1.0, max(0.01, cleanup_threshold)),
            minimum_free_bytes=max(0, minimum_free_bytes),
            sweep_interval_seconds=max(0.01, sweep_interval_seconds),
        )


@dataclass(frozen=True, slots=True)
class StorageAccounting:
    """Describe conservative managed storage usage and cleanup triggers."""

    physical_crop_bytes: int
    pending_gc_bytes: int
    relation_bytes: int
    filesystem_free_bytes: int
    quota_bytes: int
    cleanup_threshold: float = DEFAULT_CLEANUP_THRESHOLD
    minimum_free_bytes: int = DEFAULT_MINIMUM_FREE_BYTES

    @property
    def managed_bytes(self) -> int:
        """Include physical crops and relation bloat without double-counting GC files."""
        return self.physical_crop_bytes + self.relation_bytes

    @property
    def threshold_bytes(self) -> int:
        """Return the configured cleanup-start threshold in bytes."""
        return max(1, int(self.quota_bytes * self.cleanup_threshold))

    @property
    def cleanup_required(self) -> bool:
        """Return whether quota cleanup should start before the next writer."""
        return self.managed_bytes >= self.threshold_bytes

    @property
    def filesystem_guard_active(self) -> bool:
        """Return whether the filesystem is below the retention free-space guard."""
        return self.filesystem_free_bytes < self.minimum_free_bytes


class FailurePoint(StrEnum):
    """Name restart boundaries used by deterministic crash tests."""

    BEFORE_TOMBSTONE = "before_tombstone"
    AFTER_TOMBSTONE = "after_tombstone"
    BEFORE_UNLINK = "before_unlink"
    AFTER_UNLINK = "after_unlink"


@final
class InjectedRetentionFailureError(RuntimeError):
    """Represent an intentional termination at a retention recovery boundary."""

    def __init__(self, point: FailurePoint, object_key: str) -> None:
        """Retain the failure boundary and generated object key."""
        self.point = point
        self.object_key = object_key
        super().__init__(point, object_key)

    @override
    def __str__(self) -> str:
        """Render a deterministic crash-injection message."""
        return f"injected retention failure at {self.point.value} for {self.object_key}"


class RetentionClock(Protocol):
    """Provide an injectable UTC clock for deterministic retention decisions."""

    def now(self) -> datetime:
        """Return the current timezone-aware UTC timestamp."""
        ...


class FailureInjector(Protocol):
    """Provide deterministic termination at one named filesystem boundary."""

    def check(self, point: FailurePoint, object_key: str) -> None:
        """Raise when the configured failure point is reached."""
        ...


class QuotaSuppression(Protocol):
    """Consume the existing publisher hook for quota-evicted active tracks."""

    def mark_quota_evicted(self, appearance_id: AppearanceId) -> None:
        """Suppress new representatives until the track's END event."""
        ...


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Report safe startup cleanup within the configured crop root."""

    removed_jpegs: int
    removed_temps: int
    referenced_files: int
    skipped_files: int


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Report one age/quota/GC sweep and its managed-storage observation."""

    age_candidates: int
    quota_candidates: int
    tombstoned: int
    unlinked: int
    gc_failures: int
    suppressed_tracks: int
    managed_bytes_before: int
    managed_bytes_after: int
    relation_bytes_after: int
    physical_crop_bytes_after: int
    pending_gc_bytes_after: int
    filesystem_free_bytes_after: int
    storage_full: bool
    errors: tuple[str, ...] = ()


__all__ = [
    "DEFAULT_CLEANUP_THRESHOLD",
    "DEFAULT_MINIMUM_FREE_BYTES",
    "DEFAULT_QUOTA_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_SWEEP_INTERVAL_SECONDS",
    "FailureInjector",
    "FailurePoint",
    "InjectedRetentionFailure",
    "InjectedRetentionFailureError",
    "QuotaSuppression",
    "ReconciliationReport",
    "RetentionClock",
    "RetentionSettings",
    "StorageAccounting",
    "SweepReport",
]

InjectedRetentionFailure = InjectedRetentionFailureError
