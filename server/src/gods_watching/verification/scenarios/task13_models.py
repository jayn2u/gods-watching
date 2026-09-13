"""Typed evidence emitted by the Task 13 retention driver."""

from typing import Literal

from gods_watching.verification.models import VerificationModel


class RetentionEvidence(VerificationModel):
    """Record age and quota eviction observed against the real pipeline worker."""

    published_before: int
    age_backdated: int
    age_evicted: int
    age_crops_removed: bool
    quota_bytes: int
    quota_snapshot: int
    quota_evicted: int
    quota_oldest_first: bool
    active_victims: int
    active_victims_not_republished: bool
    managed_bytes_after: int
    threshold_bytes: int
    within_budget_or_full: bool
    search_results_checked: int
    search_crops_readable: bool
    unrelated_paths_untouched: bool
    worker_exit_code: int | None
    open_tracks_ended_on_stop: bool


class CrashRecoveryEvidence(VerificationModel):
    """Record restart reconciliation after the worker process was killed."""

    published_before_kill: int
    killed_exit_code: int | None
    orphan_removed: bool
    temporary_removed: bool
    tombstoned_crop_unlinked: bool
    tombstoned_row_finalized: bool
    referenced_crops_intact: bool
    published_after_restart: int
    publishing_resumed: bool
    worker_exit_code: int | None


class Task13DriverEvidence(VerificationModel):
    """Combine the observations for one Task 13 scenario mode."""

    mode: Literal["retention", "retention-crash"]
    retention: RetentionEvidence | None = None
    crash: CrashRecoveryEvidence | None = None


__all__ = ["CrashRecoveryEvidence", "RetentionEvidence", "Task13DriverEvidence"]
