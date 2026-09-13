"""Build binary checks from typed Task 13 driver evidence."""

from typing import Literal

from gods_watching.verification.models import Check

from .task13_models import CrashRecoveryEvidence, RetentionEvidence, Task13DriverEvidence

_MISSING = "driver produced no observations for this mode"


def build_checks(
    mode: Literal["retention", "retention-crash"],
    evidence: Task13DriverEvidence,
    *,
    cleanup_succeeded: bool,
) -> tuple[Check, ...]:
    """Require every retention or crash-recovery observable plus exact cleanup."""
    behavior = (
        _retention_checks(evidence.retention)
        if mode == "retention"
        else _crash_checks(evidence.crash)
    )
    return (
        *behavior,
        Check(
            name="owned-resource-cleanup",
            passed=cleanup_succeeded,
            detail=f"cleanup_succeeded={cleanup_succeeded}",
        ),
    )


def _retention_checks(retention: RetentionEvidence | None) -> tuple[Check, ...]:
    names = (
        "age-eviction-reclaims-crops",
        "quota-evicts-oldest-first",
        "active-quota-victims-stay-suppressed",
        "managed-bytes-within-budget",
        "no-broken-search-references",
        "unrelated-paths-untouched",
        "graceful-worker-stop",
    )
    if retention is None:
        return tuple(Check(name=name, passed=False, detail=_MISSING) for name in names)
    observed = (
        (
            retention.age_evicted > 0
            and retention.age_evicted == retention.age_backdated
            and retention.age_crops_removed,
            (
                f"backdated={retention.age_backdated}; evicted={retention.age_evicted}; "
                f"crops_removed={retention.age_crops_removed}"
            ),
        ),
        (
            retention.quota_evicted > 0 and retention.quota_oldest_first,
            (
                f"quota_bytes={retention.quota_bytes}; snapshot={retention.quota_snapshot}; "
                f"evicted={retention.quota_evicted}; oldest_first={retention.quota_oldest_first}"
            ),
        ),
        (
            retention.active_victims_not_republished,
            (
                f"active_victims={retention.active_victims}; "
                f"not_republished={retention.active_victims_not_republished}"
            ),
        ),
        (
            retention.within_budget_or_full,
            (
                f"managed_after={retention.managed_bytes_after}; "
                f"threshold={retention.threshold_bytes}; quota={retention.quota_bytes}"
            ),
        ),
        (
            retention.search_crops_readable,
            (
                f"results_checked={retention.search_results_checked}; "
                f"readable={retention.search_crops_readable}"
            ),
        ),
        (
            retention.unrelated_paths_untouched,
            f"untouched={retention.unrelated_paths_untouched}",
        ),
        (
            retention.worker_exit_code == 0 and retention.open_tracks_ended_on_stop,
            (
                f"exit_code={retention.worker_exit_code}; "
                f"open_tracks_ended={retention.open_tracks_ended_on_stop}"
            ),
        ),
    )
    return tuple(
        Check(name=name, passed=passed, detail=detail)
        for name, (passed, detail) in zip(names, observed, strict=True)
    )


def _crash_checks(crash: CrashRecoveryEvidence | None) -> tuple[Check, ...]:
    names = (
        "orphan-and-temporary-reconciled",
        "interrupted-gc-replayed",
        "referenced-crops-intact",
        "publishing-resumed-after-restart",
        "graceful-worker-stop",
    )
    if crash is None:
        return tuple(Check(name=name, passed=False, detail=_MISSING) for name in names)
    observed = (
        (
            crash.orphan_removed and crash.temporary_removed,
            f"orphan_removed={crash.orphan_removed}; temporary_removed={crash.temporary_removed}",
        ),
        (
            crash.tombstoned_crop_unlinked and crash.tombstoned_row_finalized,
            (
                f"crop_unlinked={crash.tombstoned_crop_unlinked}; "
                f"row_finalized={crash.tombstoned_row_finalized}"
            ),
        ),
        (
            crash.referenced_crops_intact,
            f"referenced_crops_intact={crash.referenced_crops_intact}",
        ),
        (
            crash.publishing_resumed,
            (
                f"before_kill={crash.published_before_kill}; killed_exit={crash.killed_exit_code}; "
                f"after_restart={crash.published_after_restart}"
            ),
        ),
        (
            crash.worker_exit_code == 0,
            f"exit_code={crash.worker_exit_code}",
        ),
    )
    return tuple(
        Check(name=name, passed=passed, detail=detail)
        for name, (passed, detail) in zip(names, observed, strict=True)
    )


__all__ = ["build_checks"]
