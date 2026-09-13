"""Build binary checks from typed Task 11 driver evidence."""

from math import isclose
from typing import Literal

from gods_watching.verification.models import Check

from .task11_models import Task11DriverEvidence

_EMBEDDING_DIMENSION = 512
_UPGRADED_VERSION = 2


def build_checks(
    mode: Literal["appearance", "appearance-stale"],
    evidence: Task11DriverEvidence,
    *,
    cleanup_succeeded: bool,
) -> tuple[Check, ...]:
    """Require real active publication and all selected adversity observables."""
    active = evidence.active
    checks = (
        Check(
            name="active-searchable-before-exit",
            passed=active.active_before_exit and active.unique_result,
            detail=(
                f"appearance={active.appearance_id}; unique={active.unique_result}; "
                f"version={active.representative_version}"
            ),
        ),
        Check(
            name="jpeg-vector-revision-consistent",
            passed=(
                active.jpeg_rgb
                and active.embedding_dimension == _EMBEDDING_DIMENSION
                and isclose(active.embedding_norm, 1.0, abs_tol=1e-3)
                and bool(active.model_revision)
            ),
            detail=(
                f"jpeg_rgb={active.jpeg_rgb}; crop={active.crop_dimensions}; "
                f"embedding={active.embedding_dimension}; norm={active.embedding_norm:.6f}; "
                f"revision={active.model_revision}"
            ),
        ),
        Check(
            name="end-finalized",
            passed=active.ended_after_shutdown,
            detail=f"ended_after_shutdown={active.ended_after_shutdown}",
        ),
    )
    if mode == "appearance":
        return (
            *checks,
            Check(
                name="owned-resource-cleanup",
                passed=cleanup_succeeded,
                detail=f"cleanup_succeeded={cleanup_succeeded}",
            ),
        )
    stale = evidence.stale
    if stale is None:
        return (
            *checks,
            Check(name="stale-evidence-present", passed=False, detail="stale evidence absent"),
            Check(
                name="owned-resource-cleanup",
                passed=cleanup_succeeded,
                detail=f"cleanup_succeeded={cleanup_succeeded}",
            ),
        )
    return (
        *checks,
        Check(
            name="late-completions-rejected",
            passed=(
                stale.old_generation_outcome == "stale_generation"
                and stale.old_generation_row_absent
                and stale.late_version_outcome == "stale_version"
                and stale.committed_version == _UPGRADED_VERSION
                and stale.current_crop_preserved
                and stale.rejected_crop_cleaned
            ),
            detail=(
                f"generation={stale.old_generation_outcome}; "
                f"version={stale.late_version_outcome}; committed={stale.committed_version}; "
                f"crop_preserved={stale.current_crop_preserved}; "
                f"rejected_crop_cleaned={stale.rejected_crop_cleaned}"
            ),
        ),
        Check(
            name="upgrade-and-restart-recovery",
            passed=(
                stale.clipped_to_inside_upgrade
                and stale.old_version_gc_enqueued
                and stale.orphan_removed_on_restart
                and stale.temporary_removed_on_restart
            ),
            detail=(
                f"border_upgrade={stale.clipped_to_inside_upgrade}; "
                f"old_gc={stale.old_version_gc_enqueued}; "
                f"orphan={stale.orphan_removed_on_restart}; "
                f"temporary={stale.temporary_removed_on_restart}"
            ),
        ),
        Check(
            name="owned-resource-cleanup",
            passed=cleanup_succeeded,
            detail=f"cleanup_succeeded={cleanup_succeeded}",
        ),
    )


__all__ = ["build_checks"]
