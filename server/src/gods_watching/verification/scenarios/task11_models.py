"""Typed evidence emitted by the Task 11 publication driver."""

from typing import Literal

from gods_watching.verification.models import VerificationModel


class ActivePublicationEvidence(VerificationModel):
    """Record the active row and crop observed before worker shutdown."""

    appearance_id: str
    active_before_exit: bool
    unique_result: bool
    representative_version: int
    jpeg_rgb: bool
    crop_dimensions: tuple[int, int]
    embedding_dimension: int
    embedding_norm: float
    model_revision: str
    ended_after_shutdown: bool


class StaleRecoveryEvidence(VerificationModel):
    """Record late-completion rejection, replacement GC, and restart recovery."""

    old_generation_outcome: str
    old_generation_row_absent: bool
    late_version_outcome: str
    committed_version: int
    current_crop_preserved: bool
    rejected_crop_cleaned: bool
    old_version_gc_enqueued: bool
    orphan_removed_on_restart: bool
    temporary_removed_on_restart: bool
    clipped_to_inside_upgrade: bool


class Task11DriverEvidence(VerificationModel):
    """Combine real ingest publication with optional stale-path observations."""

    mode: Literal["appearance", "appearance-stale"]
    active: ActivePublicationEvidence
    stale: StaleRecoveryEvidence | None = None


__all__ = [
    "ActivePublicationEvidence",
    "StaleRecoveryEvidence",
    "Task11DriverEvidence",
]
