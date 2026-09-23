"""Domain values for the durable global CLIP model transition."""

# ruff: noqa: TC003, D107, TRY003, EM101

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Final, final
from uuid import UUID


class TransitionPhase(StrEnum):
    """Durable phases exposed to the operator and worker."""

    QUEUED = "queued"
    PREPARING = "preparing"
    REINDEXING = "reindexing"
    ACTIVATING = "activating"
    ROLLING_BACK = "rolling_back"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


ACTIVE_PHASES: Final[frozenset[TransitionPhase]] = frozenset(
    {
        TransitionPhase.QUEUED,
        TransitionPhase.PREPARING,
        TransitionPhase.REINDEXING,
        TransitionPhase.ACTIVATING,
        TransitionPhase.ROLLING_BACK,
    }
)


class TransitionError(RuntimeError):
    """Base class for safe model transition failures."""

    code: str = "model_transition_failed"
    message: str

    def __init__(self, message: str, *, code: str | None = None) -> None:
        self.message = message
        if code is not None:
            self.code = code
        super().__init__(message)


@final
class ModelSelectionConflictError(TransitionError):
    """A durable transition is already queued or running."""

    code: str = "model_transition_conflict"


@final
class ModelNotPreparedError(TransitionError):
    """The selected immutable model package is unavailable locally."""

    code: str = "model_not_prepared"


@final
class ModelTransitionNotFoundError(TransitionError):
    """A requested transition id is not present in durable state."""

    code: str = "model_transition_not_found"


@final
class MaintenanceModeError(TransitionError):
    """Search or publication was attempted while a switch owns the lock."""

    code: str = "model_transition_maintenance"


@final
class TransitionRecoveryError(TransitionError):
    """Runtime and durable active identity could not be reconciled."""

    code: str = "model_transition_recovery_failed"


@dataclass(frozen=True, slots=True)
class PreparedModelStatus:
    """Describe whether one immutable package is usable without downloading."""

    prepared: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        """Reject contradictory preparation status values."""
        if self.prepared and self.reason is not None:
            raise ValueError("prepared model status cannot include a reason")


@dataclass(frozen=True, slots=True)
class TransitionState:
    """Safe durable transition state suitable for API serialization."""

    id: UUID
    source_model_id: str
    target_model_id: str
    phase: TransitionPhase
    processed: int
    total: int
    skipped: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    source_model_revision: str | None = None
    target_model_revision: str | None = None
    source_dimension: int | None = None
    target_dimension: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @property
    def active(self) -> bool:
        """Return whether the job still owns maintenance mode."""
        return self.phase in ACTIVE_PHASES


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Outcome of one worker run, including whether target became active."""

    state: TransitionState
    activated: bool


class CropSkipReason(StrEnum):
    """Only crop-local failures that may be skipped during staging."""

    MISSING = "missing_crop"
    UNDECODABLE = "undecodable_crop"


__all__ = [
    "ACTIVE_PHASES",
    "CropSkipReason",
    "MaintenanceModeError",
    "ModelNotPreparedError",
    "ModelSelectionConflictError",
    "ModelTransitionNotFoundError",
    "PreparedModelStatus",
    "TransitionError",
    "TransitionPhase",
    "TransitionRecoveryError",
    "TransitionResult",
    "TransitionState",
]
