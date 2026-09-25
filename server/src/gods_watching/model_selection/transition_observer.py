"""Monotonic observations from a production model transition."""

# ruff: noqa: D102

from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gods_watching.model_selection.registry import ClipModelPackage

PHASES = (
    "pipeline_stop", "runtime_switch", "stage_population", "crop_embedding",
    "activation", "pipeline_restart",
)


@dataclass(frozen=True, slots=True)
class TransitionMeasurement:
    """Observed timing and identities for one transition attempt."""
    source_identity: tuple[str, str, int]
    target_identity: tuple[str, str, int]
    phase_seconds: dict[str, float]
    phase_spans: dict[str, tuple[float, float]]
    crop_seconds: tuple[float, ...]
    completed_crops: int
    complete: bool

    @property
    def measured_fixed_seconds(self) -> float | None:
        if not self.complete:
            return None
        return sum(self.phase_seconds[phase] for phase in PHASES if phase != "crop_embedding")


@dataclass(slots=True)
class TransitionObserver:
    """Collect only successfully completed phase and committed crop events."""

    source_identity: tuple[str, str, int] | None = None
    target_identity: tuple[str, str, int] | None = None
    phase_seconds: dict[str, float] = field(default_factory=dict)
    phase_spans: dict[str, tuple[float, float]] = field(default_factory=dict)
    crop_seconds: list[float] = field(default_factory=list)
    complete: bool = False
    fresh: bool = False
    _starts: dict[str, float] = field(default_factory=dict)

    def clear(self) -> None:
        self.source_identity = None
        self.target_identity = None
        self.phase_seconds.clear()
        self.phase_spans.clear()
        self.crop_seconds.clear()
        self._starts.clear()
        self.complete = False
        self.fresh = False

    def begin(
        self, source: ClipModelPackage, target: ClipModelPackage, *, fresh: bool = True
    ) -> None:
        self.clear()
        self.source_identity = (source.model_id, source.revision, source.dimension)
        self.target_identity = (target.model_id, target.revision, target.dimension)
        self.fresh = fresh

    def start(self, phase: str) -> None:
        self._starts[phase] = monotonic()

    def end(self, phase: str) -> None:
        started_at = self._starts.pop(phase)
        ended_at = monotonic()
        self.phase_spans[phase] = (started_at, ended_at)
        self.phase_seconds[phase] = ended_at - started_at

    def committed_crop(self, embedding_seconds: float) -> None:
        self.crop_seconds.append(embedding_seconds)

    def finish(self) -> None:
        self.complete = self.fresh and all(phase in self.phase_seconds for phase in PHASES)

    @property
    def measurement(self) -> TransitionMeasurement | None:
        if self.source_identity is None or self.target_identity is None:
            return None
        return TransitionMeasurement(
            source_identity=self.source_identity,
            target_identity=self.target_identity,
            phase_seconds=dict(self.phase_seconds),
            phase_spans=dict(self.phase_spans),
            crop_seconds=tuple(self.crop_seconds),
            completed_crops=len(self.crop_seconds),
            complete=self.complete,
        )
