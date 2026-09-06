"""Typed failures raised at verification trust boundaries."""

from dataclasses import dataclass
from pathlib import Path
from typing import override


@dataclass(frozen=True, slots=True)
class InvalidScenarioNameError(Exception):
    """Reject a scenario name outside the stable CLI grammar."""

    value: str

    @override
    def __str__(self) -> str:
        """Return a value-independent boundary message."""
        return "scenario must match ^[a-z][a-z0-9-]{0,63}$"


@dataclass(frozen=True, slots=True)
class EvidencePathError(Exception):
    """Reject unsafe or stale evidence destinations."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Exclude the caller-controlled path from the message."""
        return f"evidence directory rejected: {self.reason}"


@dataclass(frozen=True, slots=True)
class DuplicateScenarioError(Exception):
    """Reject ambiguous scenario dispatch."""

    name: str

    @override
    def __str__(self) -> str:
        """Identify the conflicting public scenario name."""
        return f"scenario registered more than once: {self.name}"
