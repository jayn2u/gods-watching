"""Scenario registration and planned scenario inventory."""

import importlib
import pkgutil
import re
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Final, final

from .errors import DuplicateScenarioError, InvalidScenarioNameError
from .models import ScenarioDefinition, ScenarioName, UnavailableScenario

_SCENARIO_PATTERN: Final = re.compile(r"^[a-z][a-z0-9-]{0,63}$")

_PLANNED_SCENARIOS: Final[tuple[tuple[str, int], ...]] = (
    ("appearance", 11), ("appearance-stale", 11), ("camera-auth", 12),
    ("camera-auth-denied", 12), ("clip", 8), ("clip-errors", 8),
    ("corrupt-fixture", 3), ("corrupt-model-assets", 2), ("detector", 7),
    ("detector-errors", 7), ("faults", 22), ("fixture-inputs", 3), ("full", 24),
    ("ingest", 10), ("ingest-outage", 10), ("load", 21), ("media", 9),
    ("media-denied", 9), ("model-assets", 2), ("offline", 19),
    ("offline-missing-assets", 19), ("overload", 21), ("packaging", 19),
    ("restart", 22), ("retention", 13), ("retention-crash", 13),
    ("retrieval-negative", 20), ("retrieval-quality", 20), ("search", 14),
    ("search-errors", 14), ("service-outages", 18), ("status", 18),
)


@final
class ScenarioCatalog:
    """Collect per-task registrations while guarding concurrent discovery."""

    def __init__(self) -> None:
        """Create an empty process-local catalog."""
        self._definitions: dict[ScenarioName, ScenarioDefinition] = {}
        self._lock = threading.Lock()

    def register(self, definition: ScenarioDefinition) -> None:
        """Add exactly one definition for a public name."""
        with self._lock:
            if definition.name in self._definitions:
                raise DuplicateScenarioError(name=definition.name)
            self._definitions[definition.name] = definition

    def snapshot(self) -> tuple[ScenarioDefinition, ...]:
        """Return an immutable view for a run-local registry."""
        with self._lock:
            return tuple(self._definitions.values())


_CATALOG: Final = ScenarioCatalog()


def register_scenario(definition: ScenarioDefinition) -> None:
    """Register a definition from a task-owned module under scenarios/."""
    _CATALOG.register(definition)


def _discover_scenario_modules() -> None:
    scenario_path = Path(__file__).parent / "scenarios"
    prefix = "gods_watching.verification.scenarios."
    for module in pkgutil.iter_modules([str(scenario_path)], prefix):
        _ = importlib.import_module(module.name)


def parse_scenario_name(value: str) -> ScenarioName:
    """Parse one untrusted CLI scenario name."""
    if _SCENARIO_PATTERN.fullmatch(value) is None:
        raise InvalidScenarioNameError(value=value)
    return ScenarioName(value)


@final
class ScenarioRegistry:
    """Build an immutable-by-use map, allowing placeholders to be replaced."""

    def __init__(self) -> None:
        """Start a run-local empty registry."""
        self._definitions: dict[ScenarioName, ScenarioDefinition] = {}

    def register(self, definition: ScenarioDefinition) -> None:
        """Register a new name or replace its planned placeholder."""
        existing = self._definitions.get(definition.name)
        if existing is None or (
            existing.owner_task is not None and definition.owner_task is None
        ):
            self._definitions[definition.name] = definition
            return
        raise DuplicateScenarioError(name=definition.name)

    def get(self, name: ScenarioName) -> ScenarioDefinition | None:
        """Return the exact registered scenario definition."""
        return self._definitions.get(name)

    def names(self) -> tuple[ScenarioName, ...]:
        """Return every registered scenario name in stable order."""
        return tuple(sorted(self._definitions))


def build_registry(additional: Iterable[ScenarioDefinition] = ()) -> ScenarioRegistry:
    """Create a fresh registry so tests and concurrent runs share no state."""
    _discover_scenario_modules()
    registry = ScenarioRegistry()
    for name, owner_task in _PLANNED_SCENARIOS:
        registry.register(UnavailableScenario(name=ScenarioName(name), owner_task=owner_task))
    for definition in _CATALOG.snapshot():
        registry.register(definition)
    for definition in additional:
        registry.register(definition)
    return registry
