"""Stable verification registration and execution interface."""

from .context import ScenarioContext
from .errors import DuplicateScenarioError, EvidencePathError, InvalidScenarioNameError
from .executor import execute_scenario
from .models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    RunResult,
    ScenarioContextProtocol,
    ScenarioDefinition,
    ScenarioName,
    ScenarioReport,
    UnavailableScenario,
)
from .registry import ScenarioRegistry, build_registry, parse_scenario_name, register_scenario

__all__ = [
    "Check",
    "DuplicateScenarioError",
    "EvidenceKind",
    "EvidencePathError",
    "ImplementedScenario",
    "InvalidScenarioNameError",
    "RunResult",
    "ScenarioContext",
    "ScenarioContextProtocol",
    "ScenarioDefinition",
    "ScenarioName",
    "ScenarioRegistry",
    "ScenarioReport",
    "UnavailableScenario",
    "build_registry",
    "execute_scenario",
    "parse_scenario_name",
    "register_scenario",
]
