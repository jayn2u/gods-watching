from dataclasses import dataclass
from pathlib import Path

from gods_watching.verification import ImplementedScenario, ScenarioName, build_registry
from gods_watching.verification.models import RunId
from gods_watching.verification.scenarios.task12_live_runtime import driver_command


@dataclass(frozen=True, slots=True)
class _Context:
    run_id: RunId
    run_root: Path
    runtime_root: Path
    compose_project: str
    allocated_port: int


def test_live_boundary_scenario_is_installed_and_real_runtime() -> None:
    # Given: the installed scenario registry
    registry = build_registry()

    # When: the Task 12 live-boundary scenario is resolved
    scenario = registry.get(ScenarioName("camera-auth-live-boundaries"))

    # Then: it is an executable real-runtime scenario with the required tools
    assert isinstance(scenario, ImplementedScenario)
    assert {"docker", "ffmpeg", "ffprobe", "node", "uv"} <= set(scenario.required_commands)
    assert scenario.timeout_seconds >= 300.0


def test_live_boundary_driver_command_targets_existing_live_runtime_module(tmp_path: Path) -> None:
    # Given: one isolated verification context and an evidence output path
    context = _Context(
        run_id=RunId("run-12e"),
        run_root=tmp_path / "task-12e",
        runtime_root=tmp_path / "task-12e" / "runtime",
        compose_project="gw-task-12e",
        allocated_port=31241,
    )

    # When: the scenario builds its child-process command
    command = driver_command(context, output=context.run_root / "result.json")

    # Then: the command invokes the module that owns the live driver
    assert command[-2:] == ("-m", "gods_watching.verification.scenarios.task12_live_runtime")
