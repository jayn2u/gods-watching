from gods_watching.verification.models import ImplementedScenario
from gods_watching.verification.registry import build_registry, parse_scenario_name


def test_task11_scenarios_replace_planned_placeholders() -> None:
    # Given: the auto-discovered verification catalog.
    registry = build_registry()

    # When: both Task 11 public scenario names are resolved.
    appearance = registry.get(parse_scenario_name("appearance"))
    stale = registry.get(parse_scenario_name("appearance-stale"))

    # Then: each name is executable with bounded external dependencies.
    assert isinstance(appearance, ImplementedScenario)
    assert isinstance(stale, ImplementedScenario)
    assert appearance.required_commands == ("docker", "ffmpeg", "ffprobe", "uv")
    assert stale.required_commands == ("docker", "ffmpeg", "ffprobe", "uv")
    assert appearance.timeout_seconds == 300.0
    assert stale.timeout_seconds == 300.0
