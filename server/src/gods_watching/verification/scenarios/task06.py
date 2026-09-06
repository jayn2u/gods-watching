"""Register the verification harness self-check."""

from gods_watching.verification.models import ImplementedScenario
from gods_watching.verification.registry import parse_scenario_name, register_scenario
from gods_watching.verification.self_check import run_harness_self_check

register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("harness-self-check"),
        runner=run_harness_self_check,
        timeout_seconds=5.0,
    )
)
