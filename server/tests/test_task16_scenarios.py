from gods_watching.verification.models import ImplementedScenario
from gods_watching.verification.registry import build_registry, parse_scenario_name
from gods_watching.verification.scenarios.task16_checks import build_checks
from gods_watching.verification.scenarios.task16_models import (
    PlaywrightPhase,
    SearchUiEvidence,
    Task16DriverEvidence,
)


def _phase(**changes: int) -> PlaywrightPhase:
    values = {"exit_code": 0, "expected": 2, "unexpected": 0, "skipped": 1, "flaky": 0}
    values.update(changes)
    return PlaywrightPhase.model_validate(values)


def _evidence(**changes: object) -> Task16DriverEvidence:
    values: dict[str, object] = {
        "normal": _phase(),
        "outage": _phase(expected=1, skipped=2),
        "outage_marker_seen": True,
        "inference_recovered": True,
    }
    values.update(changes)
    return Task16DriverEvidence(ui=SearchUiEvidence.model_validate(values))


def test_search_ui_scenario_is_registered_with_bounded_dependencies() -> None:
    # Given / When
    scenario = build_registry().get(parse_scenario_name("search-ui"))

    # Then
    assert isinstance(scenario, ImplementedScenario)
    assert scenario.required_commands == ("docker", "ffmpeg", "ffprobe", "uv", "pnpm")
    assert scenario.timeout_seconds == 1_200.0


def test_search_ui_checks_pass_with_complete_real_runs() -> None:
    # Given / When
    checks = build_checks(web_build_succeeded=True, evidence=_evidence(), cleanup_succeeded=True)

    # Then
    assert [check.name for check in checks] == [
        "web-build-succeeded",
        "real-api-search-flows-pass",
        "inference-outage-recovers-in-ui",
        "owned-resource-cleanup",
    ]
    assert all(check.passed for check in checks)


def test_search_ui_checks_fail_when_a_flow_is_skipped_or_recovery_is_missing() -> None:
    # Given: one normal flow skipped and inference never came back
    evidence = _evidence(normal=_phase(expected=1, skipped=2), inference_recovered=False)

    # When
    checks = {
        check.name: check
        for check in build_checks(
            web_build_succeeded=True, evidence=evidence, cleanup_succeeded=True
        )
    }

    # Then
    assert not checks["real-api-search-flows-pass"].passed
    assert not checks["inference-outage-recovers-in-ui"].passed


def test_missing_ui_evidence_fails_behavior_checks() -> None:
    # Given / When
    checks = build_checks(
        web_build_succeeded=True, evidence=Task16DriverEvidence(), cleanup_succeeded=True
    )

    # Then
    assert [check.name for check in checks if check.passed] == [
        "web-build-succeeded",
        "owned-resource-cleanup",
    ]
