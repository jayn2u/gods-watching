from gods_watching.verification.models import ImplementedScenario
from gods_watching.verification.registry import build_registry, parse_scenario_name
from gods_watching.verification.scenarios.task14_checks import build_checks
from gods_watching.verification.scenarios.task14_models import (
    SearchErrorsEvidence,
    SearchEvidence,
    Task14DriverEvidence,
)


def _search(**changes: object) -> SearchEvidence:
    values: dict[str, object] = {
        "unauthenticated_status": 401,
        "cross_origin_status": 403,
        "browse_results": 6,
        "browse_camera_filter_only": True,
        "browse_time_filter_only": True,
        "text_status": 200,
        "text_results": 5,
        "text_ranked_desc": True,
        "text_stable": True,
        "similar_status": 200,
        "similar_results": 4,
        "similar_excludes_seed": True,
        "similar_ranked_desc": True,
        "similar_stable": True,
        "detail_status": 200,
        "crop_status": 200,
        "crop_jpeg": True,
        "crop_no_store": True,
    }
    values.update(changes)
    return SearchEvidence.model_validate(values)


def _errors(**changes: object) -> SearchErrorsEvidence:
    values: dict[str, object] = {
        "blank_text_status": 422,
        "unknown_camera_status": 422,
        "expired_seed_status": 404,
        "expired_detail_status": 404,
        "expired_crop_status": 404,
        "outage_text_status": 503,
        "outage_browse_status": 200,
        "outage_similar_status": 200,
    }
    values.update(changes)
    return SearchErrorsEvidence.model_validate(values)


def test_task14_scenarios_replace_planned_placeholders() -> None:
    # Given: the auto-discovered verification catalog
    registry = build_registry()

    # When: both Task 14 public scenario names are resolved
    search = registry.get(parse_scenario_name("search"))
    errors = registry.get(parse_scenario_name("search-errors"))

    # Then: each is executable with bounded external dependencies
    for scenario in (search, errors):
        assert isinstance(scenario, ImplementedScenario)
        assert scenario.required_commands == ("docker", "ffmpeg", "ffprobe", "uv")
        assert scenario.timeout_seconds == 420.0


def test_search_checks_pass_with_complete_real_observations() -> None:
    # Given
    evidence = Task14DriverEvidence(mode="search", search=_search())

    # When
    checks = build_checks("search", evidence, cleanup_succeeded=True)

    # Then
    assert [check.name for check in checks] == [
        "authorization-required",
        "cross-origin-search-rejected",
        "filtered-browse-eligible-only",
        "text-search-ranked-and-stable",
        "similar-search-ranked-stable-excludes-seed",
        "detail-and-private-crop",
        "owned-resource-cleanup",
    ]
    assert all(check.passed for check in checks)


def test_search_checks_fail_on_unstable_text_ranking_or_empty_results() -> None:
    # Given: repeated text queries changed order, and similar search returned nothing
    evidence = Task14DriverEvidence(
        mode="search", search=_search(text_stable=False, similar_results=0)
    )

    # When
    checks = {c.name: c for c in build_checks("search", evidence, cleanup_succeeded=True)}

    # Then
    assert not checks["text-search-ranked-and-stable"].passed
    assert not checks["similar-search-ranked-stable-excludes-seed"].passed


def test_search_errors_checks_require_outage_isolation() -> None:
    # Given: an inference outage also broke vector-only browse
    evidence = Task14DriverEvidence(mode="search-errors", errors=_errors(outage_browse_status=503))

    # When
    checks = {c.name: c for c in build_checks("search-errors", evidence, cleanup_succeeded=True)}

    # Then
    assert list(checks) == [
        "invalid-input-rejected",
        "expired-seed-and-crop-not-served",
        "text-inference-outage-reported",
        "vector-search-survives-inference-outage",
        "owned-resource-cleanup",
    ]
    assert checks["text-inference-outage-reported"].passed
    assert not checks["vector-search-survives-inference-outage"].passed


def test_missing_mode_evidence_fails_every_behavior_check() -> None:
    # Given
    evidence = Task14DriverEvidence(mode="search-errors")

    # When
    checks = build_checks("search-errors", evidence, cleanup_succeeded=True)

    # Then
    assert [check.name for check in checks if check.passed] == ["owned-resource-cleanup"]
