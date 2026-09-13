from gods_watching.verification.models import ImplementedScenario
from gods_watching.verification.registry import build_registry, parse_scenario_name
from gods_watching.verification.scenarios.task13_checks import build_checks
from gods_watching.verification.scenarios.task13_models import (
    CrashRecoveryEvidence,
    RetentionEvidence,
    Task13DriverEvidence,
)


def _retention(**changes: object) -> RetentionEvidence:
    values: dict[str, object] = {
        "published_before": 12,
        "age_backdated": 3,
        "age_evicted": 3,
        "age_crops_removed": True,
        "quota_bytes": 40_000,
        "quota_snapshot": 9,
        "quota_evicted": 5,
        "quota_oldest_first": True,
        "active_victims": 1,
        "active_victims_not_republished": True,
        "managed_bytes_after": 30_000,
        "threshold_bytes": 38_000,
        "within_budget_or_full": True,
        "search_results_checked": 4,
        "search_crops_readable": True,
        "unrelated_paths_untouched": True,
        "worker_exit_code": 0,
        "open_tracks_ended_on_stop": True,
    }
    values.update(changes)
    return RetentionEvidence.model_validate(values)


def _crash(**changes: object) -> CrashRecoveryEvidence:
    values: dict[str, object] = {
        "published_before_kill": 6,
        "killed_exit_code": -9,
        "orphan_removed": True,
        "temporary_removed": True,
        "tombstoned_crop_unlinked": True,
        "tombstoned_row_finalized": True,
        "referenced_crops_intact": True,
        "published_after_restart": 9,
        "publishing_resumed": True,
        "worker_exit_code": 0,
    }
    values.update(changes)
    return CrashRecoveryEvidence.model_validate(values)


def test_task13_scenarios_replace_planned_placeholders() -> None:
    # Given: the auto-discovered verification catalog
    registry = build_registry()

    # When: both Task 13 public scenario names are resolved
    retention = registry.get(parse_scenario_name("retention"))
    crash = registry.get(parse_scenario_name("retention-crash"))

    # Then: each is executable with bounded external dependencies
    for scenario in (retention, crash):
        assert isinstance(scenario, ImplementedScenario)
        assert scenario.required_commands == ("docker", "ffmpeg", "ffprobe", "uv")
        assert scenario.timeout_seconds == 420.0


def test_retention_checks_pass_only_with_complete_real_observations() -> None:
    # Given: retention evidence where every observable held
    evidence = Task13DriverEvidence(mode="retention", retention=_retention())

    # When
    checks = build_checks("retention", evidence, cleanup_succeeded=True)

    # Then
    assert [check.name for check in checks] == [
        "age-eviction-reclaims-crops",
        "quota-evicts-oldest-first",
        "active-quota-victims-stay-suppressed",
        "managed-bytes-within-budget",
        "no-broken-search-references",
        "unrelated-paths-untouched",
        "graceful-worker-stop",
        "owned-resource-cleanup",
    ]
    assert all(check.passed for check in checks)


def test_retention_checks_fail_on_out_of_order_quota_eviction() -> None:
    # Given: quota eviction removed a newer appearance before an older one
    evidence = Task13DriverEvidence(
        mode="retention", retention=_retention(quota_oldest_first=False)
    )

    # When
    checks = {
        check.name: check for check in build_checks("retention", evidence, cleanup_succeeded=True)
    }

    # Then
    assert not checks["quota-evicts-oldest-first"].passed


def test_retention_checks_fail_when_nothing_was_evicted() -> None:
    # Given: the sweep never removed anything, so order and reclamation are unproven
    evidence = Task13DriverEvidence(
        mode="retention", retention=_retention(age_evicted=0, quota_evicted=0)
    )

    # When
    checks = {
        check.name: check for check in build_checks("retention", evidence, cleanup_succeeded=True)
    }

    # Then
    assert not checks["age-eviction-reclaims-crops"].passed
    assert not checks["quota-evicts-oldest-first"].passed


def test_crash_checks_require_reconciliation_and_resumed_publishing() -> None:
    # Given: a restart that left the orphan crop in place
    evidence = Task13DriverEvidence(mode="retention-crash", crash=_crash(orphan_removed=False))

    # When
    checks = {
        check.name: check
        for check in build_checks("retention-crash", evidence, cleanup_succeeded=True)
    }

    # Then
    assert list(checks) == [
        "orphan-and-temporary-reconciled",
        "interrupted-gc-replayed",
        "referenced-crops-intact",
        "publishing-resumed-after-restart",
        "graceful-worker-stop",
        "owned-resource-cleanup",
    ]
    assert not checks["orphan-and-temporary-reconciled"].passed
    assert checks["publishing-resumed-after-restart"].passed


def test_missing_mode_evidence_fails_every_behavior_check() -> None:
    # Given: the driver produced no retention observations
    evidence = Task13DriverEvidence(mode="retention")

    # When
    checks = build_checks("retention", evidence, cleanup_succeeded=True)

    # Then: only the independent cleanup receipt can pass
    assert [check.name for check in checks if check.passed] == ["owned-resource-cleanup"]
