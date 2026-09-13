"""Build binary checks from typed Task 16 driver evidence."""

from typing import Final

from gods_watching.verification.models import Check

from .task16_models import PlaywrightPhase, Task16DriverEvidence

_MISSING: Final = "driver produced no search UI observations"
_NORMAL_TESTS: Final = 2
_OUTAGE_TESTS: Final = 1


def _phase_detail(phase: PlaywrightPhase) -> str:
    return (
        f"exit={phase.exit_code}; expected={phase.expected}; unexpected={phase.unexpected}; "
        f"skipped={phase.skipped}; flaky={phase.flaky}"
    )


def build_checks(
    *, web_build_succeeded: bool, evidence: Task16DriverEvidence, cleanup_succeeded: bool
) -> tuple[Check, ...]:
    """Require a production build, every real-API UI flow, outage recovery, and cleanup."""
    ui = evidence.ui
    build = Check(
        name="web-build-succeeded",
        passed=web_build_succeeded,
        detail=f"web_build_succeeded={web_build_succeeded}",
    )
    cleanup = Check(
        name="owned-resource-cleanup",
        passed=cleanup_succeeded,
        detail=f"cleanup_succeeded={cleanup_succeeded}",
    )
    if ui is None:
        return (
            build,
            Check(name="real-api-search-flows-pass", passed=False, detail=_MISSING),
            Check(name="inference-outage-recovers-in-ui", passed=False, detail=_MISSING),
            cleanup,
        )
    normal, outage = ui.normal, ui.outage
    return (
        build,
        Check(
            name="real-api-search-flows-pass",
            passed=(
                normal.exit_code == 0
                and normal.expected == _NORMAL_TESTS
                and normal.unexpected == 0
                and normal.flaky == 0
            ),
            detail=_phase_detail(normal),
        ),
        Check(
            name="inference-outage-recovers-in-ui",
            passed=(
                outage.exit_code == 0
                and outage.expected == _OUTAGE_TESTS
                and outage.unexpected == 0
                and outage.flaky == 0
                and ui.outage_marker_seen
                and ui.inference_recovered
            ),
            detail=(
                f"{_phase_detail(outage)}; marker_seen={ui.outage_marker_seen}; "
                f"recovered={ui.inference_recovered}"
            ),
        ),
        cleanup,
    )


__all__ = ["build_checks"]
