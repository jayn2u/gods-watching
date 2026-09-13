"""Typed evidence emitted by the Task 16 real-API search UI driver."""

from gods_watching.verification.models import VerificationModel


class PlaywrightPhase(VerificationModel):
    """Summarize one Playwright run of the real-API search spec."""

    exit_code: int
    expected: int
    unexpected: int
    skipped: int
    flaky: int


class SearchUiEvidence(VerificationModel):
    """Record the built UI exercising the real search API, including an inference outage."""

    normal: PlaywrightPhase
    outage: PlaywrightPhase
    outage_marker_seen: bool
    inference_recovered: bool


class Task16DriverEvidence(VerificationModel):
    """Wrap the Task 16 observations for the scenario report."""

    ui: SearchUiEvidence | None = None


__all__ = ["PlaywrightPhase", "SearchUiEvidence", "Task16DriverEvidence"]
