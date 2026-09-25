"""Render human-readable PR comments and Bug Report Issues."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from .models import BugReport, Journey, JourneyRun

_PR_COMMENT_MARKER: Final = "<!-- agent-journeys:pr-comment -->"


@dataclass(frozen=True, slots=True)
class RunContext:
    """Identify the pull request and workflow run associated with one report."""

    repository: str
    pr_number: int
    head_sha: str
    run_url: str


def render_pr_comment(runs: Sequence[JourneyRun], context: RunContext) -> str:
    """Render the stable Verdict table, Bug Reports, Observations, and run link."""
    table_rows = [
        f"| `{_inline(run.journey_id)}` | {run.verdict} | {run.attempts} |" for run in runs
    ]
    table = "\n".join(
        (
            "| Journey | Verdict | Attempts |",
            "| --- | --- | ---: |",
            *table_rows,
        )
    )
    bug_rows: list[str] = []
    for run in runs:
        if run.bug_report is None:
            continue
        outcome_ids = ", ".join(outcome.id for outcome in run.bug_report.outcomes)
        outcome_fingerprint = f"(fingerprint `{run.bug_report.fingerprint}`)"
        bug_rows.append(
            f"- `{_inline(run.journey_id)}`: {outcome_ids} {outcome_fingerprint}"
        )
    observation_rows = [
        f"- `{_inline(run.journey_id)}`: {_inline(observation.text)}"
        for run in runs
        for observation in run.observations
    ]
    failure_rows = [
        f"- `{_inline(run.journey_id)}`: harness detail: {_inline(run.failure)}"
        for run in runs
        if run.failure is not None
    ]
    if not bug_rows:
        bug_rows.append("No confirmed Bug Reports.")
    if not observation_rows and not failure_rows:
        observation_rows.append("No additional Observations.")
    observation_rows.extend(failure_rows)
    return "\n".join(
        (
            _PR_COMMENT_MARKER,
            "",
            "## Journey Verdicts",
            table,
            "",
            "## Bug Reports",
            *bug_rows,
            "",
            "## Observations",
            *observation_rows,
            "",
            f"[View workflow run]({context.run_url})",
            "",
        )
    )


def render_issue(report: BugReport, journey: Journey, context: RunContext) -> tuple[str, str]:
    """Render an Issue title and reproduction context for one confirmed Bug Report."""
    outcome_ids = ", ".join(outcome.id for outcome in report.outcomes)
    title = f"[Agent Journey] {journey.title}: {outcome_ids}"
    observed_by_id = {violation.outcome_id: violation for violation in report.observed}
    reason_by_id = dict(
        zip(
            (outcome.id for outcome in report.outcomes),
            report.judge_reasons,
            strict=True,
        )
    )
    outcome_sections: list[str] = []
    for outcome in report.outcomes:
        violation = observed_by_id.get(outcome.id)
        observed_text = (
            violation.observed if violation is not None else "No observed text was recorded."
        )
        evidence_files = violation.evidence_files if violation is not None else ()
        evidence = ", ".join(f"`{_inline(path)}`" for path in evidence_files) or "None recorded."
        reason = reason_by_id.get(outcome.id, "No Cross-check reason was recorded.")
        outcome_sections.extend(
            (
                f"### Expected Outcome {outcome.id}",
                _inline(outcome.text),
                "",
                "### Observed",
                _inline(observed_text),
                "",
                "### Cross-check",
                _inline(reason),
                "",
                f"Evidence: {evidence}",
                "",
            )
        )
    marker = (
        f"<!-- agent-journeys:fingerprint={report.fingerprint} "
        f"journey={journey.id} pr={context.pr_number} -->"
    )
    body = "\n".join(
        (
            marker,
            "",
            "## Confirmed Bug Report",
            f"Journey: **{_inline(journey.title)}** (`{_inline(journey.id)}`)",
            f"Pull request: #{context.pr_number}",
            f"Head: `{_inline(context.head_sha)}`",
            "",
            *outcome_sections,
            f"[Run details]({context.run_url})",
            "",
        )
    )
    return title, body


def check_conclusion(
    runs: Sequence[JourneyRun],
    harness_failed: bool,
) -> Literal["success", "neutral", "failure"]:
    """Return the non-blocking GitHub check conclusion for a completed harness run."""
    if harness_failed:
        return "failure"
    if any(run.bug_report is not None for run in runs):
        return "neutral"
    return "success"


def _inline(value: str) -> str:
    return value.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


__all__ = ["RunContext", "check_conclusion", "render_issue", "render_pr_comment"]
