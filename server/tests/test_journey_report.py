"""Snapshot tests for Journey comments, Issues, and check conclusions."""

from pathlib import Path

import pytest

from gods_watching.journeys.models import (
    BugReport,
    ExpectedOutcome,
    Journey,
    JourneyRun,
    Observation,
    Verdict,
    Violation,
)
from gods_watching.journeys.report import (
    RunContext,
    check_conclusion,
    render_issue,
    render_pr_comment,
)


def sample_journey(journey_id: str = "login", title: str = "Sign in") -> Journey:
    """Return the source record used by both Markdown snapshots."""
    return Journey(
        id=journey_id,
        title=title,
        setup=(),
        tags=(),
        preconditions="The sign-in screen is available.",
        goal="Confirm operator access is protected.",
        expected_outcomes=(
            ExpectedOutcome(id="E1", text="An invalid password is visibly rejected."),
        ),
        source_path=Path(f"{journey_id}.md"),
    )


def sample_context() -> RunContext:
    """Return fixed pull request metadata for stable rendering snapshots."""
    return RunContext(
        repository="owner/repository",
        pr_number=42,
        head_sha="abc123def456",
        run_url="https://github.test/owner/repository/actions/runs/7",
    )


def test_pr_comment_snapshot_has_verdict_bug_observation_sections_and_run_link() -> None:
    """A PR comment starts with its marker and follows the required Markdown layout."""
    runs = (
        JourneyRun(
            journey_id="login",
            verdict="pass",
            bug_report=None,
            observations=(Observation(text="The sign-in response was briefly delayed."),),
            attempts=1,
            failure=None,
        ),
    )

    comment = render_pr_comment(runs, sample_context())

    assert comment == (
        "<!-- agent-journeys:pr-comment -->\n\n"
        "## Journey Verdicts\n"
        "| Journey | Verdict | Attempts |\n"
        "| --- | --- | ---: |\n"
        "| `login` | pass | 1 |\n\n"
        "## Bug Reports\n"
        "No confirmed Bug Reports.\n\n"
        "## Observations\n"
        "- `login`: The sign-in response was briefly delayed.\n\n"
        "[View workflow run](https://github.test/owner/repository/actions/runs/7)\n"
    )


def test_issue_snapshot_contains_fingerprint_and_reproduction_context() -> None:
    """A Bug Report Issue contains its deduplication marker and supporting evidence."""
    journey = sample_journey()
    report = BugReport(
        journey_id="login",
        outcomes=(journey.expected_outcomes[0],),
        observed=(
            Violation(
                outcome_id="E1",
                observed="The console opened after the invalid password.",
                evidence_files=("attempt-1/login-error.png",),
            ),
        ),
        judge_reasons=("The screenshot shows the authenticated console.",),
        fingerprint="a1b2c3d4",
    )

    title, body = render_issue(report, journey, sample_context())

    assert title == "[Agent Journey] Sign in: E1"
    assert body == (
        "<!-- agent-journeys:fingerprint=a1b2c3d4 journey=login pr=42 -->\n\n"
        "## Confirmed Bug Report\n"
        "Journey: **Sign in** (`login`)\n"
        "Pull request: #42\n"
        "Head: `abc123def456`\n\n"
        "### Expected Outcome E1\n"
        "An invalid password is visibly rejected.\n\n"
        "### Observed\n"
        "The console opened after the invalid password.\n\n"
        "### Cross-check\n"
        "The screenshot shows the authenticated console.\n\n"
        "Evidence: `attempt-1/login-error.png`\n\n"
        "[Run details](https://github.test/owner/repository/actions/runs/7)\n"
    )


@pytest.mark.parametrize(
    ("verdicts", "harness_failed", "expected"),
    [
        ((), False, "success"),
        (("pass", "inconclusive"), False, "success"),
        (("bug",), False, "neutral"),
        (("bug",), True, "failure"),
        (("pass",), True, "failure"),
    ],
    ids=("empty", "inconclusive-is-neutral", "bug", "harness-failed", "failed-harness"),
)
def test_check_conclusion_matches_harness_and_bug_policy(
    verdicts: tuple[Verdict, ...],
    harness_failed: bool,
    expected: str,
) -> None:
    """Confirmed Bug Reports are neutral, harness failures fail, and others pass."""
    runs = tuple(
        JourneyRun(
            journey_id=f"journey-{index}",
            verdict=verdict,
            bug_report=(
                BugReport(
                    journey_id=f"journey-{index}",
                    outcomes=(ExpectedOutcome(id="E1", text="The behavior is correct."),),
                    observed=(),
                    judge_reasons=(),
                    fingerprint=f"fingerprint-{index}",
                )
                if verdict == "bug"
                else None
            ),
            observations=(),
            attempts=1,
            failure=None,
        )
        for index, verdict in enumerate(verdicts)
    )

    assert check_conclusion(runs, harness_failed) == expected
