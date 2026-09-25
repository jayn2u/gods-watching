"""GitHub reporting tests with an argument-recording gh runner."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeGuard, cast

from gods_watching.journeys.agents import CompletedCommand
from gods_watching.journeys.github import GitHubClient, close_pr_issues, publish_results
from gods_watching.journeys.models import (
    BugReport,
    ExpectedOutcome,
    Journey,
    JourneyRun,
    Observation,
    Violation,
)
from gods_watching.journeys.report import RunContext


class RecordingGitHubRunner:
    """Record subprocess vectors and capture file-backed request payloads."""

    def __init__(self, issues: list[dict[str, object]] | None = None) -> None:
        self.issues: list[dict[str, object]] = issues or []
        self.pr_comments: list[dict[str, object]] = []
        self.commands: list[tuple[str, ...]] = []
        self.request_payloads: list[object] = []
        self.body_texts: list[str] = []
        self.created_issues: list[dict[str, str]] = []
        self.issue_comments: list[tuple[int, str]] = []
        self.closed_issues: list[tuple[int, str]] = []
        self.next_issue_number: int = 100

    def __call__(self, command: Sequence[str], cwd: Path) -> CompletedCommand:
        del cwd
        arguments = tuple(command)
        self.commands.append(arguments)
        if arguments[1:3] == ("issue", "list"):
            return CompletedCommand(0, json.dumps(self.issues), "")
        if arguments[1:2] == ("api",):
            return self._handle_api(arguments)
        if arguments[1:3] == ("issue", "create"):
            return self._create_issue(arguments)
        if arguments[1:3] == ("issue", "comment"):
            return self._comment_issue(arguments)
        return CompletedCommand(0, "{}", "")

    def _handle_api(self, arguments: tuple[str, ...]) -> CompletedCommand:
        endpoint = arguments[2]
        if endpoint.endswith("/comments") and "--method" not in arguments:
            return CompletedCommand(0, json.dumps(self.pr_comments), "")
        payload = self._read_input(arguments)
        if endpoint.endswith("/check-runs"):
            self.request_payloads.append(payload)
            return CompletedCommand(0, "{}", "")
        if endpoint.endswith("/comments"):
            self.request_payloads.append(payload)
            if _is_json_object(payload):
                body = payload.get("body")
                if isinstance(body, str):
                    self.body_texts.append(body)
        if (
            "/issues/" in endpoint
            and _is_json_object(payload)
            and payload.get("state") == "closed"
        ):
            issue_number = int(endpoint.rsplit("/", maxsplit=1)[1])
            reason = str(payload.get("state_reason"))
            self.closed_issues.append((issue_number, reason))
        self.request_payloads.append(payload)
        return CompletedCommand(0, "{}", "")

    def _create_issue(self, arguments: tuple[str, ...]) -> CompletedCommand:
        title = arguments[arguments.index("--title") + 1]
        body = self._read_body_file(arguments)
        self.created_issues.append({"title": title, "body": body})
        self.next_issue_number += 1
        self.issues.append({"number": self.next_issue_number, "body": body})
        self.body_texts.append(body)
        return CompletedCommand(0, f"https://github.test/issues/{self.next_issue_number}", "")

    def _comment_issue(self, arguments: tuple[str, ...]) -> CompletedCommand:
        number = int(arguments[3])
        body = self._read_body_file(arguments)
        self.issue_comments.append((number, body))
        self.body_texts.append(body)
        return CompletedCommand(0, "comment created", "")

    def _read_body_file(self, command: tuple[str, ...]) -> str:
        path = Path(command[command.index("--body-file") + 1])
        return path.read_text(encoding="utf-8")

    def _read_input(self, command: tuple[str, ...]) -> object:
        if "--input" not in command:
            return None
        path = Path(command[command.index("--input") + 1])
        return cast("object", json.loads(path.read_text(encoding="utf-8")))


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


def sample_journey(journey_id: str, title: str) -> Journey:
    """Return a Journey with one expected condition for issue rendering."""
    return Journey(
        id=journey_id,
        title=title,
        setup=(),
        tags=(),
        preconditions="The application is available.",
        goal=f"Confirm {title.lower()} works.",
        expected_outcomes=(ExpectedOutcome(id="E1", text=f"{title} behaves correctly."),),
        source_path=Path(f"{journey_id}.md"),
    )


def bug_report(journey: Journey, fingerprint_value: str) -> BugReport:
    """Build one Cross-check-confirmed report for a Journey."""
    outcome = journey.expected_outcomes[0]
    return BugReport(
        journey_id=journey.id,
        outcomes=(outcome,),
        observed=(
            Violation(
                outcome_id=outcome.id,
                observed=f"Observed {journey.id} failure.",
                evidence_files=("attempt-1/error.png",),
            ),
        ),
        judge_reasons=("The evidence confirms the violation.",),
        fingerprint=fingerprint_value,
    )


def context() -> RunContext:
    """Return fixed PR metadata for publisher tests."""
    return RunContext("owner/repository", 42, "abc123", "https://github.test/runs/7")


def test_find_open_agent_issues_extracts_fingerprint_markers() -> None:
    """The issue-list contract parses hidden markers without using the title."""
    issue_body = "<!-- agent-journeys:fingerprint=abc123 journey=login pr=42 -->\nDetails"
    runner = RecordingGitHubRunner([{"number": 17, "body": issue_body}])
    client = GitHubClient(runner, "owner/repository")

    issues = client.find_open_agent_issues()

    assert len(issues) == 1
    assert issues[0].number == 17
    assert issues[0].fingerprint == "abc123"
    assert issues[0].journey_id == "login"
    assert issues[0].pr_number == 42
    assert runner.commands[0] == (
        "gh",
        "issue",
        "list",
        "--label",
        "agent-reported",
        "--state",
        "open",
        "--json",
        "number,body",
        "--limit",
        "500",
        "--repo",
        "owner/repository",
    )


def test_publish_results_comments_on_duplicate_creates_new_and_closes_matching_pass_issue() -> None:
    """Deduplication, creation, and pass closure use marker identity and PR scope."""
    login = sample_journey("login", "Operator sign-in")
    search = sample_journey("text-search", "Text search")
    cameras = sample_journey("live-cameras", "Live cameras")
    duplicate_fp = "a1b2c3d4"
    runner = RecordingGitHubRunner(
        [
            {
                "number": 11,
                "body": f"<!-- agent-journeys:fingerprint={duplicate_fp} journey=login pr=42 -->",
            },
            {
                "number": 12,
                "body": "<!-- agent-journeys:fingerprint=bead journey=live-cameras pr=42 -->",
            },
            {
                "number": 13,
                "body": "<!-- agent-journeys:fingerprint=babe journey=live-cameras pr=99 -->",
            },
        ]
    )
    client = GitHubClient(runner, "owner/repository")
    runs = (
        JourneyRun(
            journey_id="login",
            verdict="bug",
            bug_report=bug_report(login, duplicate_fp),
            observations=(),
            attempts=2,
            failure=None,
        ),
        JourneyRun(
            journey_id="text-search",
            verdict="bug",
            bug_report=bug_report(search, "feedface"),
            observations=(),
            attempts=2,
            failure=None,
        ),
        JourneyRun(
            journey_id="live-cameras",
            verdict="pass",
            bug_report=None,
            observations=(Observation(text="Wall loaded after a brief delay."),),
            attempts=1,
            failure=None,
        ),
    )

    publish_results(runs, (login, search, cameras), context(), client)

    assert runner.issue_comments[0][0] == 11
    assert f"fingerprint={duplicate_fp}" in runner.issue_comments[0][1]
    assert len(runner.created_issues) == 1
    assert runner.created_issues[0]["title"] == "[Agent Journey] Text search: E1"
    assert runner.closed_issues == [(12, "completed")]
    assert 13 not in {number for number, _reason in runner.closed_issues}
    assert any("--body-file" in command for command in runner.commands)
    assert any("--input" in command for command in runner.commands)
    assert all(
        body_text not in command
        for command in runner.commands
        for body_text in runner.body_texts
    )


def test_pr_comment_is_created_or_updated_by_hidden_marker() -> None:
    """A prior marked comment is patched instead of creating a second comment."""
    existing_comment: dict[str, object] = {
        "id": 901,
        "body": "<!-- agent-journeys:pr-comment -->\nold",
    }
    runner = RecordingGitHubRunner()
    runner.pr_comments = [existing_comment]
    client = GitHubClient(runner, "owner/repository")

    client.upsert_pr_comment(42, "<!-- agent-journeys:pr-comment -->\nnew")

    assert runner.commands[1][2:4] == (
        "repos/owner/repository/issues/comments/901",
        "--method",
    )
    assert "PATCH" in runner.commands[1]
    assert runner.request_payloads == [{"body": "<!-- agent-journeys:pr-comment -->\nnew"}]


def test_create_check_run_sends_summary_in_a_json_input_file() -> None:
    """The check-run summary is file-backed and absent from the command arguments."""
    runner = RecordingGitHubRunner()
    client = GitHubClient(runner, "owner/repository")

    client.create_check_run("abc123", "neutral", "Confirmed Bug Reports need triage.")

    assert runner.commands[0][:5] == (
        "gh",
        "api",
        "repos/owner/repository/check-runs",
        "--method",
        "POST",
    )
    assert runner.commands[0][-2] == "--input"
    assert runner.request_payloads == [
        {
            "name": "agent-journeys",
            "head_sha": "abc123",
            "status": "completed",
            "conclusion": "neutral",
            "output": {
                "title": "Agent Journey verification",
                "summary": "Confirmed Bug Reports need triage.",
            },
        }
    ]
    assert "Confirmed Bug Reports need triage." not in runner.commands[0]


def test_close_pr_issues_leaves_other_pull_requests_open() -> None:
    """Closing an unmerged PR affects only open marked Issues for that PR."""
    runner = RecordingGitHubRunner(
        [
            {"number": 21, "body": "<!-- agent-journeys:fingerprint=a journey=login pr=42 -->"},
            {"number": 22, "body": "<!-- agent-journeys:fingerprint=b journey=login pr=43 -->"},
        ]
    )
    client = GitHubClient(runner, "owner/repository")

    close_pr_issues(client, 42)

    assert runner.closed_issues == [(21, "not_planned")]
