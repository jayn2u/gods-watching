"""File-backed GitHub CLI operations and Journey issue publishing."""

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, Literal, TypeGuard, cast, override

from .agents import CommandRunner
from .models import Journey, JourneyRun
from .report import RunContext, render_issue, render_pr_comment

_ISSUE_MARKER_PATTERN: Final = re.compile(
    r"<!--\s*agent-journeys:fingerprint=([a-f0-9]+)\s+journey=([a-z0-9-]+)\s+pr=([0-9]+)\s*-->"
)
_PR_COMMENT_MARKER: Final = "<!-- agent-journeys:pr-comment -->"


@dataclass(frozen=True, slots=True)
class AgentIssue:
    """Capture issue metadata parsed from its hidden Journey marker."""

    number: int
    body: str
    fingerprint: str | None
    journey_id: str | None
    pr_number: int | None


@dataclass(frozen=True, slots=True)
class GitHubCommandError(RuntimeError):
    """Report a failed gh invocation without echoing request file contents."""

    command: tuple[str, ...]
    returncode: int

    @override
    def __str__(self) -> str:
        """Return a generic command failure without leaking arguments or payloads."""
        return f"GitHub CLI command failed with exit code {self.returncode}"


@dataclass(frozen=True, slots=True)
class GitHubOutputError(ValueError):
    """Reject malformed GitHub CLI JSON at the integration boundary."""

    endpoint: str
    reason: str

    @override
    def __str__(self) -> str:
        """Describe the invalid endpoint response without copying its contents."""
        return f"GitHub CLI returned invalid output for {self.endpoint}: {self.reason}"


@dataclass(frozen=True, slots=True)
class GitHubClient:
    """Use gh with argument vectors and temporary files for every request body."""

    runner: CommandRunner
    repository: str

    def upsert_pr_comment(self, pr: int, body: str) -> None:
        """Update the marked PR comment or create it when the marker is absent."""
        endpoint = f"repos/{self.repository}/issues/{pr}/comments"
        comments = _json_array(self._run(("gh", "api", endpoint, "--paginate")), endpoint)
        existing_id = _marked_comment_id(comments)
        if existing_id is None:
            _ = self._api_request(endpoint, "POST", {"body": body})
            return
        _ = self._api_request(
            f"repos/{self.repository}/issues/comments/{existing_id}",
            "PATCH",
            {"body": body},
        )

    def find_open_agent_issues(self) -> tuple[AgentIssue, ...]:
        """List open agent Issues and parse their Journey markers."""
        command = (
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
            self.repository,
        )
        entries = _json_array(self._run(command), "gh issue list")
        issues: list[AgentIssue] = []
        for entry in entries:
            if not _is_json_object(entry):
                continue
            number = entry.get("number")
            body = entry.get("body")
            if not isinstance(number, int) or not isinstance(body, str):
                continue
            match = _ISSUE_MARKER_PATTERN.search(body)
            issues.append(
                AgentIssue(
                    number=number,
                    body=body,
                    fingerprint=match.group(1) if match else None,
                    journey_id=match.group(2) if match else None,
                    pr_number=int(match.group(3)) if match else None,
                )
            )
        return tuple(issues)

    def create_issue(self, title: str, body: str) -> None:
        """Create a triage-labeled issue with its body supplied through a temp file."""
        with _temporary_file(body, suffix=".md") as body_path:
            command = (
                "gh",
                "issue",
                "create",
                "--title",
                title,
                "--body-file",
                str(body_path),
                "--label",
                "agent-reported",
                "--label",
                "needs-triage",
                "--repo",
                self.repository,
            )
            _ = self._run(command)

    def comment_issue(self, number: int, body: str) -> None:
        """Add an Issue comment from a temporary Markdown file."""
        with _temporary_file(body, suffix=".md") as body_path:
            command = (
                "gh",
                "issue",
                "comment",
                str(number),
                "--body-file",
                str(body_path),
                "--repo",
                self.repository,
            )
            _ = self._run(command)

    def close_issue(
        self,
        number: int,
        reason: Literal["completed", "not planned"],
        comment: str,
    ) -> None:
        """Close an Issue with its REST state reason and optional file-backed comment."""
        state_reason = "not_planned" if reason == "not planned" else "completed"
        endpoint = f"repos/{self.repository}/issues/{number}"
        _ = self._api_request(
            endpoint,
            "PATCH",
            {"state": "closed", "state_reason": state_reason},
        )
        if comment:
            _ = self._api_request(f"{endpoint}/comments", "POST", {"body": comment})

    def create_check_run(self, head_sha: str, conclusion: str, summary: str) -> None:
        """Create the completed, non-blocking Agent Journey check run."""
        payload: dict[str, object] = {
            "name": "agent-journeys",
            "head_sha": head_sha,
            "status": "completed",
            "conclusion": conclusion,
            "output": {
                "title": "Agent Journey verification",
                "summary": summary,
            },
        }
        _ = self._api_request(f"repos/{self.repository}/check-runs", "POST", payload)

    def _api_request(self, endpoint: str, method: str, payload: Mapping[str, object]) -> object:
        with _temporary_file(json.dumps(payload), suffix=".json") as input_path:
            command = (
                "gh",
                "api",
                endpoint,
                "--method",
                method,
                "--input",
                str(input_path),
            )
            output = self._run(command)
        if not output.strip():
            return {}
        try:
            return cast("object", json.loads(output))
        except json.JSONDecodeError as error:
            raise GitHubOutputError(endpoint, "response is not valid JSON") from error

    def _run(self, command: tuple[str, ...]) -> str:
        result = self.runner(command, Path.cwd())
        if result.returncode != 0:
            raise GitHubCommandError(command=command, returncode=result.returncode)
        return result.stdout


def publish_results(
    runs: Sequence[JourneyRun],
    journeys: Sequence[Journey],
    context: RunContext,
    client: GitHubClient,
) -> None:
    """Upsert the PR comment, deduplicate Bug Reports, and close matching passed Journeys."""
    client.upsert_pr_comment(context.pr_number, render_pr_comment(runs, context))
    issues = client.find_open_agent_issues()
    journeys_by_id = {journey.id: journey for journey in journeys}
    for run in runs:
        if run.bug_report is None:
            continue
        journey = journeys_by_id[run.journey_id]
        title, body = render_issue(run.bug_report, journey, context)
        duplicate = next(
            (issue for issue in issues if issue.fingerprint == run.bug_report.fingerprint),
            None,
        )
        if duplicate is None:
            client.create_issue(title, body)
        else:
            client.comment_issue(duplicate.number, body)

    for run in runs:
        if run.verdict != "pass":
            continue
        for issue in issues:
            if issue.journey_id == run.journey_id and issue.pr_number == context.pr_number:
                client.close_issue(
                    issue.number,
                    "completed",
                    f"Journey `{run.journey_id}` passed in [run]({context.run_url}).",
                )


def close_pr_issues(client: GitHubClient, pr_number: int) -> None:
    """Close open marked Bug Report Issues for one PR as not planned."""
    for issue in client.find_open_agent_issues():
        if issue.pr_number == pr_number:
            client.close_issue(
                issue.number,
                "not planned",
                f"Pull request #{pr_number} closed without merge; this Bug Report is not planned.",
            )


def _marked_comment_id(comments: list[object]) -> int | None:
    for comment in comments:
        if not _is_json_object(comment):
            continue
        body = comment.get("body")
        comment_id = comment.get("id")
        if isinstance(body, str) and _PR_COMMENT_MARKER in body and isinstance(comment_id, int):
            return comment_id
    return None


def _json_array(output: str, endpoint: str) -> list[object]:
    try:
        decoded = cast("object", json.loads(output))
    except json.JSONDecodeError as error:
        raise GitHubOutputError(endpoint, "response is not valid JSON") from error
    if not _is_json_array(decoded):
        raise GitHubOutputError(endpoint, "response must be a JSON array")
    return decoded


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


@contextmanager
def _temporary_file(contents: str, *, suffix: str) -> Iterator[Path]:
    path: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", suffix=suffix, delete=False) as file:
            path = Path(file.name)
            _ = file.write(contents)
        yield path
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


__all__ = [
    "AgentIssue",
    "GitHubClient",
    "GitHubCommandError",
    "GitHubOutputError",
    "close_pr_issues",
    "publish_results",
]
