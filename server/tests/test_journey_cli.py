"""CLI tests with an injected Journey services factory."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, NoReturn, Self, TypeGuard, cast

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from gods_watching import cli as product_cli
from gods_watching.journeys import cli as journey_cli
from gods_watching.journeys.agents import ClaudeCliExecutor
from gods_watching.journeys.models import (
    ExecutorReport,
    Journey,
    OperatorCredentials,
)
from gods_watching.journeys.stack import ComposeJourneyStack

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from urllib.request import Request

    from gods_watching.journeys.agents import JourneyExecutor, Judge
    from gods_watching.journeys.pipeline import JourneyStack


class RecordingStack:
    """Record shutdown while satisfying the public Journey stack protocol."""

    events: list[str]
    shutdown_calls: int

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.shutdown_calls = 0

    def prepare_journey(self, journey: Journey) -> None:
        del journey

    def collect_evidence(self, evidence_dir: Path) -> None:
        del evidence_dir

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.events.append("shutdown")


class RecordingGuard:
    """Record entry and exit around the harness body."""

    events: list[str]
    entered: bool
    exited: bool

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.entered = False
        self.exited = False

    def __enter__(self) -> Self:
        self.entered = True
        self.events.append("guard-enter")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc_value, traceback
        self.exited = True
        self.events.append("guard-exit")
        return False


class NeverExecutor:
    """Fail loudly if a test unexpectedly starts a browser Journey."""

    def execute(
        self,
        journey: Journey,
        *,
        base_url: str,
        credentials: OperatorCredentials,
        evidence_dir: Path,
    ) -> NoReturn:
        del journey, base_url, credentials, evidence_dir
        failure_message = "unexpected executor use"
        raise AssertionError(failure_message)


class NeverJudge:
    """Fail loudly if a test unexpectedly starts a Cross-check."""

    def judge(
        self,
        journey: Journey,
        report: ExecutorReport,
        *,
        evidence_dir: Path,
    ) -> NoReturn:
        del journey, report, evidence_dir
        failure_message = "unexpected Cross-check use"
        raise AssertionError(failure_message)


class RecordingCheckClient:
    """Capture the check summary emitted after publishing a failed harness run."""

    summaries: list[str]

    def __init__(self) -> None:
        self.summaries = []

    def create_check_run(self, head_sha: str, conclusion: str, summary: str) -> None:
        del head_sha, conclusion
        self.summaries.append(summary)


@pytest.mark.parametrize(
    "publish",
    [False, True],
    ids=("without-publish", "with-check-summary"),
)
@pytest.mark.parametrize(
    ("failure_message", "expected_failure"),
    [
        (
            "pipeline failed with private details",
            "RuntimeError: pipeline failed with private details",
        ),
        ("pipeline failed with fixture-secret-value", "RuntimeError"),
    ],
)
def test_run_shuts_down_and_writes_results_when_journey_runner_raises(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_message: str,
    expected_failure: str,
    publish: bool,
) -> None:
    """The run factory supplies fakes and cleanup completes after a pipeline exception."""
    journey_dir = tmp_path / "journeys"
    journey_dir.mkdir()
    journey_contents = """---
id: login
title: Sign in
setup:
tags:
---
## Preconditions
The application is ready.

## Goal
Confirm sign-in.

## Expected Outcomes
- [E1] An invalid password is rejected.
"""
    _ = (journey_dir / "login.md").write_text(
        journey_contents,
        encoding="utf-8",
    )
    events: list[str] = []
    stack = RecordingStack(events)
    guard = RecordingGuard(events)
    fixture_prepared: list[bool] = []
    runner_calls: list[tuple[Journey, ...]] = []

    def fail_run(  # noqa: PLR0913
        journeys: Sequence[Journey],
        *,
        stack: JourneyStack,
        executor: JourneyExecutor,
        judge: Judge,
        base_url: str,
        credentials: OperatorCredentials,
        run_dir: Path,
    ) -> NoReturn:
        del stack, executor, judge, base_url, credentials, run_dir
        runner_calls.append(tuple(journeys))
        events.append("pipeline")
        raise RuntimeError(failure_message)

    def prepare_fixtures() -> None:
        fixture_prepared.append(True)
        events.append("fixtures")

    services = journey_cli.JourneyServices(
        stack=stack,
        guard=guard,
        executor=NeverExecutor(),
        judge=NeverJudge(),
        base_url="http://localhost:18080",
        credentials=OperatorCredentials(
            username="fixture-operator",
            password=SecretStr("fixture-secret-value"),
        ),
        prepare_fixtures=prepare_fixtures,
        run_journeys=fail_run,
    )
    factory_calls: list[tuple[Path, Path, Path]] = []

    def factory(
        repository_root: Path,
        env_path: Path,
        run_dir: Path,
    ) -> journey_cli.JourneyServices:
        factory_calls.append((repository_root, env_path, run_dir))
        return services

    monkeypatch.setattr(journey_cli, "create_journey_services", factory)
    check_client = RecordingCheckClient()
    if publish:
        def check_client_factory(repository: str) -> RecordingCheckClient:
            del repository
            return check_client

        def skip_issue_and_comment_publish(*args: object, **kwargs: object) -> None:
            del args, kwargs

        monkeypatch.setattr(journey_cli, "create_github_client", check_client_factory)
        monkeypatch.setattr(journey_cli, "publish_results", skip_issue_and_comment_publish)
    run_dir = tmp_path / "run-output"
    arguments = [
        "run",
        "--pr",
        "42",
        "--head-sha",
        "abc123",
        "--run-url",
        "https://github.test/runs/7",
        "--run-dir",
        str(run_dir),
        "--journeys-dir",
        str(journey_dir),
        "--env-file",
        str(tmp_path / ".env"),
    ]
    if publish:
        arguments.extend(("--repository", "owner/repository", "--publish"))
    else:
        arguments.append("--no-publish")
    result = CliRunner().invoke(
        journey_cli.app,
        arguments,
    )

    results_path = run_dir / "results.json"
    results_text = results_path.read_text(encoding="utf-8")
    payload = _json_object(results_text)
    assert result.exit_code == 1
    assert guard.entered is True
    assert guard.exited is True
    assert stack.shutdown_calls == 1
    assert events == ["guard-enter", "fixtures", "pipeline", "shutdown", "guard-exit"]
    assert fixture_prepared == [True]
    assert tuple(journey.id for journey in runner_calls[0]) == ("login",)
    assert factory_calls == [(Path.cwd(), tmp_path / ".env", run_dir)]
    assert payload["harness_failed"] is True
    assert payload["failure"] == expected_failure
    assert expected_failure in result.output
    assert f"Journey harness failed: {expected_failure}" in result.stderr
    assert "fixture-secret-value" not in result.output
    assert "fixture-secret-value" not in results_text
    if publish:
        assert check_client.summaries == [
            f"Journey harness failed: {expected_failure}."
        ]


def test_create_journey_services_passes_the_cli_env_file_to_compose(
    tmp_path: Path,
) -> None:
    """The same resolved environment file configures both Compose projects."""
    env_file = tmp_path / ".env"
    _ = env_file.write_text(
        dedent(
            """\
            COMPOSE_PROJECT_NAME=gw-ci
            GW_PUBLIC_ORIGIN=http://localhost:18080
            GW_OPERATOR_USERNAME=fixture-operator
            GW_OPERATOR_PASSWORD=fixture-secret-value
            GW_FIXTURE_RTSP_PORT=38554
            """
        ),
        encoding="utf-8",
    )
    repository_root = tmp_path / "repository"
    services = journey_cli.create_journey_services(
        repository_root,
        env_file,
        tmp_path / "run-output",
    )

    assert isinstance(services.stack, ComposeJourneyStack)
    assert services.stack.env_file == env_file
    assert isinstance(services.executor, ClaudeCliExecutor)
    assert services.executor.playwright_cli_path == (
        repository_root / "web/node_modules/@playwright/mcp/cli.js"
    )


def test_download_fixture_uses_browser_user_agent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The fixture request carries the browser-like User-Agent and streams bytes."""
    requests: list[Request] = []
    timeouts: list[float] = []

    def fake_urlopen(request: Request, *, timeout: float) -> BytesIO:
        requests.append(request)
        timeouts.append(timeout)
        return BytesIO(b"fixture bytes")

    monkeypatch.setattr(journey_cli, "urlopen", fake_urlopen)
    destination = tmp_path / "camera.mp4"

    journey_cli.download_fixture(
        "https://fixture.example/camera.mp4",
        destination,
    )

    assert len(requests) == 1
    assert requests[0].get_header("User-agent") == (
        "Mozilla/5.0 (X11; Linux x86_64) gods-watching-journeys"
    )
    assert timeouts == [60]
    assert destination.read_bytes() == b"fixture bytes"


def test_product_cli_registers_the_journeys_command() -> None:
    """The user-facing gods-watching command exposes the Journey subcommands."""
    result = CliRunner().invoke(product_cli.app, ["journeys", "--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "close-pr-issues" in result.stdout


def test_close_pr_issues_command_uses_the_injected_client_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The closed-PR command delegates to a file-backed fake GitHub client."""
    client = object()
    repositories: list[str] = []

    def create_client(repository: str) -> object:
        repositories.append(repository)
        return client

    monkeypatch.setattr(journey_cli, "create_github_client", create_client)
    def close_fake(actual_client: object, pr: int) -> None:
        del actual_client, pr

    monkeypatch.setattr(journey_cli, "close_pr_issues", close_fake)

    result = CliRunner().invoke(
        journey_cli.app,
        ["close-pr-issues", "--pr", "42", "--repository", "owner/repository"],
    )

    assert result.exit_code == 0
    assert repositories == ["owner/repository"]


def _json_object(contents: str) -> dict[str, object]:
    decoded = cast("object", json.loads(contents))
    if not _is_json_object(decoded):
        raise AssertionError
    return decoded


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))
