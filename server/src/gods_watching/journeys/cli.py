"""Typer commands for running and closing Journey verification reports."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from http.cookiejar import CookieJar
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Protocol, cast
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

import typer
from typer.models import OptionInfo

from .agents import (
    ClaudeCliExecutor,
    ClaudeCliJudge,
    run_command,
)
from .catalog import load_catalog
from .fixtures import ensure_fixture_videos
from .github import GitHubClient, close_pr_issues, publish_results
from .http_setup import HttpSetup
from .pipeline import JourneyStack, run_journeys
from .report import RunContext, check_conclusion
from .stack import ComposeJourneyStack, DevStackGuard, load_ci_environment

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from http.client import HTTPResponse
    from types import TracebackType

    from .agents import JourneyExecutor, Judge
    from .models import Journey, JourneyRun, OperatorCredentials

from .models import StackError

_REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
_PLAYWRIGHT_CLI_RELATIVE_PATH = Path("web/node_modules/@playwright/mcp/cli.js")
_PR_OPTION: Final = OptionInfo(default="--pr")
_HEAD_SHA_OPTION: Final = OptionInfo(default="--head-sha")
_RUN_URL_OPTION: Final = OptionInfo(default="--run-url")
_RUN_DIR_OPTION: Final = OptionInfo(default="--run-dir")
_REPOSITORY_OPTION: Final = OptionInfo(default="--repository")
_JOURNEYS_DIR_OPTION: Final = OptionInfo(default="--journeys-dir")
_ENV_FILE_OPTION: Final = OptionInfo(default="--env-file")
_PUBLISH_OPTION: Final = OptionInfo(default="--publish/--no-publish")
_CLOSE_PR_OPTION: Final = OptionInfo(default="--pr")
_CLOSE_REPOSITORY_OPTION: Final = OptionInfo(default="--repository")
_DEFAULT_JOURNEYS_DIR: Final = Path("qa/journeys")
_DEFAULT_ENV_FILE: Final = Path(".env")


class ShutdownJourneyStack(JourneyStack, Protocol):
    """Add explicit teardown to the pipeline's stack interface."""

    def shutdown(self) -> None:
        """Stop and remove the two isolated CI Compose projects."""
        ...


class JourneyRunGuard(Protocol):
    """Expose the development stack context boundary to the CLI."""

    def __enter__(self) -> object:
        """Stop the development projects that were already running."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restore those projects and allow active exceptions to propagate."""
        ...


class JourneyRunner(Protocol):
    """Expose the pure verdict pipeline as an injectable service."""

    def __call__(  # noqa: PLR0913
        self,
        journeys: Sequence[Journey],
        *,
        stack: JourneyStack,
        executor: JourneyExecutor,
        judge: Judge,
        base_url: str,
        credentials: OperatorCredentials,
        run_dir: Path,
    ) -> tuple[JourneyRun, ...]:
        """Run the supplied Journeys and return their terminal records."""
        ...


@dataclass(frozen=True, slots=True)
class JourneyServices:
    """Provide the runtime adapters used by one CLI Journey invocation."""

    stack: ShutdownJourneyStack
    guard: JourneyRunGuard
    executor: JourneyExecutor
    judge: Judge
    base_url: str
    credentials: OperatorCredentials
    prepare_fixtures: Callable[[], None]
    run_journeys: JourneyRunner


app = typer.Typer(
    name="journeys",
    help="Run and report natural-language Journeys against the CI stack.",
    no_args_is_help=True,
)


def create_journey_services(
    repository_root: Path,
    env_path: Path,
    run_dir: Path,
) -> JourneyServices:
    """Build the real CI adapters behind the CLI's injectable factory seam."""
    del run_dir
    environment = load_ci_environment(env_path)
    project_names = _development_project_names(env_path)
    guard = DevStackGuard(run_command, project_names, repository_root)
    cookie_jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))
    http_setup = HttpSetup(opener=opener, base_url=environment.base_url)
    stack = ComposeJourneyStack(
        run_command,
        repository_root,
        environment,
        http_setup,
        env_file=env_path,
    )
    executor = ClaudeCliExecutor(
        runner=run_command,
        claude_binary="claude",
        node_binary=shutil.which("node") or "node",
        playwright_cli_path=repository_root / _PLAYWRIGHT_CLI_RELATIVE_PATH,
    )
    judge = ClaudeCliJudge(run_command, "claude")

    def prepare_fixtures() -> None:
        ensure_fixture_videos(
            repository_root / "assets/test-streams.json",
            repository_root,
            download=download_fixture,
        )

    return JourneyServices(
        stack=stack,
        guard=guard,
        executor=executor,
        judge=judge,
        base_url=environment.base_url,
        credentials=environment.credentials,
        prepare_fixtures=prepare_fixtures,
        run_journeys=run_journeys,
    )


def create_github_client(repository: str) -> GitHubClient:
    """Build the file-backed GitHub client used by publish and close commands."""
    return GitHubClient(run_command, repository)


@app.command()
def run(  # noqa: PLR0913
    pr: Annotated[int, _PR_OPTION],
    head_sha: Annotated[str, _HEAD_SHA_OPTION],
    run_url: Annotated[str, _RUN_URL_OPTION],
    run_dir: Annotated[Path, _RUN_DIR_OPTION],
    repository: Annotated[str, _REPOSITORY_OPTION] = "",
    journeys_dir: Annotated[Path, _JOURNEYS_DIR_OPTION] = _DEFAULT_JOURNEYS_DIR,
    env_file: Annotated[Path, _ENV_FILE_OPTION] = _DEFAULT_ENV_FILE,
    publish: Annotated[bool, _PUBLISH_OPTION] = False,
) -> None:
    """Run the Journey catalog and save machine-readable results."""
    if pr < 1:
        typer.echo("pull request number must be positive", err=True)
        raise typer.Exit(code=2)
    repository_root = _REPOSITORY_ROOT
    resolved_env_path = _resolve(repository_root, env_file)
    resolved_journeys_dir = _resolve(repository_root, journeys_dir)
    resolved_run_dir = run_dir.expanduser().resolve()
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    try:
        services = create_journey_services(
            repository_root,
            resolved_env_path,
            resolved_run_dir,
        )
    except Exception as error:
        failure = _failure_detail(error)
        payload = _results_payload((), harness_failed=True, failure=failure)
        _write_results(resolved_run_dir, payload)
        typer.echo(f"Journey harness failed: {failure}", err=True)
        raise typer.Exit(code=1) from error

    runs, journeys, failure = _execute_services(
        services,
        resolved_journeys_dir,
        resolved_run_dir,
    )
    harness_failed = failure is not None
    if publish:
        repository_name = repository or os.environ.get("GITHUB_REPOSITORY")
        if not repository_name:
            failure = failure or "RepositoryRequired"
            harness_failed = True
        else:
            context = RunContext(repository_name, pr, head_sha, run_url)
            failure = _publish(
                context,
                runs,
                journeys,
                failure=failure,
                credentials=services.credentials,
            )
            harness_failed = failure is not None

    payload = _results_payload(runs, harness_failed=harness_failed, failure=failure)
    _write_results(resolved_run_dir, payload)
    typer.echo(json.dumps(payload, sort_keys=True))
    if harness_failed:
        typer.echo(f"Journey harness failed: {failure}", err=True)
        raise typer.Exit(code=1)


@app.command("close-pr-issues")
def close_pr_issues_command(
    pr: Annotated[int, _CLOSE_PR_OPTION],
    repository: Annotated[str, _CLOSE_REPOSITORY_OPTION],
) -> None:
    """Close marked Bug Report Issues when a pull request is not merged."""
    client = create_github_client(repository)
    close_pr_issues(client, pr)


def _execute_services(
    services: JourneyServices,
    journeys_dir: Path,
    run_dir: Path,
) -> tuple[tuple[JourneyRun, ...], tuple[Journey, ...], str | None]:
    journeys: tuple[Journey, ...] = ()
    runs: tuple[JourneyRun, ...] = ()
    pipeline_failure: str | None = None
    shutdown_failure: str | None = None
    shutdown_attempted = False
    try:
        with services.guard:
            try:
                journeys = load_catalog(journeys_dir)
                services.prepare_fixtures()
                runs = services.run_journeys(
                    journeys,
                    stack=services.stack,
                    executor=services.executor,
                    judge=services.judge,
                    base_url=services.base_url,
                    credentials=services.credentials,
                    run_dir=run_dir,
                )
            finally:
                shutdown_attempted = True
                shutdown_failure = _shutdown_stack(services)
    except Exception as error:  # noqa: BLE001
        pipeline_failure = _failure_detail(error, services.credentials)
    finally:
        if not shutdown_attempted:
            shutdown_failure = _shutdown_stack(services)
    return runs, journeys, pipeline_failure or shutdown_failure


def _shutdown_stack(services: JourneyServices) -> str | None:
    try:
        services.stack.shutdown()
    except Exception as error:  # noqa: BLE001
        return _failure_detail(error, services.credentials)
    return None


def _publish(
    context: RunContext,
    runs: Sequence[JourneyRun],
    journeys: Sequence[Journey],
    *,
    failure: str | None,
    credentials: OperatorCredentials,
) -> str | None:
    harness_failed = failure is not None
    client = create_github_client(context.repository)
    try:
        publish_results(runs, journeys, context, client)
    except Exception as error:  # noqa: BLE001
        failure = failure or _failure_detail(error, credentials)
        harness_failed = True
    try:
        client.create_check_run(
            context.head_sha,
            check_conclusion(runs, harness_failed),
            _check_summary(runs, failure),
        )
    except Exception as error:  # noqa: BLE001
        failure = failure or _failure_detail(error, credentials)
    return failure


def _failure_detail(
    error: Exception,
    credentials: OperatorCredentials | None = None,
) -> str:
    """Format safe harness detail without exposing the configured operator password."""
    error_type = type(error).__name__
    if not isinstance(error, StackError) and credentials is None:
        return error_type
    message = str(error)
    if credentials is not None:
        password = credentials.password.get_secret_value()
        if password and password in message:
            return error_type
    return f"{error_type}: {message}"


def _development_project_names(env_path: Path) -> tuple[str, ...]:
    configured = os.environ.get("GW_JOURNEY_DEV_PROJECTS")
    if configured is None:
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                key, separator, value = line.partition("=")
                if separator and key.strip() == "GW_JOURNEY_DEV_PROJECTS":
                    configured = value.strip().strip("'\"")
                    break
        except OSError:
            configured = None
    selected = configured or "gods-watching"
    return tuple(name.strip() for name in selected.split(",") if name.strip())


def download_fixture(url: str, destination: Path) -> None:
    """Download a fixture video to a local destination.

    Args:
        url: The HTTP URL of the fixture video.
        destination: The local path where the video will be saved.
    """
    request = Request(  # noqa: S310
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) gods-watching-journeys"},
    )
    response = cast("HTTPResponse", urlopen(request, timeout=60))  # noqa: S310
    with response, destination.open("wb") as fixture_file:
        while chunk := response.read(1024 * 1024):
            _ = fixture_file.write(chunk)


def _resolve(repository_root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else repository_root / expanded


def _results_payload(
    runs: Sequence[JourneyRun],
    *,
    harness_failed: bool,
    failure: str | None,
) -> dict[str, object]:
    serialized_runs = [cast("object", run.model_dump(mode="json")) for run in runs]
    return {"harness_failed": harness_failed, "failure": failure, "runs": serialized_runs}


def _write_results(run_dir: Path, payload: dict[str, object]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _check_summary(runs: Sequence[JourneyRun], failure: str | None) -> str:
    if failure is not None:
        return f"Journey harness failed: {failure}."
    bug_count = sum(run.bug_report is not None for run in runs)
    inconclusive_count = sum(run.verdict == "inconclusive" for run in runs)
    return f"{bug_count} confirmed Bug Reports; {inconclusive_count} inconclusive Journeys."


__all__ = [
    "JourneyServices",
    "app",
    "close_pr_issues_command",
    "create_github_client",
    "create_journey_services",
    "download_fixture",
    "run",
]
