"""Isolation and Compose command tests using recording fakes only."""

import json
from collections.abc import Sequence
from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import SecretStr

from gods_watching.journeys import stack as stack_module
from gods_watching.journeys.agents import CompletedCommand
from gods_watching.journeys.models import Journey, OperatorCredentials, SetupStep, StackError
from gods_watching.journeys.stack import (
    CiEnvironment,
    ComposeJourneyStack,
    DevStackGuard,
    load_ci_environment,
)


class RecordingRunner:
    """Return configured Compose state while preserving every argument vector."""

    def __init__(self, *, projects: list[dict[str, str]] | None = None) -> None:
        self.projects: list[dict[str, str]] = projects or []
        self.commands: list[tuple[tuple[str, ...], Path]] = []

    def __call__(self, command: Sequence[str], cwd: Path) -> CompletedCommand:
        arguments = tuple(command)
        self.commands.append((arguments, cwd))
        if arguments == ("docker", "compose", "ls", "--format", "json"):
            return CompletedCommand(0, json.dumps(self.projects), "")
        if "logs" in arguments:
            return CompletedCommand(0, "gateway started\n", "")
        return CompletedCommand(0, "", "")


class RecordingHttp:
    """Record the public setup actions and expose a scripted gateway status."""

    def __init__(self, *, session_status: int | None = 401) -> None:
        self.session_status: int | None = session_status
        self.actions: list[tuple[object, ...]] = []

    def probe_session(self) -> int | None:
        self.actions.append(("probe_session",))
        return self.session_status

    def login(self, credentials: OperatorCredentials) -> None:
        self.actions.append(("login", credentials.username))

    def register_fixture_cameras(self, rtsp_port: int) -> tuple[str, ...]:
        self.actions.append(("register_fixture_cameras", rtsp_port))
        return ("camera-id-1", "camera-id-2", "camera-id-3", "camera-id-4")

    def wait_cameras_streaming(
        self,
        camera_ids: Sequence[str],
        timeout_s: float = 180,
    ) -> None:
        self.actions.append(("wait_cameras_streaming", tuple(camera_ids), timeout_s))


def sample_journey(setup: tuple[SetupStep, ...] = ("login", "cameras")) -> Journey:
    """Return the minimal Journey setup declaration needed by Compose tests."""
    return Journey(
        id="live-cameras",
        title="Monitor fixture cameras",
        setup=setup,
        tags=(),
        preconditions="Fixture cameras are available.",
        goal="Confirm that the camera sources can be monitored.",
        expected_outcomes=(),
        source_path=Path("live-cameras.md"),
    )


def ci_environment() -> CiEnvironment:
    """Return a valid CI stack configuration without reading any local secrets."""
    return CiEnvironment(
        base_url="http://127.0.0.1:18080",
        credentials=OperatorCredentials(
            username="fixture-operator",
            password=SecretStr("fixture-secret-value"),
        ),
        fixture_rtsp_port=38554,
    )


@pytest.mark.parametrize("raises", [False, True], ids=("normal-exit", "exception-exit"))
def test_dev_stack_guard_stops_and_restarts_only_running_requested_projects(
    tmp_path: Path,
    raises: bool,
) -> None:
    """The guard restores the exact development projects that were running."""
    runner = RecordingRunner(
        projects=[
            {"Name": "gods-watching", "Status": "running(4)"},
            {"Name": "stopped-project", "Status": "exited(0)"},
            {"Name": "other-running", "Status": "running(1)"},
        ]
    )
    guard = DevStackGuard(
        runner=runner,
        project_names=("gods-watching", "stopped-project", "other-running"),
        repository_root=tmp_path,
    )

    if raises:
        failure_message = "journey failed"
        with pytest.raises(RuntimeError, match="journey failed"), guard:
            raise RuntimeError(failure_message)
    else:
        with guard:
            pass

    assert [command for command, _cwd in runner.commands] == [
        ("docker", "compose", "ls", "--format", "json"),
        ("docker", "compose", "-p", "gods-watching", "stop"),
        ("docker", "compose", "-p", "other-running", "stop"),
        ("docker", "compose", "-p", "gods-watching", "start"),
        ("docker", "compose", "-p", "other-running", "start"),
    ]


def test_dev_stack_guard_does_not_restart_projects_that_were_already_stopped(
    tmp_path: Path,
) -> None:
    """A listed project absent from the running set stays stopped."""
    runner = RecordingRunner(projects=[])

    with DevStackGuard(runner, ("gods-watching",), tmp_path):
        pass

    assert [command for command, _cwd in runner.commands] == [
        ("docker", "compose", "ls", "--format", "json")
    ]


@pytest.mark.parametrize("project_name", ["gw-ci", "gw-ci-fixtures"])
def test_dev_stack_guard_refuses_ci_projects(tmp_path: Path, project_name: str) -> None:
    """The development guard cannot stop or restart a CI Compose project."""
    runner = RecordingRunner()

    with pytest.raises(StackError, match="CI project"):
        _ = DevStackGuard(runner, (project_name,), tmp_path)

    assert runner.commands == []


def test_compose_stack_prepares_ci_projects_and_runs_declared_setup_in_order(
    tmp_path: Path,
) -> None:
    """Journey preparation uses only the two CI project names and setup order."""
    runner = RecordingRunner()
    http = RecordingHttp(session_status=401)
    env_file = tmp_path / ".env"
    stack = ComposeJourneyStack(
        runner,
        tmp_path,
        ci_environment(),
        http,
        env_file=env_file,
    )

    stack.prepare_journey(sample_journey())

    assert [command for command, _cwd in runner.commands] == [
        (
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-p",
            "gw-ci",
            "down",
            "--volumes",
            "--remove-orphans",
        ),
        (
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-p",
            "gw-ci-fixtures",
            "-f",
            "deploy/compose.fixtures.yaml",
            "--profile",
            "fixtures",
            "up",
            "-d",
        ),
        ("docker", "compose", "--env-file", str(env_file), "-p", "gw-ci", "up", "-d"),
    ]
    assert all(
        command[command.index("-p") + 1] in {"gw-ci", "gw-ci-fixtures"}
        for command, _cwd in runner.commands
    )
    assert http.actions == [
        ("probe_session",),
        ("login", "fixture-operator"),
        ("register_fixture_cameras", 38554),
        (
            "wait_cameras_streaming",
            ("camera-id-1", "camera-id-2", "camera-id-3", "camera-id-4"),
            180,
        ),
    ]


def test_compose_stack_times_out_when_gateway_never_becomes_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unhealthy session endpoint ends the bounded gateway readiness wait."""
    clock = iter((0.0, 601.0))
    monkeypatch.setattr(stack_module, "monotonic", lambda: next(clock))
    runner = RecordingRunner()
    http = RecordingHttp(session_status=503)
    stack = ComposeJourneyStack(
        runner,
        tmp_path,
        ci_environment(),
        http,
        env_file=tmp_path / ".env",
    )

    with pytest.raises(StackError, match="gateway did not become ready"):
        stack.prepare_journey(sample_journey(setup=()))

    assert len(runner.commands) == 3
    assert http.actions == [("probe_session",)]


def test_compose_stack_collects_logs_and_shutdown_targets_only_ci_projects(
    tmp_path: Path,
) -> None:
    """Evidence collection and teardown remain bounded to CI Compose projects."""
    runner = RecordingRunner()
    env_file = tmp_path / ".env"
    stack = ComposeJourneyStack(
        runner,
        tmp_path,
        ci_environment(),
        RecordingHttp(),
        env_file=env_file,
    )
    evidence_dir = tmp_path / "evidence"

    stack.collect_evidence(evidence_dir)
    stack.shutdown()

    assert (evidence_dir / "compose.log").read_text(encoding="utf-8") == "gateway started\n"
    assert [command for command, _cwd in runner.commands] == [
        (
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-p",
            "gw-ci",
            "logs",
            "--no-color",
            "--timestamps",
        ),
        ("docker", "compose", "--env-file", str(env_file), "-p", "gw-ci", "down", "--volumes"),
        (
            "docker",
            "compose",
            "--env-file",
            str(env_file),
            "-p",
            "gw-ci-fixtures",
            "down",
            "--volumes",
        ),
    ]


def test_ci_environment_requires_the_gw_ci_project_and_reads_expected_fields(
    tmp_path: Path,
) -> None:
    """The runner env file provides only the declared CI origin and credentials."""
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
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

    environment = load_ci_environment(env_path)

    assert environment.base_url == "http://localhost:18080"
    assert environment.credentials.username == "fixture-operator"
    assert environment.credentials.password.get_secret_value() == "fixture-secret-value"
    assert environment.fixture_rtsp_port == 38554


def test_ci_environment_rejects_a_non_ci_compose_project(tmp_path: Path) -> None:
    """The stack config cannot direct teardown at a development project."""
    env_path = tmp_path / ".env"
    _ = env_path.write_text(
        dedent(
            """\
            COMPOSE_PROJECT_NAME=gods-watching
            GW_PUBLIC_ORIGIN=http://localhost:18080
            GW_OPERATOR_USERNAME=admin
            GW_OPERATOR_PASSWORD=fixture-secret-value
            GW_FIXTURE_RTSP_PORT=38554
            """
        ),
        encoding="utf-8",
    )

    with pytest.raises(StackError, match="COMPOSE_PROJECT_NAME must be gw-ci"):
        _ = load_ci_environment(env_path)
