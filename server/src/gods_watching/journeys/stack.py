"""Guard the development stack and control isolated Journey Compose projects."""

from __future__ import annotations

import json
from dataclasses import dataclass
from time import monotonic, sleep
from typing import TYPE_CHECKING, Final, Protocol, Self, TypeGuard, cast
from urllib.parse import urlsplit

from pydantic import SecretStr

from .models import OperatorCredentials, StackError, raise_stack_error

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from .agents import CommandRunner, CompletedCommand
    from .models import Journey


class JourneySetupClient(Protocol):
    """Expose the HTTP setup operations needed by a Compose Journey stack."""

    def probe_session(self) -> int | None:
        """Return the status of the session readiness endpoint."""
        ...

    def login(self, credentials: OperatorCredentials) -> None:
        """Create a session for the configured operator."""
        ...

    def register_fixture_cameras(self, rtsp_port: int) -> tuple[str, ...]:
        """Register all fixture camera sources and return their IDs."""
        ...

    def wait_cameras_streaming(
        self,
        camera_ids: tuple[str, ...],
        timeout_s: float = 180,
    ) -> None:
        """Wait until the given camera IDs are online."""
        ...

_CI_PROJECTS: Final = frozenset(("gw-ci", "gw-ci-fixtures"))
_GATEWAY_TIMEOUT_S: Final = 600.0
_GATEWAY_POLL_INTERVAL_S: Final = 2.0
_MAX_TCP_PORT: Final = 65535
_MIN_TCP_PORT: Final = 1
_MIN_ENV_QUOTE_LENGTH: Final = 2


@dataclass(frozen=True, slots=True)
class CiEnvironment:
    """Hold the public origin, operator credentials, and fixture RTSP port."""

    base_url: str
    credentials: OperatorCredentials
    fixture_rtsp_port: int


class DevStackGuard:
    """Stop running development projects temporarily and restart their prior set."""

    _runner: CommandRunner
    _project_names: tuple[str, ...]
    _repository_root: Path

    def __init__(
        self,
        runner: CommandRunner,
        project_names: tuple[str, ...],
        repository_root: Path,
    ) -> None:
        """Create a guard for the specified development Compose projects."""
        if any(name in _CI_PROJECTS for name in project_names):
            raise_stack_error("development stack guard cannot target a CI project")
        self._runner = runner
        self._project_names = tuple(dict.fromkeys(project_names))
        self._repository_root = repository_root
        self._stopped: list[str] = []

    def __enter__(self) -> Self:
        """Stop only the listed development projects that are currently running."""
        running_projects = self._running_projects()
        for name in self._project_names:
            if name not in running_projects:
                continue
            self._stopped.append(name)
            try:
                result = self._runner(
                    ("docker", "compose", "-p", name, "stop"),
                    self._repository_root,
                )
            except OSError as error:
                _ = self._restore_stopped()
                raise_stack_error("development Compose project could not be stopped", cause=error)
            if result.returncode != 0:
                _ = self._restore_stopped()
                raise_stack_error("development Compose project stop command failed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Restart the stopped projects and let the original exception propagate."""
        del exc_value, traceback
        restore_error = self._restore_stopped()
        if restore_error is not None and exc_type is None:
            raise_stack_error(restore_error)
        return False

    def _running_projects(self) -> set[str]:
        try:
            result = self._runner(
                ("docker", "compose", "ls", "--format", "json"),
                self._repository_root,
            )
        except OSError as error:
            raise_stack_error("Docker Compose project list could not be read", cause=error)
        if result.returncode != 0:
            raise_stack_error("Docker Compose project list command failed")
        try:
            decoded = cast("object", json.loads(result.stdout))
        except json.JSONDecodeError as error:
            raise_stack_error("Docker Compose returned invalid project JSON", cause=error)
        if not _is_json_array(decoded):
            raise_stack_error("Docker Compose project list must be a JSON array")
        running: set[str] = set()
        for item in decoded:
            if not _is_json_object(item):
                continue
            name = item.get("Name")
            status = item.get("Status")
            if isinstance(name, str) and isinstance(status, str) and status.startswith("running"):
                running.add(name)
        return running

    def _restore_stopped(self) -> str | None:
        first_error: str | None = None
        for name in self._stopped:
            try:
                result = self._runner(
                    ("docker", "compose", "-p", name, "start"),
                    self._repository_root,
                )
            except OSError:
                if first_error is None:
                    first_error = "development Compose project could not be restarted"
                continue
            if result.returncode != 0 and first_error is None:
                first_error = "development Compose project start command failed"
        self._stopped.clear()
        return first_error


@dataclass(frozen=True, slots=True)
class ComposeJourneyStack:
    """Reset and prepare a Journey using only the dedicated CI Compose projects."""

    runner: CommandRunner
    repository_root: Path
    env: CiEnvironment
    http: JourneySetupClient
    env_file: Path

    def prepare_journey(self, journey: Journey) -> None:
        """Reset both CI projects, wait for the gateway, and establish setup steps."""
        _ = self._run_compose("-p", "gw-ci", "down", "--volumes", "--remove-orphans")
        _ = self._run_compose(
            "-p",
            "gw-ci-fixtures",
            "-f",
            "deploy/compose.fixtures.yaml",
            "--profile",
            "fixtures",
            "up",
            "-d",
        )
        _ = self._run_compose("-p", "gw-ci", "up", "-d")
        self._wait_for_gateway()
        for setup_step in journey.setup:
            if setup_step == "login":
                self.http.login(self.env.credentials)
            elif setup_step == "cameras":
                camera_ids = self.http.register_fixture_cameras(self.env.fixture_rtsp_port)
                self.http.wait_cameras_streaming(camera_ids)

    def collect_evidence(self, evidence_dir: Path) -> None:
        """Write timestamped application Compose logs into the attempt directory."""
        result = self._run_compose(
            "-p",
            "gw-ci",
            "logs",
            "--no-color",
            "--timestamps",
        )
        evidence_dir.mkdir(parents=True, exist_ok=True)
        try:
            _ = (evidence_dir / "compose.log").write_text(result.stdout, encoding="utf-8")
        except OSError as error:
            raise_stack_error("Compose evidence log could not be written", cause=error)

    def shutdown(self) -> None:
        """Tear down and remove volumes from both isolated CI projects."""
        first_error: str | None = None
        for project_name in ("gw-ci", "gw-ci-fixtures"):
            try:
                _ = self._run_compose("-p", project_name, "down", "--volumes")
            except StackError as error:
                if first_error is None:
                    first_error = str(error)
        if first_error is not None:
            raise_stack_error(first_error)

    def _wait_for_gateway(self) -> None:
        deadline = monotonic() + _GATEWAY_TIMEOUT_S
        while True:
            if self.http.probe_session() in {200, 401}:
                return
            if monotonic() >= deadline:
                raise_stack_error("gateway did not become ready within 600 seconds")
            sleep(_GATEWAY_POLL_INTERVAL_S)

    def _run_compose(self, *arguments: str) -> CompletedCommand:
        command = ("docker", "compose", "--env-file", str(self.env_file), *arguments)
        try:
            result = self.runner(command, self.repository_root)
        except OSError as error:
            raise_stack_error("CI Docker Compose command could not run", cause=error)
        if result.returncode != 0:
            raise_stack_error("CI Docker Compose command failed")
        return result


def load_ci_environment(env_path: Path) -> CiEnvironment:
    """Parse a CI .env file and reject any Compose project other than gw-ci."""
    try:
        values = _parse_env(env_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as error:
        raise_stack_error("Journey CI environment file cannot be read", cause=error)
    if values.get("COMPOSE_PROJECT_NAME") != "gw-ci":
        raise_stack_error("COMPOSE_PROJECT_NAME must be gw-ci")
    base_url = _required_value(values, "GW_PUBLIC_ORIGIN")
    parsed_origin = urlsplit(base_url)
    if (
        parsed_origin.scheme not in {"http", "https"}
        or not parsed_origin.netloc
        or parsed_origin.path not in {"", "/"}
    ):
        raise_stack_error("GW_PUBLIC_ORIGIN must be an absolute origin")
    username = _required_value(values, "GW_OPERATOR_USERNAME")
    password = _required_value(values, "GW_OPERATOR_PASSWORD")
    port_value = _required_value(values, "GW_FIXTURE_RTSP_PORT")
    try:
        fixture_rtsp_port = int(port_value)
    except ValueError as error:
        raise_stack_error("GW_FIXTURE_RTSP_PORT must be an integer", cause=error)
    if not _MIN_TCP_PORT <= fixture_rtsp_port <= _MAX_TCP_PORT:
        raise_stack_error("GW_FIXTURE_RTSP_PORT must be between 1 and 65535")
    return CiEnvironment(
        base_url=base_url.rstrip("/"),
        credentials=OperatorCredentials(username=username, password=SecretStr(password)),
        fixture_rtsp_port=fixture_rtsp_port,
    )


def _parse_env(contents: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in contents.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _separator, raw_value = stripped.partition("=")
        value = raw_value.strip()
        if (
            len(value) >= _MIN_ENV_QUOTE_LENGTH
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]
        values[key.strip()] = value
    return values


def _required_value(values: dict[str, str], key: str) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise_stack_error(f"{key} is required by Journey CI")
    return value


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


__all__ = ["CiEnvironment", "ComposeJourneyStack", "DevStackGuard", "load_ci_environment"]
