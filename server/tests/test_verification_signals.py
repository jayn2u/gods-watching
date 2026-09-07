import errno
import json
import os
import pty
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

REPOSITORY_ROOT = Path(__file__).parents[2]
CLI = REPOSITORY_ROOT / "gods-watching"


@dataclass(frozen=True, slots=True)
class ManifestEvent:
    name: str
    state: str
    detail: str


def test_verify_cli_normal_completion_is_unchanged(tmp_path: Path) -> None:
    completed = subprocess.run(  # noqa: S603
        (
            str(CLI),
            "verify",
            "--scenario",
            "harness-self-check",
            "--evidence",
            str(tmp_path / "normal"),
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 0
    assert json.loads(completed.stdout)["outcome"] == "passed"


def test_verify_cli_signal_callback_cannot_publish_success(tmp_path: Path) -> None:
    script = "\n".join(
        (
            "import asyncio",
            "import os",
            "import signal",
            "import sys",
            "from pathlib import Path",
            "",
            "import anyio",
            "import click",
            "",
            f"sys.path.insert(0, str(Path({str(REPOSITORY_ROOT)!r}) / 'server' / 'src'))",
            "import gods_watching.cli as cli",
            "",
            "class FakeResult:",
            "    exit_code = 0",
            "",
            "    def model_dump_json(self):",
            '        return \'{"outcome":"passed"}\'',
            "",
            "async def run_fake_scenario():",
            "    asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)",
            "    await anyio.sleep(0)",
            "    await anyio.sleep(0.05)",
            "    return FakeResult()",
            "",
            "def fake_execute_scenario(**kwargs):",
            "    del kwargs",
            "    return anyio.run(run_fake_scenario)",
            "",
            "cli.execute_scenario = fake_execute_scenario",
            "try:",
            f"    cli.verify('harness-self-check', Path({str(tmp_path / 'callback-race')!r}))",
            "except click.exceptions.Exit as error:",
            "    raise SystemExit(error.exit_code)",
        )
    )
    completed = subprocess.run(  # noqa: S603
        (sys.executable, "-c", script),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 143
    assert completed.stdout == ""
    assert "Exception in callback" not in completed.stderr
    assert "VerificationTerminated" not in completed.stderr


def test_launcher_pty_propagates_child_exit_status(tmp_path: Path) -> None:
    capture = tmp_path / "launcher.typescript"
    command = f"{shlex.quote(str(CLI))} invalid-command"
    completed = subprocess.run(  # noqa: S603
        ("/usr/bin/script", "-qefc", command, str(capture)),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 2
    assert "No such command" in capture.read_text(encoding="utf-8")


def test_verify_cli_timeout_cleans_owned_resources_immediately(tmp_path: Path) -> None:
    evidence = tmp_path / "launcher-timeout"
    completed = subprocess.run(  # noqa: S603
        (
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=5s",
            "2s",
            str(CLI),
            "verify",
            "--scenario",
            "media",
            "--evidence",
            str(evidence),
        ),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert completed.returncode == 124
    container_name = _media_gateway_container_name(evidence / "resource-manifest.json")
    _assert_cleanup(evidence, container_name)
    _assert_gateway_lifecycle(evidence / "resource-manifest.json")


def test_verify_cli_sigterm_between_registration_and_start_cleans_owned_resources(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "registered-before-started"
    process = _start_media_verification(evidence)
    container_name = ""
    try:
        container_name = _wait_for_media_gateway_registration(
            evidence / "resource-manifest.json",
            process,
            require_started=False,
        )
        _terminate_with_sigterm(process)

        assert process.returncode == 143
        _assert_cleanup(evidence, container_name)
        _assert_gateway_lifecycle(evidence / "resource-manifest.json")
    finally:
        _cleanup_test_process(process, container_name, evidence / "runtime")


def test_verify_cli_sigterm_after_gateway_start_cleans_owned_resources(tmp_path: Path) -> None:
    evidence = tmp_path / "terminated"
    process = _start_media_verification(evidence)
    container_name = ""
    try:
        container_name = _wait_for_media_gateway_registration(
            evidence / "resource-manifest.json",
            process,
            require_started=True,
        )
        _terminate_with_sigterm(process)

        assert process.returncode == 143
        _assert_cleanup(evidence, container_name)
        _assert_gateway_lifecycle(evidence / "resource-manifest.json")
    finally:
        _cleanup_test_process(process, container_name, evidence / "runtime")


def test_verify_cli_repeated_sigterm_during_cleanup_is_cooperative(tmp_path: Path) -> None:
    evidence = tmp_path / "repeated-terminated"
    process = _start_media_verification(evidence)
    container_name = ""
    try:
        container_name = _wait_for_media_gateway_registration(
            evidence / "resource-manifest.json",
            process,
            require_started=True,
        )
        _terminate_with_repeated_sigterm(process)

        assert process.returncode == 143
        _assert_cleanup(evidence, container_name)
        _assert_gateway_lifecycle(evidence / "resource-manifest.json")
    finally:
        _cleanup_test_process(process, container_name, evidence / "runtime")


def test_foreground_pty_ctrl_c_waits_for_cleanup_and_returns_nonzero(
    tmp_path: Path,
) -> None:
    for attempt in range(2):
        evidence = tmp_path / f"foreground-pty-ctrl-c-{attempt}"
        returncode, manifest_path = _run_foreground_pty_ctrl_c(evidence, tmp_path / "uv-bin")

        assert returncode == 130
        _assert_cleanup(evidence, _media_gateway_container_name(manifest_path))
        _assert_gateway_lifecycle(manifest_path)


def _run_foreground_pty_ctrl_c(  # noqa: C901
    evidence: Path, uv_bin: Path
) -> tuple[int, Path]:
    command = (
        str(CLI),
        "verify",
        "--scenario",
        "media",
        "--evidence",
        str(evidence),
    )
    uv_bin.mkdir(exist_ok=True)
    uv_shim = uv_bin / "uv"
    uv_path = next(Path("/snap/astral-uv").glob("*/bin/uv"), Path(shutil.which("uv") or "uv"))
    _ = uv_shim.write_text(f'#!/bin/sh\nexec {shlex.quote(str(uv_path))} "$@"\n', encoding="utf-8")
    _ = uv_shim.chmod(0o700)
    pid, master_fd = pty.fork()
    if pid == 0:
        environment = os.environ.copy()
        _ = environment.pop("VIRTUAL_ENV", None)
        environment["PATH"] = f"{uv_bin}{os.pathsep}{environment['PATH']}"
        os.execvpe(command[0], command, environment)  # noqa: S606

    output = bytearray()
    deadline = time.monotonic() + 30
    signal_sent = False
    status: int | None = None
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.01)
            if ready:
                try:
                    output.extend(os.read(master_fd, 65536))
                except OSError as error:
                    if error.errno != errno.EIO:
                        raise
                    break
            if not signal_sent and _gateway_started(evidence / "resource-manifest.json"):
                _ = os.write(master_fd, b"\x03")
                signal_sent = True
            if signal_sent:
                waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
                if waited_pid == pid:
                    status = waited_status
                    break
        if not signal_sent:
            message = "foreground PTY scenario did not start its media gateway"
            raise AssertionError(message)
        if status is None:
            _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status), evidence / "resource-manifest.json"
    finally:
        if status is None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                _ = os.waitpid(pid, 0)
        with suppress(OSError):
            os.close(master_fd)


def _gateway_started(manifest_path: Path) -> bool:
    if not manifest_path.exists():
        return False
    return any(event.state == "started" for event in _gateway_events(manifest_path))


def _start_media_verification(evidence: Path) -> subprocess.Popen[str]:
    return subprocess.Popen(  # noqa: S603
        (
            str(CLI),
            "verify",
            "--scenario",
            "media",
            "--evidence",
            str(evidence),
        ),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _terminate_with_sigterm(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        os.kill(process.pid, signal.SIGSTOP)
        os.kill(process.pid, signal.SIGTERM)
        os.kill(process.pid, signal.SIGCONT)
    _ = process.communicate(timeout=20)


def _terminate_with_repeated_sigterm(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        os.kill(process.pid, signal.SIGCONT)
        for _ in range(3):
            os.kill(process.pid, signal.SIGTERM)
            time.sleep(0.025)
    _ = process.communicate(timeout=20)


def _assert_cleanup(evidence: Path, container_name: str) -> None:
    assert not (evidence / "runtime").exists()
    assert not _container_exists(container_name)


def _assert_gateway_lifecycle(manifest_path: Path) -> None:
    events = _gateway_events(manifest_path)
    states = [event.state for event in events]
    assert states == ["registered", "started", "cleaned"]


def _cleanup_test_process(
    process: subprocess.Popen[str], container_name: str, runtime: Path
) -> None:
    if process.poll() is None:
        process.kill()
    _ = process.communicate(timeout=10)
    if container_name and _container_exists(container_name):
        _ = subprocess.run(  # noqa: S603
            ("/usr/bin/docker", "container", "stop", "--timeout", "2", container_name),
            check=False,
            capture_output=True,
        )
        _ = subprocess.run(  # noqa: S603
            ("/usr/bin/docker", "container", "remove", container_name),
            check=False,
            capture_output=True,
        )
    if runtime.exists():
        shutil.rmtree(runtime)


def _wait_for_media_gateway_registration(
    manifest_path: Path,
    process: subprocess.Popen[str],
    *,
    require_started: bool,
) -> str:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if manifest_path.exists():
            gateway_events = _gateway_events(manifest_path)
            registered = next(
                (event for event in gateway_events if event.state == "registered"), None
            )
            started = any(event.state == "started" for event in gateway_events)
            if registered is not None:
                parts = registered.detail.split()
                container_name = parts[parts.index("--name") + 1]
                if not require_started and not started:
                    os.kill(process.pid, signal.SIGSTOP)
                    return container_name
                if require_started and started and _container_exists(container_name):
                    os.kill(process.pid, signal.SIGSTOP)
                    return container_name
        time.sleep(0.001)
    message = "media gateway was not registered before deadline"
    raise AssertionError(message)


def _gateway_events(manifest_path: Path) -> list[ManifestEvent]:
    raw = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    raw_events = raw.get("events")
    assert isinstance(raw_events, list)
    events: list[ManifestEvent] = []
    for raw_event in cast("list[object]", raw_events):
        assert isinstance(raw_event, dict)
        raw_event = cast("dict[str, object]", raw_event)
        name = raw_event.get("name")
        state = raw_event.get("state")
        detail = raw_event.get("detail")
        assert isinstance(name, str)
        assert isinstance(state, str)
        assert isinstance(detail, str)
        if name == "media-gateway":
            events.append(ManifestEvent(name=name, state=state, detail=detail))
    return events


def _media_gateway_container_name(manifest_path: Path) -> str:
    registered = next(
        event for event in _gateway_events(manifest_path) if event.state == "registered"
    )
    parts = registered.detail.split()
    return parts[parts.index("--name") + 1]


def _container_exists(name: str) -> bool:
    inspected = subprocess.run(  # noqa: S603
        ("/usr/bin/docker", "container", "inspect", name),
        check=False,
        capture_output=True,
    )
    return inspected.returncode == 0
