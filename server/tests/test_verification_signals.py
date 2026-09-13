import copy
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
from typing import cast, final

import anyio
import pytest

from gods_watching.verification.context import (
    ResourceLedger,
    ScenarioContext,
    ScenarioContextConfig,
    VerificationInterruptedError,
)
from gods_watching.verification.models import RunId
from gods_watching.verification.scenarios import task09_stack

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
    manifest_path = evidence / "resource-manifest.json"
    gateway_events = _gateway_events(manifest_path)
    container_name = (
        _media_gateway_container_name(manifest_path)
        if gateway_events
        else _gateway_cleanup_container_name(evidence / "task-9-gateway-cleanup.json")
    )
    _assert_cleanup(evidence, container_name, expected_signum=signal.SIGTERM)
    if gateway_events:
        _assert_gateway_lifecycle(manifest_path)


def test_pre_process_interruption_is_recorded_before_resource_start(tmp_path: Path) -> None:
    manifest_path = tmp_path / "resource-manifest.json"
    context = ScenarioContext(
        ScenarioContextConfig(
            run_id=RunId("pre-process-interruption"),
            run_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            compose_project="gw-verify-pre-process",
            allocated_port=20_001,
            interrupt_reader=lambda: signal.SIGTERM,
        ),
        ledger=ResourceLedger(manifest_path),
    )

    async def exercise() -> None:
        with pytest.raises(VerificationInterruptedError):
            async with context.process(name="never-started", command=("/usr/bin/false",)):
                pytest.fail("interrupted process must not start")

    anyio.run(exercise)

    _assert_signal_receipt(manifest_path, expected_signum=signal.SIGTERM)
    raw = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    assert all(
        event.get("name") != "never-started"
        for event in cast("list[dict[str, object]]", raw["events"])
    )


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
        _assert_cleanup(evidence, container_name, expected_signum=signal.SIGTERM)
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
        _assert_cleanup(evidence, container_name, expected_signum=signal.SIGTERM)
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
        _assert_cleanup(evidence, container_name, expected_signum=signal.SIGTERM)
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
        _assert_cleanup(
            evidence,
            _media_gateway_container_name(manifest_path),
            expected_signum=signal.SIGINT,
        )
        _assert_gateway_lifecycle(manifest_path)


def test_foreground_pty_double_ctrl_c_during_cleanup_is_cooperative(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "foreground-pty-double-ctrl-c"
    returncode, manifest_path = _run_foreground_pty_double_ctrl_c(evidence)

    assert returncode == 130
    _assert_task10_cleanup(evidence, manifest_path)


def test_foreground_pty_fast_double_ctrl_c_during_media_cleanup_is_cooperative(
    tmp_path: Path,
) -> None:
    evidence = tmp_path / "foreground-pty-fast-double-ctrl-c-media"
    returncode, manifest_path = _run_foreground_pty_fast_double_ctrl_c(evidence)

    assert returncode == 130
    _assert_cleanup(
        evidence,
        _media_gateway_container_name(manifest_path),
        expected_signum=signal.SIGINT,
    )
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


def _run_foreground_pty_double_ctrl_c(evidence: Path) -> tuple[int, Path]:  # noqa: C901
    command = (
        str(CLI),
        "verify",
        "--scenario",
        "ingest",
        "--evidence",
        str(evidence),
    )
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execv(command[0], command)  # noqa: S606

    output = bytearray()
    deadline = time.monotonic() + 180
    first_signal_sent = False
    second_signal_sent = False
    status: int | None = None
    manifest_path = evidence / "resource-manifest.json"
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
            if not first_signal_sent and _task10_event_started(
                manifest_path, "task10-ingest-driver"
            ):
                _ = os.write(master_fd, b"\x03")
                first_signal_sent = True
            if (
                first_signal_sent
                and not second_signal_sent
                and _task10_event_started(manifest_path, "task10-fixtures-down")
            ):
                _ = os.write(master_fd, b"\x03")
                second_signal_sent = True
            waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                status = waited_status
                break
        assert first_signal_sent, "foreground PTY ingest did not start its driver"
        assert second_signal_sent, "foreground PTY ingest did not enter fixture cleanup"
        if status is None:
            _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status), manifest_path
    finally:
        if status is None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                _ = os.waitpid(pid, 0)
        with suppress(OSError):
            os.close(master_fd)
        _ = (evidence / "pty-output.bin").write_bytes(output)


def _run_foreground_pty_fast_double_ctrl_c(evidence: Path) -> tuple[int, Path]:  # noqa: C901
    command = (
        str(CLI),
        "verify",
        "--scenario",
        "media",
        "--evidence",
        str(evidence),
    )
    pid, master_fd = pty.fork()
    if pid == 0:
        os.execv(command[0], command)  # noqa: S606

    output = bytearray()
    deadline = time.monotonic() + 30
    first_signal_at: float | None = None
    second_signal_sent = False
    status: int | None = None
    manifest_path = evidence / "resource-manifest.json"
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
            now = time.monotonic()
            if first_signal_at is None and _gateway_started(manifest_path):
                _ = os.write(master_fd, b"\x03")
                first_signal_at = now
            if (
                first_signal_at is not None
                and not second_signal_sent
                and now - first_signal_at >= 0.30
            ):
                _ = os.write(master_fd, b"\x03")
                second_signal_sent = True
            waited_pid, waited_status = os.waitpid(pid, os.WNOHANG)
            if waited_pid == pid:
                status = waited_status
                break
        assert first_signal_at is not None, "foreground PTY media gateway did not start"
        assert second_signal_sent, "foreground PTY media verifier exited before repeated Ctrl-C"
        if status is None:
            _, status = os.waitpid(pid, 0)
        return os.waitstatus_to_exitcode(status), manifest_path
    finally:
        if status is None:
            with suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            with suppress(ChildProcessError):
                _ = os.waitpid(pid, 0)
        with suppress(OSError):
            os.close(master_fd)
        _ = (evidence / "pty-output-fast-double.bin").write_bytes(output)


def _task10_event_started(manifest_path: Path, name: str) -> bool:
    if not manifest_path.exists():
        return False
    try:
        raw = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False
    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        return False
    for raw_event in cast("list[object]", raw_events):
        if not isinstance(raw_event, dict):
            continue
        event = cast("dict[str, object]", raw_event)
        if event.get("name") == name and event.get("state") == "started":
            return True
    return False


def _assert_task10_cleanup(evidence: Path, manifest_path: Path) -> None:
    assert not (evidence / "runtime").exists()
    raw = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    _assert_signal_receipt(manifest_path, expected_signum=signal.SIGINT)
    raw_events = raw["events"]
    assert isinstance(raw_events, list)
    by_name: dict[str, list[str]] = {}
    for raw_event in cast("list[object]", raw_events):
        if not isinstance(raw_event, dict):
            continue
        event = cast("dict[str, object]", raw_event)
        name = event.get("name")
        if isinstance(name, str):
            by_name.setdefault(name, []).append(str(event.get("state")))
    for name in (
        "task10-ingest-driver",
        "task10-triton-down",
        "task10-fixtures-down",
        "task10-resource-inspect-label",
        "task10-resource-inspect-name",
    ):
        assert by_name.get(name) == ["registered", "started", "cleaned"]
    cleanup = cast(
        "dict[str, object]",
        json.loads((evidence / "task10-cleanup.json").read_text(encoding="utf-8")),
    )
    assert cleanup["triton_remove_exit_code"] == 0
    assert cleanup["compose_down_exit_code"] == 0
    assert cleanup["inspect_return_code"] == 0
    assert cleanup["remaining_owned_resources"] == ""


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


def test_gateway_cleanup_receipt_rejects_misleading_operations(tmp_path: Path) -> None:
    owner = "gw-verify-test-media"
    receipt_path = tmp_path / "task-9-gateway-interrupt-cleanup.json"
    valid = _valid_gateway_cleanup_receipt(owner)
    _ = receipt_path.write_text(json.dumps(valid), encoding="utf-8")

    stop_failure = copy.deepcopy(valid)
    stop_operation = _receipt_operation(stop_failure, 1)
    stop_operation["returncode"] = 1
    stop_operation["stderr"] = "synthetic injected stop failure"
    _ = receipt_path.write_text(json.dumps(stop_failure), encoding="utf-8")
    with pytest.raises(AssertionError, match="stop-failed"):
        _assert_gateway_cleanup_receipts(tmp_path, owner)

    remove_failure = copy.deepcopy(valid)
    remove_operation = _receipt_operation(remove_failure, 2)
    remove_operation["returncode"] = 1
    remove_operation["stderr"] = "synthetic injected remove failure"
    _ = receipt_path.write_text(json.dumps(remove_failure), encoding="utf-8")
    with pytest.raises(AssertionError, match="remove-force-failed"):
        _assert_gateway_cleanup_receipts(tmp_path, owner)

    foreign_owner = copy.deepcopy(valid)
    foreign_owner["container_name"] = "foreign-owner-media"
    _ = receipt_path.write_text(json.dumps(foreign_owner), encoding="utf-8")
    with pytest.raises(AssertionError, match="container-name-mismatch"):
        _assert_gateway_cleanup_receipts(tmp_path, owner)

    _ = receipt_path.write_text(json.dumps(valid), encoding="utf-8")
    _assert_gateway_cleanup_receipts(tmp_path, owner)


def test_gateway_cleanup_receipt_finishes_inflight_operation_after_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    owner = "gw-verify-slow-media"
    inspect_calls = 0

    @final
    class Completed:
        def __init__(self, returncode: int, *, stderr: bytes = b"", stdout: bytes = b"") -> None:
            self.returncode: int = returncode
            self.stderr: bytes = stderr
            self.stdout: bytes = stdout

    async def slow_run_process(command: tuple[str, ...], **kwargs: object) -> Completed:
        del kwargs
        nonlocal inspect_calls
        await anyio.sleep(0.8)
        if command[1:3] == ("container", "inspect"):
            inspect_calls += 1
            if inspect_calls == 1:
                return Completed(0, stdout=b'[{"Id":"owned"}]')
            return Completed(
                1,
                stderr=f"Error response from daemon: No such container: {owner}".encode(),
                stdout=b"[]",
            )
        return Completed(0)

    monkeypatch.setattr(anyio, "run_process", slow_run_process)
    context = ScenarioContext(
        ScenarioContextConfig(
            run_id=RunId("slow-cleanup-receipt"),
            run_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            compose_project="gw-verify-slow-cleanup",
            allocated_port=20_002,
        ),
        ledger=ResourceLedger(tmp_path / "resource-manifest.json"),
    )

    anyio.run(
        lambda: task09_stack.cleanup_media_gateway(
            context,
            owner,
            receipt_name="task-9-gateway-cleanup.json",
        )
    )

    receipt = cast(
        "dict[str, object]",
        json.loads((tmp_path / "task-9-gateway-cleanup.json").read_text()),
    )
    raw_operations = receipt["operations"]
    assert isinstance(raw_operations, list)
    operations = cast("list[dict[str, object]]", raw_operations)
    assert [operation["operation"] for operation in operations] == [
        "wait-for-create",
        "stop",
        "remove-force",
        "inspect",
    ]
    assert receipt["timed_out"] is False
    assert receipt["deadline_exceeded"] is True
    assert all(isinstance(operation["duration_seconds"], float) for operation in operations)


def _valid_gateway_cleanup_receipt(owner: str) -> dict[str, object]:
    no_such = f"Error response from daemon: No such container: {owner}"
    return {
        "container_name": owner,
        "operations": [
            {"operation": "wait-for-create", "timed_out": True},
            {
                "command": [
                    "/usr/bin/docker",
                    "container",
                    "stop",
                    "--timeout",
                    "1",
                    owner,
                ],
                "operation": "stop",
                "returncode": 1,
                "stderr": no_such,
                "stdout": "",
            },
            {
                "command": [
                    "/usr/bin/docker",
                    "container",
                    "remove",
                    "--force",
                    owner,
                ],
                "operation": "remove-force",
                "returncode": 0,
                "stderr": no_such,
                "stdout": "",
            },
            {
                "command": [
                    "/usr/bin/docker",
                    "container",
                    "inspect",
                    owner,
                ],
                "operation": "inspect",
                "returncode": 1,
                "stderr": no_such,
                "stdout": "[]",
            },
        ],
        "timed_out": False,
    }


def _receipt_operation(receipt: dict[str, object], index: int) -> dict[str, object]:
    raw_operations = receipt.get("operations")
    assert isinstance(raw_operations, list)
    raw_operation = cast("list[object]", raw_operations)[index]
    assert isinstance(raw_operation, dict)
    return cast("dict[str, object]", raw_operation)


def _assert_cleanup(evidence: Path, container_name: str, *, expected_signum: int) -> None:
    assert not (evidence / "runtime").exists()
    assert not _container_exists(container_name)
    _assert_gateway_cleanup_receipts(evidence, container_name)
    _assert_signal_receipt(evidence / "resource-manifest.json", expected_signum)


def _assert_gateway_cleanup_receipts(evidence: Path, expected_owner: str) -> None:
    receipt_paths = tuple(
        evidence / name
        for name in (
            "task-9-gateway-interrupt-cleanup.json",
            "task-9-gateway-cleanup.json",
        )
        if (evidence / name).exists()
    )
    assert receipt_paths, "gateway cleanup receipt is missing"
    for receipt_path in receipt_paths:
        errors = _gateway_cleanup_receipt_errors(receipt_path, expected_owner)
        assert not errors, f"{receipt_path.name}: {', '.join(errors)}"


def _gateway_cleanup_receipt_errors(receipt_path: Path, expected_owner: str) -> tuple[str, ...]:
    try:
        raw = cast("dict[str, object]", json.loads(receipt_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, TypeError):
        return ("invalid-json",)
    errors: list[str] = []
    if raw.get("container_name") != expected_owner:
        errors.append("container-name-mismatch")
    if raw.get("timed_out") is not False:
        errors.append("cleanup-timed-out")
    raw_operations = raw.get("operations")
    if not isinstance(raw_operations, list):
        return (*errors, "operations-not-list")
    operation_values = cast("list[object]", raw_operations)
    operations = [
        cast("dict[str, object]", operation)
        for operation in operation_values
        if isinstance(operation, dict)
    ]
    expected_names = ("wait-for-create", "stop", "remove-force", "inspect")
    names = tuple(operation.get("operation") for operation in operations)
    if names != expected_names or len(operations) != len(expected_names):
        errors.append("operation-order-mismatch")
        return tuple(errors)

    expected_commands = {
        "stop": [
            "/usr/bin/docker",
            "container",
            "stop",
            "--timeout",
            "1",
            expected_owner,
        ],
        "remove-force": [
            "/usr/bin/docker",
            "container",
            "remove",
            "--force",
            expected_owner,
        ],
        "inspect": ["/usr/bin/docker", "container", "inspect", expected_owner],
    }
    for operation in operations[1:]:
        operation_name = cast("str", operation["operation"])
        if operation.get("command") != expected_commands[operation_name]:
            errors.append(f"{operation_name}-command-owner-mismatch")

    if not isinstance(operations[0].get("timed_out"), bool):
        errors.append("wait-for-create-timeout-missing")
    errors.extend(_stop_operation_errors(operations[1], expected_owner))
    errors.extend(_remove_operation_errors(operations[2], expected_owner))
    errors.extend(_inspect_operation_errors(operations[3], expected_owner))
    return tuple(errors)


def _stop_operation_errors(operation: dict[str, object], expected_owner: str) -> list[str]:
    no_such = f"Error response from daemon: No such container: {expected_owner}"
    return_code = operation.get("returncode")
    if return_code == 0:
        return [] if operation.get("stderr") == "" else ["stop-stderr-on-success"]
    if return_code == 1 and operation.get("stdout") == "" and operation.get("stderr") == no_such:
        return []
    return ["stop-failed"]


def _remove_operation_errors(operation: dict[str, object], expected_owner: str) -> list[str]:
    no_such = f"Error response from daemon: No such container: {expected_owner}"
    if operation.get("returncode") != 0:
        return ["remove-force-failed"]
    if operation.get("stderr") not in ("", no_such):
        return ["remove-force-stderr-unexpected"]
    if operation.get("stdout") not in ("", expected_owner):
        return ["remove-force-owner-output-mismatch"]
    return []


def _inspect_operation_errors(operation: dict[str, object], expected_owner: str) -> list[str]:
    no_such = f"Error response from daemon: No such container: {expected_owner}"
    if (
        operation.get("returncode") == 1
        and operation.get("stdout") == "[]"
        and operation.get("stderr") == no_such
    ):
        return []
    return ["inspect-did-not-prove-absence"]


def _assert_signal_receipt(manifest_path: Path, expected_signum: int) -> None:
    raw = cast("dict[str, object]", json.loads(manifest_path.read_text(encoding="utf-8")))
    raw_events = raw.get("events")
    assert isinstance(raw_events, list)
    signal_events: list[dict[str, object]] = []
    for raw_event in cast("list[object]", raw_events):
        if not isinstance(raw_event, dict):
            continue
        event = cast("dict[str, object]", raw_event)
        if (
            event.get("kind") == "signal"
            and event.get("name") == "verification"
            and event.get("state") == "received"
        ):
            signal_events.append(event)
    assert len(signal_events) == 1
    assert signal_events[0].get("detail") == f"signum={expected_signum}"


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


def _gateway_cleanup_container_name(receipt_path: Path) -> str:
    raw = cast("dict[str, object]", json.loads(receipt_path.read_text(encoding="utf-8")))
    container_name = raw.get("container_name")
    assert isinstance(container_name, str)
    return container_name


def _container_exists(name: str) -> bool:
    inspected = subprocess.run(  # noqa: S603
        ("/usr/bin/docker", "container", "inspect", name),
        check=False,
        capture_output=True,
    )
    return inspected.returncode == 0
