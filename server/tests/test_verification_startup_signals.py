import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
CLI = REPOSITORY_ROOT / "gods-watching"


def test_cli_preserves_sigterm_received_during_module_import(tmp_path: Path) -> None:
    hook_root = tmp_path / "import-hook"
    hook_root.mkdir()
    _ = (hook_root / "sitecustomize.py").write_text(
        "\n".join(  # noqa: FLY002
            (
                "import builtins",
                "import os",
                "import signal",
                "_real_import = builtins.__import__",
                "def _interrupting_import(name, globals=None, locals=None, fromlist=(), level=0):",
                "    if name == 'typer':",
                "        builtins.__import__ = _real_import",
                "        os.kill(os.getpid(), signal.SIGTERM)",
                "    return _real_import(name, globals, locals, fromlist, level)",
                "builtins.__import__ = _interrupting_import",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    evidence = tmp_path / "import-race"
    completed = _run_cli(
        (
            "verify",
            "--scenario",
            "harness-self-check",
            "--evidence",
            str(evidence),
        ),
        extra_environment={"PYTHONPATH": _pythonpath(hook_root)},
    )

    assert completed.returncode == 143
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert not evidence.exists()


def test_launcher_preserves_sigterm_received_during_uv_resolution(tmp_path: Path) -> None:
    uv_root = tmp_path / "uv-bin"
    uv_root.mkdir()
    uv = uv_root / "uv"
    _ = uv.write_text(
        "\n".join(  # noqa: FLY002
            (
                "#!/bin/sh",
                'kill -TERM "$PPID"',
                "sleep 0.25",
                'printf "%s\\n" "$GW_TEST_PYTHON"',
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _ = uv.chmod(0o700)
    evidence = tmp_path / "launcher-race"
    completed = _run_cli(
        (
            "verify",
            "--scenario",
            "harness-self-check",
            "--evidence",
            str(evidence),
        ),
        extra_environment={
            "PATH": f"{uv_root}{os.pathsep}{os.environ['PATH']}",
            "GW_TEST_PYTHON": sys.executable,
        },
    )

    assert completed.returncode == 143
    assert completed.stdout == ""
    assert completed.stderr == ""
    assert not evidence.exists()


def test_launcher_forwards_sigterm_to_child_after_uv_resolution(tmp_path: Path) -> None:
    uv_root = tmp_path / "uv-bin"
    uv_root.mkdir()
    fake_python = tmp_path / "fake-python"
    started = tmp_path / "child-started"
    terminated = tmp_path / "child-terminated"
    _ = fake_python.write_text(
        "\n".join(
            (
                "#!/bin/sh",
                f": > {started}",
                f"trap ': > {terminated}; exit 143' TERM INT",
                "while :; do sleep 1; done",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    _ = fake_python.chmod(0o700)
    uv = uv_root / "uv"
    _ = uv.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$GW_TEST_PYTHON\"\n",
        encoding="utf-8",
    )
    _ = uv.chmod(0o700)

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{uv_root}{os.pathsep}{environment['PATH']}",
            "GW_TEST_PYTHON": str(fake_python),
        }
    )
    process = subprocess.Popen(  # noqa: S603
        (
            str(CLI),
            "verify",
            "--scenario",
            "harness-self-check",
            "--evidence",
            str(tmp_path / "launcher-forward"),
        ),
        cwd=REPOSITORY_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert started.exists(), "launcher child did not start"
        os.kill(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 2
        while not terminated.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert terminated.exists(), "launcher did not forward SIGTERM to its child"
        _ = process.communicate(timeout=5)
        assert process.returncode == 143
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            _ = process.communicate(timeout=5)


def _run_cli(
    arguments: tuple[str, ...], *, extra_environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(extra_environment)
    return subprocess.run(  # noqa: S603
        (str(CLI), *arguments),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )


def _pythonpath(hook_root: Path) -> str:
    source_root = REPOSITORY_ROOT / "server" / "src"
    return os.pathsep.join((str(hook_root), str(source_root)))
