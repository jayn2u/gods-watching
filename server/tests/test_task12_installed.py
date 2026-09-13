"""Installed Task 12 command and scenario registration contracts."""

import os
import subprocess
from pathlib import Path

from gods_watching.verification import ImplementedScenario, ScenarioName, build_registry

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_installed_credentials_command_reaches_verified_module(tmp_path: Path) -> None:
    # Given: a safe password file and no database configuration
    password_file = tmp_path / "operator.secret"
    _ = password_file.write_text("a-valid-placeholder-password\n", encoding="utf-8")
    _ = password_file.chmod(0o600)
    environment = os.environ.copy()
    _ = environment.pop("GW_DATABASE_URL", None)

    # When: the installed launcher dispatches credentials set
    completed = subprocess.run(  # noqa: S603
        (
            str(REPOSITORY_ROOT / "gods-watching"),
            "credentials",
            "set",
            "--password-file",
            str(password_file),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then: the registered module reaches its configuration boundary
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.strip() == "database configuration is unavailable"


def test_task12_scenarios_are_implemented_and_require_real_runtime() -> None:
    # Given: the installed scenario registry
    registry = build_registry()

    # When: Task 12 names are resolved
    happy = registry.get(ScenarioName("camera-auth"))
    denied = registry.get(ScenarioName("camera-auth-denied"))

    # Then: both are executable real-runtime scenarios
    assert isinstance(happy, ImplementedScenario)
    assert isinstance(denied, ImplementedScenario)
    assert {"docker", "ffmpeg", "ffprobe", "node", "uv"} <= set(happy.required_commands)
    assert {"docker", "ffmpeg", "ffprobe", "node", "uv"} <= set(denied.required_commands)
    assert happy.timeout_seconds >= 120.0
    assert denied.timeout_seconds >= 120.0
