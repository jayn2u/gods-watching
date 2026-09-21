from pathlib import Path

import pytest
from pydantic import TypeAdapter
from typer.testing import CliRunner

from gods_watching.cli import app

_RUNNER = CliRunner()
_BOOLEAN_REPORT = TypeAdapter(dict[str, bool])
_LINE_BREAK = chr(10)
_VALID_ENVIRONMENT = _LINE_BREAK.join(
    (
        "GW_CAMERA_CIPHER_KEY=MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
        "GW_OPERATOR_PASSWORD=correct-horse-battery-staple",
        "GW_POSTGRES_PASSWORD=database-password-material",
        "",
    )
)


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log_path = tmp_path / "commands.log"
    docker = tmp_path / "docker"
    docker_lines = (
        "#!/bin/sh",
        'printf "docker %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"',
        'exit "${GW_TEST_DOCKER_EXIT:-0}"',
    )
    _ = docker.write_text(
        "\n".join(docker_lines) + "\n",
        encoding="utf-8",
    )
    _ = docker.chmod(0o700)
    nvidia_smi = tmp_path / "nvidia-smi"
    _ = nvidia_smi.write_text(
        '#!/bin/sh\nprintf "nvidia-smi %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"\nexit 0\n',
        encoding="utf-8",
    )
    _ = nvidia_smi.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GW_TEST_COMMAND_LOG", str(log_path))
    monkeypatch.setattr("gods_watching.lifecycle._ENV_PATH", tmp_path / ".env")
    return log_path


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("prepare", ("docker compose config --quiet", "docker compose build")),
        ("up", ("docker compose up -d",)),
        ("down", ("docker compose down",)),
        ("status", ("docker compose ps",)),
    ],
)
def test_lifecycle_command_dispatches_compose(
    fake_runtime: Path,
    command: str,
    expected: tuple[str, ...],
) -> None:
    # Given: Docker and Compose are available through a deterministic command boundary
    # When: the operator invokes one lifecycle command
    result = _RUNNER.invoke(app, [command])

    # Then: the CLI succeeds and dispatches only the expected Compose operation
    assert result.exit_code == 0
    assert fake_runtime.read_text(encoding="utf-8").splitlines() == list(expected)


def test_doctor_reports_every_required_runtime_capability(fake_runtime: Path) -> None:
    # Given: Docker, Compose, NVIDIA tooling, and the Compose file all validate
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(_VALID_ENVIRONMENT, encoding="utf-8")
    _ = environment_path.chmod(0o600)
    # When: the operator runs doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: readiness is machine-readable and every required capability passed
    assert result.exit_code == 0
    assert _BOOLEAN_REPORT.validate_json(result.stdout) == {
        "compose": True,
        "configuration": True,
        "credentials": True,
        "docker": True,
        "gpu": True,
        "ready": True,
    }
    assert fake_runtime.read_text(encoding="utf-8").splitlines() == [
        "docker version",
        "docker compose version",
        "nvidia-smi --query-gpu=name --format=csv,noheader",
        "docker compose config --quiet",
    ]


def test_doctor_rejects_permissive_credential_file(fake_runtime: Path) -> None:
    # Given: credential material stored in a group-readable environment file
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(_VALID_ENVIRONMENT, encoding="utf-8")
    _ = environment_path.chmod(0o640)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: readiness fails without rendering any secret values
    assert result.exit_code == 1
    report = _BOOLEAN_REPORT.validate_json(result.stdout)
    assert report["credentials"] is False
    assert "correct-horse-battery-staple" not in result.stdout


def test_lifecycle_command_preserves_compose_failure_code(
    fake_runtime: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: Docker reports a concrete Compose failure
    monkeypatch.setenv("GW_TEST_DOCKER_EXIT", "7")

    # When: the operator starts the product
    result = _RUNNER.invoke(app, ["up"])

    # Then: the same failure code reaches the caller
    assert result.exit_code == 7
    assert fake_runtime.read_text(encoding="utf-8").splitlines() == ["docker compose up -d"]


def test_prepare_creates_private_first_run_credentials(fake_runtime: Path) -> None:
    # Given: a new installation without an environment credential file
    environment_path = fake_runtime.parent / ".env"

    # When: the operator prepares the product
    result = _RUNNER.invoke(app, ["prepare"])

    # Then: unique secrets are stored privately before Compose validates the stack
    assert result.exit_code == 0
    assert environment_path.stat().st_mode & 0o777 == 0o600
    values = dict(
        line.split("=", maxsplit=1)
        for line in environment_path.read_text(encoding="utf-8").splitlines()
    )
    assert len(values["GW_CAMERA_CIPHER_KEY"]) == 44
    assert len(values["GW_OPERATOR_PASSWORD"]) >= 20
    assert len(values["GW_POSTGRES_PASSWORD"]) >= 20
    assert values["GW_OPERATOR_PASSWORD"] != values["GW_POSTGRES_PASSWORD"]
    assert all("replace-with" not in value for value in values.values())


def test_prepare_preserves_existing_credentials(fake_runtime: Path) -> None:
    # Given: an operator-managed credential file already exists
    environment_path = fake_runtime.parent / ".env"
    existing = "GW_OPERATOR_PASSWORD=operator-owned-secret\n"
    _ = environment_path.write_text(existing, encoding="utf-8")
    _ = environment_path.chmod(0o600)

    # When: preparation runs again
    result = _RUNNER.invoke(app, ["prepare"])

    # Then: the existing credential material is not replaced
    assert result.exit_code == 0
    assert environment_path.read_text(encoding="utf-8") == existing
