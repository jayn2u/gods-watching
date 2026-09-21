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
        "GW_MEDIA_CONTROL_PASSWORD=control-password-material",
        "GW_MEDIA_READER_PASSWORD=reader-password-material",
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


@pytest.mark.parametrize(
    "override",
    ["GW_PUBLIC_PORT=not-a-port", "GW_PUBLIC_PORT=70000", "GW_PUBLIC_PORT=8443"],
)
def test_doctor_rejects_unusable_published_port(fake_runtime: Path, override: str) -> None:
    # Given: an environment file whose published port cannot be bound as configured
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(
        _VALID_ENVIRONMENT + _LINE_BREAK.join((override, "")), encoding="utf-8"
    )
    _ = environment_path.chmod(0o600)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: readiness fails before Compose attempts the bind
    assert result.exit_code == 1
    assert _BOOLEAN_REPORT.validate_json(result.stdout)["credentials"] is False


@pytest.mark.parametrize(
    "override",
    [
        "GW_API_PORT=8080",
        "GW_TRITON_GRPC_PORT=18000",
        "GW_MEDIA_RTSP_PORT=19997",
        "GW_POSTGRES_PORT=not-a-port",
        "GW_MEDIA_WHEP_PORT=70000",
    ],
)
def test_doctor_rejects_colliding_host_network_ports(fake_runtime: Path, override: str) -> None:
    # Given: every service binds the host namespace directly, so two services that
    # name the same port cannot both start
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(
        _VALID_ENVIRONMENT + _LINE_BREAK.join((override, "")), encoding="utf-8"
    )
    _ = environment_path.chmod(0o600)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: the collision is reported before Compose attempts the bind
    assert result.exit_code == 1
    assert _BOOLEAN_REPORT.validate_json(result.stdout)["credentials"] is False


def test_doctor_accepts_relocated_host_network_ports(fake_runtime: Path) -> None:
    # Given: an operator who moves the internal ports off values the host already uses
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(
        _VALID_ENVIRONMENT
        + _LINE_BREAK.join(("GW_API_PORT=28000", "GW_MEDIA_RTSP_PORT=28554", "")),
        encoding="utf-8",
    )
    _ = environment_path.chmod(0o600)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: the distinct overrides are accepted
    assert result.exit_code == 0
    assert _BOOLEAN_REPORT.validate_json(result.stdout)["credentials"] is True


def test_doctor_rejects_an_origin_that_names_another_port(fake_runtime: Path) -> None:
    # Given: a moved published port that the browser-facing origin still ignores
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(
        _VALID_ENVIRONMENT
        + _LINE_BREAK.join(("GW_PUBLIC_PORT=9080", "GW_PUBLIC_ORIGIN=http://localhost:8080", "")),
        encoding="utf-8",
    )
    _ = environment_path.chmod(0o600)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: the mismatch is reported before it becomes a same-origin login failure
    assert result.exit_code == 1
    assert _BOOLEAN_REPORT.validate_json(result.stdout)["credentials"] is False


def test_doctor_accepts_a_custom_published_port(fake_runtime: Path) -> None:
    # Given: an environment file that moves the console off the default port
    environment_path = fake_runtime.parent / ".env"
    _ = environment_path.write_text(
        _VALID_ENVIRONMENT
        + _LINE_BREAK.join(("GW_PUBLIC_PORT=9080", "GW_PUBLIC_ORIGIN=http://localhost:9080", "")),
        encoding="utf-8",
    )
    _ = environment_path.chmod(0o600)

    # When: the operator runs the preflight doctor
    result = _RUNNER.invoke(app, ["doctor"])

    # Then: the override is accepted
    assert result.exit_code == 0
    assert _BOOLEAN_REPORT.validate_json(result.stdout)["credentials"] is True


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
    assert len(values["GW_MEDIA_CONTROL_PASSWORD"]) >= 20
    assert len(values["GW_MEDIA_READER_PASSWORD"]) >= 20
    assert len(values["GW_OPERATOR_PASSWORD"]) >= 20
    assert len(values["GW_POSTGRES_PASSWORD"]) >= 20
    generated_secrets = {
        values["GW_MEDIA_CONTROL_PASSWORD"],
        values["GW_MEDIA_READER_PASSWORD"],
        values["GW_OPERATOR_PASSWORD"],
        values["GW_POSTGRES_PASSWORD"],
    }
    assert len(generated_secrets) == 4
    assert all("replace-with" not in value for value in values.values())


def test_compose_does_not_ship_media_gateway_credentials() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    compose = (repository_root / "compose.yaml").read_text(encoding="utf-8")
    media_config = (repository_root / "deploy/mediamtx.compose.yml").read_text(encoding="utf-8")

    assert "gw-control-internal-2026" not in compose
    assert "gw-reader-internal-2026" not in compose
    assert "authInternalUsers:" not in media_config
    assert "MTX_AUTHINTERNALUSERS_0_PASS" in compose
    assert "MTX_AUTHINTERNALUSERS_1_PASS" in compose


def test_prepare_preserves_existing_credentials(fake_runtime: Path) -> None:
    # Given: an operator-managed credential file already exists
    environment_path = fake_runtime.parent / ".env"
    existing = _VALID_ENVIRONMENT
    _ = environment_path.write_text(existing, encoding="utf-8")
    _ = environment_path.chmod(0o600)

    # When: preparation runs again
    result = _RUNNER.invoke(app, ["prepare"])

    # Then: the existing credential material is not replaced
    assert result.exit_code == 0
    assert environment_path.read_text(encoding="utf-8") == existing


def test_prepare_adds_only_missing_media_credentials(fake_runtime: Path) -> None:
    environment_path = fake_runtime.parent / ".env"
    existing = _LINE_BREAK.join(
        (
            "GW_CAMERA_CIPHER_KEY=MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
            "GW_OPERATOR_PASSWORD=operator-owned-secret",
            "GW_POSTGRES_PASSWORD=database-password-material",
            "",
        )
    )
    _ = environment_path.write_text(existing, encoding="utf-8")
    _ = environment_path.chmod(0o600)

    result = _RUNNER.invoke(app, ["prepare"])

    assert result.exit_code == 0
    contents = environment_path.read_text(encoding="utf-8")
    assert contents.startswith(existing)
    values = dict(line.split("=", maxsplit=1) for line in contents.splitlines())
    existing_values = dict(line.split("=", maxsplit=1) for line in existing.splitlines())
    assert values["GW_OPERATOR_PASSWORD"] == existing_values["GW_OPERATOR_PASSWORD"]
    assert len(values["GW_MEDIA_CONTROL_PASSWORD"]) >= 20
    assert len(values["GW_MEDIA_READER_PASSWORD"]) >= 20
