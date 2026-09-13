import pytest
from typer.testing import CliRunner

from gods_watching.cli import app

_RUNNER = CliRunner()


def test_worker_help_describes_the_pipeline_worker() -> None:
    # Given / When
    result = _RUNNER.invoke(app, ["worker", "--help"])

    # Then
    assert result.exit_code == 0
    assert "pipeline worker" in result.stdout.lower()


def test_worker_rejects_missing_configuration_before_starting_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no worker environment
    for name in (
        "GW_DATABASE_URL",
        "GW_TRITON_GRPC_URL",
        "GW_CROPS_ROOT",
        "GW_CAMERA_CIPHER_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    # When
    result = _RUNNER.invoke(app, ["worker"])

    # Then: startup stops with a configuration error naming the missing variables
    assert result.exit_code == 2
    assert "worker configuration" in result.output
    assert "GW_DATABASE_URL" in result.output
    assert "Traceback" not in result.output
