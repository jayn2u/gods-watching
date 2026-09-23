from pathlib import Path

import pytest
from typer.testing import CliRunner

from gods_watching import lifecycle
from gods_watching.cli import app


def test_prepare_lifecycle_runs_model_preparation_after_the_compose_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "commands.log"
    docker = tmp_path / "docker"
    _ = docker.write_text(
        '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"\n',
        encoding="utf-8",
    )
    _ = docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GW_TEST_COMMAND_LOG", str(log_path))
    monkeypatch.setattr(lifecycle, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(lifecycle, "_prepare_model_asset_directory", lambda: None)
    monkeypatch.setattr(lifecycle, "_invalidate_model_markers", lambda: None)
    monkeypatch.setattr(lifecycle, "_reuse_existing_yolo_asset", lambda: None)
    prepared: list[str] = []
    monkeypatch.setattr(lifecycle, "_prepare_model_assets", lambda: prepared.append("prepared"))

    result = CliRunner().invoke(app, ["prepare"])

    assert result.exit_code == 0
    assert prepared == ["prepared"]
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "docker compose config --quiet",
        "docker compose build",
    ]


def test_marker_invalidation_failure_prevents_the_compose_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_path = tmp_path / "commands.log"
    docker = tmp_path / "docker"
    _ = docker.write_text(
        '#!/bin/sh\nprintf "docker %s\\n" "$*" >> "$GW_TEST_COMMAND_LOG"\n',
        encoding="utf-8",
    )
    _ = docker.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("GW_TEST_COMMAND_LOG", str(log_path))
    monkeypatch.setattr(lifecycle, "_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(lifecycle, "_prepare_model_asset_directory", lambda: None)
    monkeypatch.setattr(lifecycle, "_reuse_existing_yolo_asset", lambda: None)

    def fail_invalidation() -> None:
        raise lifecycle.LifecycleCommandError(
            command=("model marker invalidation",), exit_code=2
        )

    monkeypatch.setattr(lifecycle, "_invalidate_model_markers", fail_invalidation)

    result = CliRunner().invoke(app, ["prepare"])

    assert result.exit_code == 2
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "docker compose config --quiet",
    ]
