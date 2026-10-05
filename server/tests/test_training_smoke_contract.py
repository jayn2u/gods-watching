from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from qa.training import run_smoke
from qa.training.run_smoke import SmokeConfigurationError, validate_interrupt_checkpoint_evidence

if TYPE_CHECKING:
    from pathlib import Path


def test_restart_requires_a_completed_checkpoint_and_active_later_epoch() -> None:
    valid = {
        "kind": "interrupt-me",
        "job_id": "job-123",
        "phase": "training",
        "current_epoch": 2,
        "checkpoint_epoch": 1,
        "total_epochs": 3,
    }
    assert validate_interrupt_checkpoint_evidence(valid) == "job-123"

    not_checkpointed = {**valid, "current_epoch": 1, "checkpoint_epoch": 0}
    completed_run = {**valid, "current_epoch": 3}
    for unsafe in (not_checkpointed, completed_run):
        with pytest.raises(SmokeConfigurationError, match="complete checkpoint"):
            _ = validate_interrupt_checkpoint_evidence(unsafe)


@pytest.mark.parametrize(("phase", "job_id"), [("shortage", None), ("coexist", "job-123")])
def test_special_runtime_phases_are_exposed_by_the_smoke_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    job_id: str | None,
) -> None:
    fake_playwright = tmp_path / "playwright"
    fake_playwright.touch()
    monkeypatch.setattr(run_smoke, "PLAYWRIGHT", fake_playwright)
    monkeypatch.setenv("GW_E2E_OPERATOR_USERNAME", "qa-operator")
    monkeypatch.setenv("GW_E2E_OPERATOR_PASSWORD", "qa-password")
    monkeypatch.setenv("GW_TRAINING_SMOKE_JOB_ID", "stale-job")
    argv = [
        "run_smoke.py",
        "--phase",
        phase,
        "--base-url",
        "http://127.0.0.1:28080",
        "--evidence-dir",
        str(tmp_path / "evidence"),
    ]
    if job_id is not None:
        argv.extend(("--job-id", job_id))
    monkeypatch.setattr(sys, "argv", argv)
    captured: dict[str, object] = {}

    def record_browser_call(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", record_browser_call)

    exit_code = run_smoke.main()

    assert exit_code == 0
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert environment["GW_TRAINING_SMOKE_PHASE"] == phase
    if job_id is None:
        assert "GW_TRAINING_SMOKE_JOB_ID" not in environment
    else:
        assert environment["GW_TRAINING_SMOKE_JOB_ID"] == job_id


def test_coexist_cli_phase_requires_a_job_id(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["run_smoke.py", "--phase", "coexist"])

    exit_code = run_smoke.main()

    assert exit_code == 2
    assert "coexist phase requires a training job ID via --job-id" in capsys.readouterr().err
