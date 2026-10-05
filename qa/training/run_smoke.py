"""Drive the real authenticated browser training workflow against an isolated stack."""

# This QA executable is invoked by its file path, not imported as a package.
# ruff: noqa: INP001, TRY003, EM101, EM102, TRY301, TRY300

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = REPOSITORY_ROOT / "web"
PLAYWRIGHT = WEB_ROOT / "node_modules" / ".bin" / "playwright"
_PHASES = (
    "validate",
    "cancel",
    "interrupt",
    "resume",
    "finish",
    "candidate",
    "camera",
    "shortage",
    "coexist",
)
_PRIVATE_FILE_MODE = 0o600
_SmokePhase = Literal[
    "validate",
    "cancel",
    "interrupt",
    "resume",
    "finish",
    "candidate",
    "camera",
    "shortage",
    "coexist",
]
_ALL_WORKFLOW: tuple[_SmokePhase, ...] = (
    "validate",
    "cancel",
    "interrupt",
    "resume",
    "finish",
    "candidate",
)


@dataclass(frozen=True, slots=True)
class _SmokeArguments:
    phase: _SmokePhase | Literal["all"]
    base_url: str | None
    operator_env_file: Path | None
    compose_env_file: Path | None
    compose_project: str | None
    compose_file: Path
    job_id: str | None
    evidence_dir: Path | None


@dataclass(slots=True)
class _ParsedSmokeArguments(argparse.Namespace):
    phase: str | None = None
    base_url: str | None = None
    operator_env_file: Path | None = None
    compose_env_file: Path | None = None
    compose_project: str | None = None
    compose_file: Path = REPOSITORY_ROOT / "compose.yaml"
    job_id: str | None = None
    evidence_dir: Path | None = None


class SmokeConfigurationError(RuntimeError):
    """A real browser proof is missing one of its explicit isolated-stack inputs."""


@dataclass(frozen=True, slots=True)
class _BrowserContext:
    evidence_root: Path
    base_url: str
    username: str
    password: str
    environment: Mapping[str, str]


def _read_env_file(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        mode = path.stat().st_mode & 0o777
        if mode != _PRIVATE_FILE_MODE:
            raise SmokeConfigurationError("operator env file must have mode 0600")
        lines = path.read_text(encoding="utf-8").splitlines()
    except SmokeConfigurationError:
        raise
    except OSError as error:
        raise SmokeConfigurationError("operator environment file is unreadable") from error
    values: dict[str, str] = {}
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _configuration(args: _SmokeArguments) -> tuple[str, str, str, dict[str, str]]:
    file_values = _read_env_file(args.operator_env_file)
    username = (
        os.environ.get("GW_E2E_OPERATOR_USERNAME")
        or os.environ.get("GW_OPERATOR_USERNAME")
        or file_values.get("GW_OPERATOR_USERNAME")
        or "admin"
    )
    password = (
        os.environ.get("GW_E2E_OPERATOR_PASSWORD")
        or os.environ.get("GW_OPERATOR_PASSWORD")
        or file_values.get("GW_OPERATOR_PASSWORD")
    )
    base_url = (
        args.base_url
        or os.environ.get("GW_BASE_URL")
        or os.environ.get("GW_PUBLIC_ORIGIN")
        or file_values.get("GW_PUBLIC_ORIGIN")
    )
    if not password:
        raise SmokeConfigurationError(
            "set GW_E2E_OPERATOR_PASSWORD or provide a protected operator env file"
        )
    if not base_url:
        raise SmokeConfigurationError("set GW_BASE_URL or provide an explicit --base-url")
    parsed = urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeConfigurationError(
            "the browser base URL must be a credential-free HTTP(S) origin"
        )
    if not PLAYWRIGHT.is_file():
        raise SmokeConfigurationError("install the locked web dependencies before browser QA")
    child_environment = os.environ.copy()
    child_environment["GW_BASE_URL"] = base_url
    child_environment["GW_E2E_OPERATOR_USERNAME"] = username
    child_environment["GW_E2E_OPERATOR_PASSWORD"] = password
    return base_url, username, password, child_environment


def _safe_environment_file(path: Path | None) -> Path:
    if path is None:
        raise SmokeConfigurationError(
            "the isolated Compose env file is required for restart phases"
        )
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as error:
        raise SmokeConfigurationError("the isolated Compose env file is unreadable") from error
    if mode != _PRIVATE_FILE_MODE:
        raise SmokeConfigurationError("the isolated Compose env file must have mode 0600")
    return path.resolve(strict=True)


def _run_browser_phase(
    phase: str,
    *,
    job_id: str | None,
    context: _BrowserContext,
) -> dict[str, object]:
    phase_state = context.evidence_root / f"{phase}-state.json"
    environment = dict(context.environment)
    environment.update(
        {
            "GW_BASE_URL": context.base_url,
            "GW_E2E_EVIDENCE_ROOT": str(context.evidence_root),
            "GW_E2E_OPERATOR_USERNAME": context.username,
            "GW_E2E_OPERATOR_PASSWORD": context.password,
            "GW_TRAINING_SMOKE_PHASE": phase,
            "GW_TRAINING_SMOKE_STATE_FILE": str(phase_state),
        }
    )
    _ = environment.pop("GW_TRAINING_SMOKE_JOB_ID", None)
    if job_id is not None:
        environment["GW_TRAINING_SMOKE_JOB_ID"] = job_id
    command = (
        str(PLAYWRIGHT),
        "test",
        "e2e/training-runtime.spec.ts",
        "--project=chromium",
        "--workers=1",
    )
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=WEB_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = _redact(completed.stdout + completed.stderr, context.password)
    if output:
        _ = sys.stdout.write(output)
    if completed.returncode != 0:
        raise SmokeConfigurationError(
            f"browser phase {phase} failed with exit {completed.returncode}"
        )
    record: dict[str, object] = {"phase": phase, "exit_code": completed.returncode}
    if phase_state.is_file():
        value = cast("object", json.loads(phase_state.read_text(encoding="utf-8")))
        if isinstance(value, dict):
            record.update(cast("dict[str, object]", value))
    return record


def _restart_supervisor(
    args: _SmokeArguments,
    env_file: Path,
    password: str,
) -> dict[str, object]:
    project = args.compose_project
    if project is None or not project.startswith("gw-training-"):
        raise SmokeConfigurationError(
            "an explicit isolated Compose project name beginning with gw-training- is required"
        )
    compose_file = args.compose_file.resolve(strict=True)
    compose_prefix = (
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "--project-name",
        project,
        "--file",
        str(compose_file),
    )
    listed = subprocess.run(  # noqa: S603
        (*compose_prefix, "ps", "--quiet", "training"),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        raise SmokeConfigurationError("cannot inspect the isolated training supervisor")
    container_ids = [item for item in listed.stdout.splitlines() if item.strip()]
    if len(container_ids) != 1:
        raise SmokeConfigurationError(
            "isolated Compose project must have one running training service"
        )
    container_id = container_ids[0]

    def started_at() -> str:
        inspected = subprocess.run(  # noqa: S603
            ("docker", "inspect", "--format", "{{.State.StartedAt}}", container_id),
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if inspected.returncode != 0:
            raise SmokeConfigurationError("cannot inspect the isolated training supervisor state")
        return inspected.stdout.strip()

    before_restart = started_at()
    command = (
        *compose_prefix,
        "restart",
        "training",
    )
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    output = _redact(completed.stdout + completed.stderr, password)
    if output:
        _ = sys.stdout.write(output)
    if completed.returncode != 0:
        raise SmokeConfigurationError(
            f"isolated training supervisor restart failed with exit {completed.returncode}"
        )
    after_restart = started_at()
    if before_restart == after_restart:
        raise SmokeConfigurationError("training supervisor restart was not observed")
    return {
        "kind": "supervisor-restarted",
        "container_id": container_id,
        "started_at_before": before_restart,
        "started_at_after": after_restart,
    }


def _redact(output: str, password: str) -> str:
    return output.replace(password, "<redacted>")


def _recorded_job_id(
    reports: Sequence[dict[str, object]],
    *,
    kind: str,
    detail: str,
) -> str:
    record = next((item for item in reversed(reports) if item.get("kind") == kind), None)
    job_id = record.get("job_id") if record is not None else None
    if not isinstance(job_id, str):
        raise SmokeConfigurationError(detail)
    return job_id


def validate_interrupt_checkpoint_evidence(record: Mapping[str, object]) -> str:
    """Require a durable completed epoch while later training work is still active."""
    job_id = record.get("job_id")
    current_epoch = record.get("current_epoch")
    checkpoint_epoch = record.get("checkpoint_epoch")
    total_epochs = record.get("total_epochs")
    valid_counts = all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (current_epoch, checkpoint_epoch, total_epochs)
    )
    if (
        record.get("kind") != "interrupt-me"
        or record.get("phase") != "training"
        or not isinstance(job_id, str)
        or not job_id
        or not valid_counts
        or not isinstance(current_epoch, int)
        or not isinstance(checkpoint_epoch, int)
        or not isinstance(total_epochs, int)
        or checkpoint_epoch < 1
        or current_epoch < checkpoint_epoch
        or current_epoch >= total_epochs
    ):
        raise SmokeConfigurationError(
            "interrupt phase lacks a complete checkpoint with active training remaining"
        )
    return job_id


def _arguments() -> _SmokeArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--phase", choices=(*_PHASES, "all"), required=True)
    _ = parser.add_argument("--base-url")
    _ = parser.add_argument("--operator-env-file", type=Path)
    _ = parser.add_argument("--compose-env-file", type=Path)
    _ = parser.add_argument("--compose-project")
    _ = parser.add_argument("--compose-file", type=Path, default=REPOSITORY_ROOT / "compose.yaml")
    _ = parser.add_argument("--job-id")
    _ = parser.add_argument("--evidence-dir", type=Path)
    parsed = _ParsedSmokeArguments()
    _ = parser.parse_args(namespace=parsed)
    raw_phase = parsed.phase
    if raw_phase == "all":
        phase: _SmokePhase | Literal["all"] = "all"
    elif raw_phase in _PHASES:
        phase = raw_phase
    else:
        raise SmokeConfigurationError("select one of the supported training smoke phases")
    if phase == "coexist" and not parsed.job_id:
        raise SmokeConfigurationError("coexist phase requires a training job ID via --job-id")
    return _SmokeArguments(
        phase=phase,
        base_url=parsed.base_url,
        operator_env_file=parsed.operator_env_file,
        compose_env_file=parsed.compose_env_file,
        compose_project=parsed.compose_project,
        compose_file=parsed.compose_file,
        job_id=parsed.job_id,
        evidence_dir=parsed.evidence_dir,
    )


def main() -> int:
    """Run one phase or the full start/cancel/restart/resume/final browser sequence."""
    try:
        args = _arguments()
        base_url, username, password, child_environment = _configuration(args)
        evidence_root = (
            args.evidence_dir.resolve()
            if args.evidence_dir is not None
            else REPOSITORY_ROOT
            / "output"
            / "training"
            / "runtime-smoke"
            / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
        _ = evidence_root.mkdir(parents=True, exist_ok=True)
        phases: Sequence[_SmokePhase] = _ALL_WORKFLOW if args.phase == "all" else (args.phase,)
        context = _BrowserContext(
            evidence_root=evidence_root,
            base_url=base_url,
            username=username,
            password=password,
            environment=child_environment,
        )
        restart_env_file: Path | None = None
        if "interrupt" in phases and "resume" in phases:
            restart_env_file = _safe_environment_file(args.compose_env_file)
            if not args.compose_project:
                raise SmokeConfigurationError(
                    "--compose-project is required for interrupted-resume browser proof"
                )

        reports: list[dict[str, object]] = []
        job_id = args.job_id
        for phase in phases:
            if phase == "resume" and job_id is None:
                job_id = _recorded_job_id(
                    reports,
                    kind="interrupt-me",
                    detail="resume phase needs a prior interrupt job or --job-id",
                )
            if phase == "candidate" and args.job_id is None:
                job_id = _recorded_job_id(
                    reports,
                    kind="succeeded",
                    detail="candidate phase needs a succeeded browser run or --job-id",
                )
            reports.append(
                _run_browser_phase(
                    phase,
                    job_id=job_id if phase in {"resume", "candidate", "coexist"} else None,
                    context=context,
                )
            )
            if phase == "interrupt":
                interrupt_evidence = next(
                    (item for item in reversed(reports) if item.get("kind") == "interrupt-me"),
                    None,
                )
                if interrupt_evidence is None:
                    raise SmokeConfigurationError("interrupt phase did not record its job identity")
                job_id = validate_interrupt_checkpoint_evidence(interrupt_evidence)
                if restart_env_file is not None:
                    reports.append(_restart_supervisor(args, restart_env_file, password))

        report: dict[str, object] = {
            "schema_version": 1,
            "outcome": "passed",
            "base_url": base_url,
            "phases": reports,
        }
        report_path = evidence_root / "training-smoke-report.json"
        _ = report_path.write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        _ = sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
        return 0
    except SmokeConfigurationError as error:
        _ = sys.stderr.write(f"training smoke failed: {error}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
