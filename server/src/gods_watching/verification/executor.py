"""Run one verification scenario and persist its terminal evidence."""

import hashlib
import os
import secrets
import shutil
import subprocess
import time
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Final

import anyio

from .context import (
    ResourceLedger,
    ScenarioContext,
    ScenarioContextConfig,
    allocate_loopback_port,
)
from .errors import EvidencePathError, InvalidScenarioNameError
from .models import (
    Check,
    ErrorRecord,
    EvidenceKind,
    RunId,
    RunResult,
    ScenarioDefinition,
    ScenarioReport,
    SourceHash,
)
from .redaction import redact
from .registry import build_registry, parse_scenario_name

_HARNESS_VERSION: Final = "1.0"
_RESULT_SCHEMA_VERSION: Final = "1"


def _prepare_evidence(path: Path) -> Path:
    if path.is_symlink():
        raise EvidencePathError(path=path, reason="symbolic links are not accepted")
    if path.exists() and not path.is_dir():
        raise EvidencePathError(path=path, reason="path is not a directory")
    if path.exists() and next(path.iterdir(), None) is not None:
        raise EvidencePathError(path=path, reason="directory must be empty")
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _run_identity(repository_root: Path) -> tuple[str, str]:
    head = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["/usr/bin/git", "diff", "--binary", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    status = subprocess.run(
        ["/usr/bin/git", "status", "--porcelain=v1", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    return head, hashlib.sha256(diff + status).hexdigest()


def _source_hashes(repository_root: Path) -> tuple[SourceHash, ...]:
    package_root = repository_root / "server/src/gods_watching"
    paths = (*sorted((package_root / "verification").rglob("*.py")), package_root / "cli.py")
    return tuple(
        SourceHash(
            path=str(path.relative_to(repository_root)),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in paths
    )


def _safe_report(report: ScenarioReport) -> ScenarioReport:
    return ScenarioReport(
        checks=tuple(
            Check(name=redact(check.name), passed=check.passed, detail=redact(check.detail))
            for check in report.checks
        ),
        artifact_paths=tuple(Path(redact(str(path))) for path in report.artifact_paths),
        evidence_kind=report.evidence_kind,
    )


async def _invoke(
    definition: ScenarioDefinition, context: ScenarioContext
) -> tuple[ScenarioReport | None, ErrorRecord | None]:
    missing = tuple(
        command for command in definition.required_commands if shutil.which(command) is None
    )
    if missing:
        return None, ErrorRecord(
            code="missing_dependency", message=f"required command unavailable: {', '.join(missing)}"
        )
    try:
        with anyio.fail_after(definition.timeout_seconds):
            return _safe_report(await definition.invoke(context)), None
    except TimeoutError:
        return None, ErrorRecord(code="cancelled", message="scenario deadline exceeded")
    except AssertionError as error:
        return None, ErrorRecord(code="assertion_failed", message=redact(str(error)))
    except (OSError, RuntimeError) as error:
        return None, ErrorRecord(code="runner_error", message=redact(str(error)))


async def _execute(
    *,
    scenario: str,
    evidence_dir: Path,
    repository_root: Path,
    additional: Iterable[ScenarioDefinition],
) -> RunResult:
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    run_id = RunId(secrets.token_hex(12))
    compose_project = f"gw-verify-{os.getpid()}-{run_id[:12]}"
    port = allocate_loopback_port()
    ledger = ResourceLedger(evidence_dir / "resource-manifest.json")
    runtime_root = evidence_dir / "runtime"
    ledger.append(
        kind="directory", name="runtime-root", state="registered", detail=str(runtime_root)
    )
    runtime_root.mkdir()
    ledger.append(kind="directory", name="runtime-root", state="started")
    context = ScenarioContext(
        ScenarioContextConfig(
            run_id=run_id,
            run_root=evidence_dir,
            runtime_root=runtime_root,
            compose_project=compose_project,
            allocated_port=port,
        ),
        ledger=ledger,
    )
    registry = build_registry(additional)
    report: ScenarioReport | None = None
    error: ErrorRecord | None = None
    try:
        try:
            parsed_name = parse_scenario_name(scenario)
        except InvalidScenarioNameError as invalid:
            error = ErrorRecord(code="invalid_scenario", message=str(invalid))
        else:
            definition = registry.get(parsed_name)
            if definition is None:
                error = ErrorRecord(
                    code="unknown_scenario", message="scenario is not registered"
                )
            elif definition.owner_task is not None:
                error = ErrorRecord(
                    code="scenario_unavailable",
                    message=f"scenario awaits implementation by task {definition.owner_task}",
                )
            else:
                report, error = await _invoke(definition, context)
    finally:
        shutil.rmtree(runtime_root)
        ledger.append(kind="directory", name="runtime-root", state="cleaned")

    checks = () if report is None else report.checks
    if error is None and not checks:
        error = ErrorRecord(code="missing_assertions", message="scenario returned no assertions")
    if error is None and any(not check.passed for check in checks):
        error = ErrorRecord(code="failed_checks", message="one or more scenario assertions failed")
    passed = error is None
    finished_at = datetime.now(UTC)
    git_sha, dirty_diff_sha256 = _run_identity(repository_root)
    result = RunResult(
        schema_version=_RESULT_SCHEMA_VERSION,
        harness_version=_HARNESS_VERSION,
        run_id=run_id,
        scenario=redact(scenario),
        outcome="passed" if passed else "failed",
        exit_code=0 if passed else 1,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.monotonic() - started_monotonic,
        git_sha=git_sha,
        dirty_diff_sha256=dirty_diff_sha256,
        source_hashes=_source_hashes(repository_root),
        run_root=Path(redact(str(evidence_dir))),
        compose_project=compose_project,
        allocated_port=port,
        checks=checks,
        artifact_paths=(Path(redact(str(evidence_dir / "resource-manifest.json"))),)
        if report is None
        else (
            Path(redact(str(evidence_dir / "resource-manifest.json"))),
            *report.artifact_paths,
        ),
        error=error,
        evidence_kind=EvidenceKind.SYNTHETIC if report is None else report.evidence_kind,
    )
    _ = (evidence_dir / "result.json").write_text(
        result.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    return result


def execute_scenario(
    *,
    scenario: str,
    evidence_dir: Path,
    repository_root: Path,
    additional: Iterable[ScenarioDefinition] = (),
) -> RunResult:
    """Validate the boundary, execute the scenario, and return its persisted result."""
    prepared = _prepare_evidence(evidence_dir)
    operation = partial(
        _execute,
        scenario=scenario,
        evidence_dir=prepared,
        repository_root=repository_root.resolve(),
        additional=additional,
    )
    return anyio.run(operation)
