import subprocess
import sys
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import anyio
import pytest

from gods_watching.verification import (
    Check,
    EvidenceKind,
    EvidencePathError,
    ImplementedScenario,
    RunResult,
    ScenarioContextProtocol,
    ScenarioReport,
    UnavailableScenario,
    build_registry,
    execute_scenario,
    parse_scenario_name,
)

REPOSITORY_ROOT = Path(__file__).parents[2]
type Runner = Callable[[ScenarioContextProtocol], Awaitable[ScenarioReport]]


def _cli(scenario: str, evidence: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        ["./gods-watching", "verify", "--scenario", scenario, "--evidence", str(evidence)],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


async def _passing_runner(context: ScenarioContextProtocol) -> ScenarioReport:
    return ScenarioReport(
        checks=(Check(name="observed", passed=True, detail=str(context.run_id)),),
        evidence_kind=EvidenceKind.SYNTHETIC,
    )


async def _assertion_runner(context: ScenarioContextProtocol) -> ScenarioReport:
    del context
    message = "rtsp://operator:top-secret@camera password=hunter2"
    raise AssertionError(message)


async def _misleading_stdout_runner(context: ScenarioContextProtocol) -> ScenarioReport:
    command = (sys.executable, "-c", "print('PASS')")
    async with context.process(name="misleading", command=command) as process:
        _ = await process.wait()
    return ScenarioReport(
        checks=(Check(name="real-assertion", passed=False, detail="observable failed"),),
        evidence_kind=EvidenceKind.SYNTHETIC,
    )


async def _hung_runner(context: ScenarioContextProtocol) -> ScenarioReport:
    command = (sys.executable, "-c", "import time; time.sleep(60)")
    async with context.process(name="hung", command=command):
        await anyio.sleep(60)
    return ScenarioReport(checks=(), evidence_kind=EvidenceKind.SYNTHETIC)


def _definition(
    name: str,
    *,
    runner: Runner = _passing_runner,
    dependencies: tuple[str, ...] = (),
    timeout: float = 2.0,
) -> ImplementedScenario:
    return ImplementedScenario(
        name=parse_scenario_name(name),
        runner=runner,
        required_commands=dependencies,
        timeout_seconds=timeout,
    )


def _execute_definition(
    scenario: str, evidence: Path, definition: ImplementedScenario
) -> RunResult:
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            execute_scenario,
            scenario=scenario,
            evidence_dir=evidence,
            repository_root=REPOSITORY_ROOT,
            additional=(definition,),
        )
        return future.result()


def test_evidence_is_written_when_scenario_is_unknown(tmp_path: Path) -> None:
    # Given: a unique empty evidence destination
    evidence = tmp_path / "unknown-evidence"
    # When: the real CLI is asked to run an unknown scenario
    completed = _cli("unknown", evidence)
    # Then: failure is truthful and machine-readable evidence binds source versions
    result = RunResult.model_validate_json((evidence / "result.json").read_text())
    assert completed.returncode != 0
    assert result.outcome == "failed"
    assert result.error is not None
    assert result.error.code == "unknown_scenario"
    assert result.git_sha
    assert result.dirty_diff_sha256
    assert result.source_hashes


def test_nonzero_when_supported_scenario_is_not_implemented(tmp_path: Path) -> None:
    # Given: an explicitly unavailable definition isolated from the production plan
    scenario = "unavailable-contract"
    definition = UnavailableScenario(name=parse_scenario_name(scenario), owner_task=999)
    evidence = tmp_path / "unavailable"
    # When: the harness dispatches that registered definition
    result = execute_scenario(
        scenario=scenario,
        evidence_dir=evidence,
        repository_root=REPOSITORY_ROOT,
        additional=(definition,),
    )
    # Then: it fails explicitly instead of claiming a skipped pass
    assert result.exit_code != 0
    assert result.error is not None
    assert result.error.code == "scenario_unavailable"


def test_missing_dependency_prevents_runner_success(tmp_path: Path) -> None:
    # Given: a scenario that requires a command absent from PATH
    definition = _definition("detector", dependencies=("gw-command-that-does-not-exist",))
    # When: the scenario is executed
    result = _execute_definition("detector", tmp_path / "missing", definition)
    # Then: the missing dependency is a binary failure
    assert result.exit_code != 0
    assert result.error is not None
    assert result.error.code == "missing_dependency"


def test_nonzero_and_redacted_when_runner_assertion_fails(tmp_path: Path) -> None:
    # Given: a runner whose assertion includes credential-bearing text
    definition = _definition("detector", runner=_assertion_runner)
    # When: the assertion reaches the harness boundary
    result = _execute_definition("detector", tmp_path / "assertion", definition)
    # Then: failure remains nonzero and evidence contains no credential
    evidence_text = (tmp_path / "assertion/result.json").read_text()
    assert result.exit_code != 0
    assert "top-secret" not in evidence_text
    assert "hunter2" not in evidence_text
    assert "[REDACTED]" in evidence_text


def test_nonzero_when_stdout_claims_pass_but_check_fails(tmp_path: Path) -> None:
    # Given: a real process that prints PASS and a failing observable check
    definition = _definition("detector", runner=_misleading_stdout_runner)
    # When: the scenario completes normally
    result = _execute_definition("detector", tmp_path / "misleading", definition)
    # Then: structured assertions determine the nonzero result
    assert result.exit_code != 0
    assert result.error is not None
    assert result.error.code == "failed_checks"


def test_cleanup_receipt_exists_after_deadline_cancellation(tmp_path: Path) -> None:
    # Given: a task-owned child process whose scenario exceeds its deadline
    definition = _definition("detector", runner=_hung_runner, timeout=0.05)
    evidence = tmp_path / "cancelled"
    # When: the harness cancels the scenario
    result = _execute_definition("detector", evidence, definition)
    # Then: cancellation fails and the registered process has a cleanup receipt
    manifest = (evidence / "resource-manifest.json").read_text()
    assert result.error is not None
    assert result.error.code == "cancelled"
    registered = manifest.index('"name": "hung"')
    started = manifest.index('"name": "hung"', registered + 1)
    cleaned = manifest.index('"name": "hung"', started + 1)
    assert registered < started < cleaned
    assert not (evidence / "runtime").exists()


def test_isolation_names_are_unique_for_concurrent_runs(tmp_path: Path) -> None:
    # Given: two independent evidence roots and one implemented definition
    definition = _definition("detector")
    roots = (tmp_path / "first", tmp_path / "second")
    # When: separate threads execute the scenario concurrently
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = tuple(
            pool.submit(
                execute_scenario,
                scenario="detector",
                evidence_dir=root,
                repository_root=REPOSITORY_ROOT,
                additional=(definition,),
            )
            for root in roots
        )
        results = tuple(future.result() for future in futures)
    # Then: run IDs, Compose projects, ports, and roots do not overlap
    assert len({result.run_id for result in results}) == 2
    assert len({result.compose_project for result in results}) == 2
    assert len({result.allocated_port for result in results}) == 2
    assert len({result.run_root for result in results}) == 2


def test_malformed_scenario_emits_failure_evidence(tmp_path: Path) -> None:
    # Given: a malformed scenario and an empty destination
    evidence = tmp_path / "malformed"
    # When: the real CLI parses the request
    completed = _cli("../bad scenario", evidence)
    # Then: the request is rejected without reflecting hostile input
    result = RunResult.model_validate_json((evidence / "result.json").read_text())
    assert completed.returncode != 0
    assert result.error is not None
    assert result.error.code == "invalid_scenario"
    assert "../bad scenario" not in result.error.message


def test_stale_nonempty_evidence_is_not_overwritten(tmp_path: Path) -> None:
    # Given: a nonempty destination containing an earlier sentinel
    evidence = tmp_path / "stale"
    evidence.mkdir()
    sentinel = evidence / "sentinel.txt"
    _ = sentinel.write_text("preserve", encoding="utf-8")
    # When: a new run targets that destination
    with pytest.raises(EvidencePathError):
        _ = execute_scenario(
            scenario="harness-self-check",
            evidence_dir=evidence,
            repository_root=REPOSITORY_ROOT,
        )
    # Then: no stale artifact is reused or overwritten
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert not (evidence / "result.json").exists()


def test_nonzero_when_evidence_path_is_a_file(tmp_path: Path) -> None:
    # Given: a file where an evidence directory is required
    evidence = tmp_path / "not-a-directory"
    _ = evidence.write_text("preserve", encoding="utf-8")
    # When: the real CLI validates the malformed evidence boundary
    completed = _cli("harness-self-check", evidence)
    # Then: it rejects the path and preserves the existing file
    assert completed.returncode == 2
    assert '"code":"invalid_evidence"' in completed.stdout
    assert evidence.read_text(encoding="utf-8") == "preserve"


def test_registry_contains_every_plan_scenario() -> None:
    # Given: the run-local default registry
    registry = build_registry()
    # When: its supported names are read
    names = set(registry.names())
    # Then: representative names from every owning wave are registered
    assert {"model-assets", "fixture-inputs", "full", "faults", "search"} <= names
    assert "harness-self-check" in names
    assert len(names) == 33


def test_evidence_happy_self_check_uses_real_owned_process(tmp_path: Path) -> None:
    # Given: the built-in harness self-check and an unrelated external marker
    evidence = tmp_path / "self-check"
    external = tmp_path / "external-marker"
    _ = external.write_text("owned elsewhere", encoding="utf-8")
    # When: the real CLI drives the disposable-process scenario
    completed = _cli("harness-self-check", evidence)
    # Then: it passes, cleans only its resources, and records the receipt
    result = RunResult.model_validate_json((evidence / "result.json").read_text())
    manifest = (evidence / "resource-manifest.json").read_text()
    assert completed.returncode == 0
    assert result.outcome == "passed"
    assert "self-check-process" in manifest
    assert '"state": "cleaned"' in manifest
    assert external.read_text(encoding="utf-8") == "owned elsewhere"
