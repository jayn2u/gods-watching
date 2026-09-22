"""Register the installed real RTSP and outage verification scenarios."""

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final, Literal

import anyio

from gods_watching.verification.context import VerificationInterruptedError
from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task10_artifacts import AdversarialPaths, fixture_provenance, write_adversarial_artifact
from .task10_checks import build_checks
from .task10_control import DriverRunConfig, run_driver
from .task10_errors import Task10ExecutionError
from .task10_models import CleanupEvidence, DriverEvidence, SourceControlEvidence
from .task10_runtime import (
    FIXTURE_RTSP_HOST,
    CommandResult,
    compose_logs,
    fixture_rtsp_port,
    inspect_resources,
    remove_triton,
    start_fixtures,
    start_triton,
    stop_fixtures,
    wait_fixture_streams,
    wait_triton,
)

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_TRITON_IMAGE: Final = "gods-watching-triton:25.02"
_CLEANUP_COMMAND_TIMEOUT_SECONDS: Final = 20.0


async def _bounded_cleanup(
    operation: Callable[[], Awaitable[CommandResult]], *, name: str
) -> CommandResult:
    result = CommandResult(return_code=-1, stdout="", stderr=f"{name} did not complete")
    with anyio.move_on_after(_CLEANUP_COMMAND_TIMEOUT_SECONDS, shield=True) as scope:
        try:
            result = await operation()
        except (OSError, RuntimeError, VerificationInterruptedError) as error:
            return CommandResult(
                return_code=-1,
                stdout="",
                stderr=f"{name} failed with {type(error).__name__}",
            )
    if scope.cancelled_caught:
        return CommandResult(
            return_code=-1,
            stdout="",
            stderr=f"{name} timed out after {_CLEANUP_COMMAND_TIMEOUT_SECONDS:.1f}s",
        )
    return result


def cleanup_check(cleanup: CleanupEvidence) -> Check:  # noqa: D103
    passed = (
        (cleanup.triton_remove_exit_code is None or cleanup.triton_remove_exit_code == 0)
        and (cleanup.compose_down_exit_code is None or cleanup.compose_down_exit_code == 0)
        and cleanup.inspect_return_code == 0
        and not cleanup.remaining_owned_resources.strip()
    )
    detail = (
        f"triton_rm={cleanup.triton_remove_exit_code}; "
        f"compose_down={cleanup.compose_down_exit_code}; "
        f"inspect={cleanup.inspect_return_code}; "
        f"remaining={cleanup.remaining_owned_resources.strip()!r}"
    )
    return Check(name="owned-resource-cleanup", passed=passed, detail=detail)


def _path_set(context: ScenarioContextProtocol, triton: str) -> dict[str, Path]:
    return {
        "driver": context.run_root / "task10-driver.json",
        "source_control": context.run_root / "task10-source-control.json",
        "driver_stdout": context.run_root / "task10-driver.stdout",
        "driver_stderr": context.run_root / "task10-driver.stderr",
        "compose_log": context.run_root / "task10-fixtures.log",
        "provenance": context.run_root / "task10-provenance.json",
        "cleanup": context.run_root / "task10-cleanup.json",
        "slow_signal": context.run_root / "task10-slow-camera-2.started",
        "registration": context.run_root / "task10-resource-registration.json",
        "triton": Path(triton),
    }


def _register_resources(context: ScenarioContextProtocol, paths: dict[str, Path]) -> None:
    payload = {
        "compose_project": context.compose_project,
        "triton_container": str(paths["triton"]),
        "compose_file": "deploy/compose.fixtures.yaml",
        "allocated_grpc_port": context.allocated_port,
        "resources": [
            "fixture-mediamtx",
            "fixture-camera-1..4",
            str(paths["triton"]),
        ],
    }
    _ = paths["registration"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def _run(  # noqa: PLR0915
    context: ScenarioContextProtocol, *, mode: Literal["ingest", "ingest-outage"]
) -> ScenarioReport:
    project = context.compose_project
    triton = f"{project}-triton"
    paths = _path_set(context, triton)
    _register_resources(context, paths)
    fixture_started = False
    triton_started = False
    triton_remove_exit_code: int | None = None
    compose_down_exit_code: int | None = None
    cleanup = CleanupEvidence()
    driver: DriverEvidence
    try:
        # Register ownership before awaiting a command whose side effect may
        # complete just before cancellation reaches this task.
        fixture_started = True
        fixture_result = await start_fixtures(context, project)
        if fixture_result.return_code != 0:
            raise Task10ExecutionError(detail="fixture Compose project failed to start")
        rtsp_host = FIXTURE_RTSP_HOST
        rtsp_port = fixture_rtsp_port()
        if not await wait_fixture_streams(context, rtsp_host=rtsp_host, rtsp_port=rtsp_port):
            raise Task10ExecutionError(detail="fixture publishers did not expose all RTSP streams")
        triton_started = True
        triton_result = await start_triton(context, container=triton)
        if triton_result.return_code != 0 or not await wait_triton(context):
            raise Task10ExecutionError(detail="pinned Triton detector did not become ready")
        provenance = fixture_provenance()
        provenance_json = (
            f'{{\n  "label": "REAL",\n  "triton_image": "{_TRITON_IMAGE}",\n'
            f'  "triton_start_exit_code": {triton_result.return_code},\n'
            f'  "fixture_ip_observed": "{rtsp_host}",\n'
            f'  "assets": {provenance.model_dump_json()}\n}}\n'
        )
        _ = paths["provenance"].write_text(
            provenance_json,
            encoding="utf-8",
        )
        driver_result = await run_driver(
            context,
            DriverRunConfig(
                mode=mode,
                rtsp_host=rtsp_host,
                rtsp_port=rtsp_port,
                output_path=paths["driver"],
                slow_signal=paths["slow_signal"],
                project=project,
                source_control_path=paths["source_control"],
                triton_port=context.allocated_port,
                repository_root=_REPOSITORY_ROOT,
            ),
        )
        _ = paths["driver_stdout"].write_text(driver_result.stdout, encoding="utf-8")
        _ = paths["driver_stderr"].write_text(driver_result.stderr, encoding="utf-8")
        if driver_result.return_code != 0:
            raise Task10ExecutionError(detail="real Task 10 driver failed")
        driver = DriverEvidence.model_validate_json(paths["driver"].read_text(encoding="utf-8"))
        logs = await compose_logs(context, project=project)
        _ = paths["compose_log"].write_text(logs.stdout + logs.stderr, encoding="utf-8")
    finally:
        with context.suppress_interruptions():
            with anyio.CancelScope(shield=True):
                if triton_started:
                    triton_result = await _bounded_cleanup(
                        lambda: remove_triton(context, container=triton), name="Triton removal"
                    )
                    triton_remove_exit_code = triton_result.return_code
                if fixture_started:
                    compose_result = await _bounded_cleanup(
                        lambda: stop_fixtures(context, project), name="Compose cleanup"
                    )
                    compose_down_exit_code = compose_result.return_code
                remaining = await _bounded_cleanup(
                    lambda: inspect_resources(context, project=project, triton=triton),
                    name="Resource inspection",
                )
                cleanup = CleanupEvidence(
                    triton_remove_exit_code=triton_remove_exit_code,
                    compose_down_exit_code=compose_down_exit_code,
                    inspect_return_code=remaining.return_code,
                    remaining_owned_resources=remaining.stdout.strip(),
                )
                _ = paths["cleanup"].write_text(
                    cleanup.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
    source_control = (
        SourceControlEvidence.model_validate_json(
            paths["source_control"].read_text(encoding="utf-8")
        )
        if paths["source_control"].exists()
        else None
    )
    adversarial = write_adversarial_artifact(
        context,
        driver,
        paths=AdversarialPaths(
            provenance=paths["provenance"],
            cleanup=paths["cleanup"],
        ),
    )
    artifacts = (
        paths["registration"],
        paths["provenance"],
        paths["driver"],
        paths["driver_stdout"],
        paths["driver_stderr"],
        paths["compose_log"],
        paths["cleanup"],
        adversarial,
    )
    if source_control is not None:
        artifacts += (paths["source_control"],)
    return ScenarioReport(
        checks=(*build_checks(mode, driver, source_control), cleanup_check(cleanup)),
        artifact_paths=artifacts,
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_ingest(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="ingest")


async def _run_outage(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="ingest-outage")


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("ingest"),
        runner=_run_ingest,
        required_commands=("docker", "ffmpeg", "ffprobe", "uv"),
        timeout_seconds=180.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("ingest-outage"),
        runner=_run_outage,
        required_commands=("docker", "ffmpeg", "ffprobe", "uv"),
        timeout_seconds=240.0,
    )
)
