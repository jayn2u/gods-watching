"""Register real retention and retention crash-recovery scenarios."""

import ipaddress
import json
from pathlib import Path
from typing import Final, Literal

import anyio

from gods_watching.verification.models import (
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task10_runtime import (
    CommandResult,
    compose_logs,
    fixture_ip,
    fixture_rtsp_port,
    inspect_resources,
    run_command,
    start_fixtures,
    stop_fixtures,
    wait_fixture_streams,
)
from .task11_runtime import (
    container_ip,
    inspect_containers,
    prepare_database,
    remove_container,
    start_postgres,
    start_triton,
    wait_triton,
)
from .task13_checks import build_checks
from .task13_errors import Task13ExecutionError
from .task13_models import Task13DriverEvidence

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_CLEANUP_STEP_SECONDS: Final = 20.0


async def _run_driver(  # noqa: PLR0913
    context: ScenarioContextProtocol,
    *,
    mode: Literal["retention", "retention-crash"],
    database_url: str,
    rtsp_host: str,
    rtsp_port: int,
    crop_root: Path,
    output: Path,
    worker_log: Path,
) -> CommandResult:
    return await run_command(
        context,
        name=f"task13-{mode}-driver",
        command=(
            "/usr/bin/env",
            "PYTHONPATH=server/src",
            f"GW_TASK13_MODE={mode}",
            f"GW_TASK13_DATABASE_URL={database_url}",
            f"GW_TASK13_TRITON_URL=127.0.0.1:{context.allocated_port}",
            f"GW_TASK13_RTSP_HOST={rtsp_host}",
            f"GW_TASK13_RTSP_PORT={rtsp_port}",
            f"GW_TASK13_CROP_ROOT={crop_root}",
            f"GW_TASK13_OUTPUT={output}",
            f"GW_TASK13_WORKER_LOG={worker_log}",
            "uv",
            "run",
            "--project",
            str(_REPOSITORY_ROOT),
            "python",
            "-m",
            "gods_watching.verification.scenarios.task13_driver",
        ),
    )


def _checked_ip(result: CommandResult, *, detail: str) -> str:
    address = result.stdout.strip()
    if result.return_code != 0:
        raise Task13ExecutionError(detail=detail)
    try:
        _ = ipaddress.ip_address(address)
    except ValueError as error:
        raise Task13ExecutionError(detail=detail) from error
    return address


async def _run(  # noqa: PLR0915
    context: ScenarioContextProtocol, *, mode: Literal["retention", "retention-crash"]
) -> ScenarioReport:
    project = context.compose_project
    triton = f"{project}-task13-triton"
    postgres = f"{project}-task13-postgres"
    output = context.run_root / (
        "task-13-retention.json" if mode == "retention" else "task-13-crash.json"
    )
    crop_root = context.runtime_root / "task13" / "crops"
    worker_log = context.run_root / "task-13-worker.log"
    driver_stdout = context.run_root / "task-13-driver.stdout"
    driver_stderr = context.run_root / "task-13-driver.stderr"
    fixture_log = context.run_root / "task-13-fixtures.log"
    cleanup_path = context.run_root / "task-13-cleanup.json"
    manifest = context.run_root / "task-13-resources.json"
    _ = manifest.write_text(
        json.dumps(
            {
                "compose_project": project,
                "containers": [triton, postgres],
                "crop_root": str(crop_root),
                "grpc_port": context.allocated_port,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    crop_root.parent.mkdir(parents=True, exist_ok=True)
    cleanup_codes: dict[str, int | None] = {
        "fixtures": None,
        "postgres": None,
        "triton": None,
        "inspect": None,
    }
    cleanup_succeeded = False
    try:
        if (await start_fixtures(context, project)).return_code != 0:
            raise Task13ExecutionError(detail="fixture RTSP project failed to start")
        rtsp_host = _checked_ip(
            await fixture_ip(context, project=project),
            detail="fixture RTSP address was unavailable",
        )
        rtsp_port = fixture_rtsp_port()
        if not await wait_fixture_streams(
            context, rtsp_host=rtsp_host, rtsp_port=rtsp_port
        ):
            raise Task13ExecutionError(detail="fixture RTSP streams were not ready")
        if (await start_postgres(context, container=postgres)).return_code != 0:
            raise Task13ExecutionError(detail="disposable PostgreSQL failed to start")
        postgres_host = _checked_ip(
            await container_ip(context, container=postgres),
            detail="disposable PostgreSQL address was unavailable",
        )
        database_url = f"postgresql+asyncpg://postgres@{postgres_host}:5432/gods_watching_task11"
        migration = await prepare_database(context, container=postgres, database_url=database_url)
        if migration.return_code != 0:
            raise Task13ExecutionError(detail="Task 13 migrations failed")
        triton_result = await start_triton(context, container=triton)
        if triton_result.return_code != 0 or not await wait_triton(
            f"127.0.0.1:{context.allocated_port}"
        ):
            raise Task13ExecutionError(detail="detector and CLIP image models were not ready")
        driver = await _run_driver(
            context,
            mode=mode,
            database_url=database_url,
            rtsp_host=rtsp_host,
            rtsp_port=rtsp_port,
            crop_root=crop_root,
            output=output,
            worker_log=worker_log,
        )
        _ = driver_stdout.write_text(driver.stdout, encoding="utf-8")
        _ = driver_stderr.write_text(driver.stderr, encoding="utf-8")
        if driver.return_code != 0 or not output.is_file():
            raise Task13ExecutionError(detail="Task 13 retention driver failed")
        evidence = Task13DriverEvidence.model_validate_json(output.read_text(encoding="utf-8"))
        logs = await compose_logs(context, project=project)
        _ = fixture_log.write_text(logs.stdout + logs.stderr, encoding="utf-8")
    finally:
        with context.suppress_interruptions(), anyio.CancelScope(shield=True):
            with anyio.move_on_after(_CLEANUP_STEP_SECONDS):
                cleanup_codes["triton"] = (
                    await remove_container(context, container=triton)
                ).return_code
            with anyio.move_on_after(_CLEANUP_STEP_SECONDS):
                cleanup_codes["postgres"] = (
                    await remove_container(context, container=postgres)
                ).return_code
            with anyio.move_on_after(_CLEANUP_STEP_SECONDS):
                cleanup_codes["fixtures"] = (await stop_fixtures(context, project)).return_code
            fixture_remaining = await inspect_resources(context, project=project, triton=triton)
            owned_remaining = await inspect_containers(context, containers=(triton, postgres))
            cleanup_codes["inspect"] = max(
                fixture_remaining.return_code, owned_remaining.return_code
            )
            remaining = fixture_remaining.stdout.strip() + owned_remaining.stdout.strip()
            cleanup_succeeded = all(code == 0 for code in cleanup_codes.values()) and not remaining
            _ = cleanup_path.write_text(
                json.dumps(
                    {
                        "exit_codes": cleanup_codes,
                        "remaining_owned_resources": remaining,
                        "succeeded": cleanup_succeeded,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
    return ScenarioReport(
        checks=build_checks(mode, evidence, cleanup_succeeded=cleanup_succeeded),
        artifact_paths=(
            manifest,
            output,
            worker_log,
            driver_stdout,
            driver_stderr,
            fixture_log,
            cleanup_path,
        ),
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_retention(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="retention")


async def _run_crash(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="retention-crash")


for scenario_name, runner in (
    ("retention", _run_retention),
    ("retention-crash", _run_crash),
):
    register_scenario(
        ImplementedScenario(
            name=parse_scenario_name(scenario_name),
            runner=runner,
            required_commands=("docker", "ffmpeg", "ffprobe", "uv"),
            timeout_seconds=420.0,
        )
    )
