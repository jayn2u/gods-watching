"""Register real authenticated search and search error-path scenarios."""

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
    fixture_ip,
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
from .task14_checks import build_checks
from .task14_errors import Task14ExecutionError
from .task14_models import Task14DriverEvidence

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_CLEANUP_STEP_SECONDS: Final = 20.0
_SEARCH_MODELS: Final = ("detector", "clip_image", "clip_text")


def _checked_ip(result: CommandResult, *, detail: str) -> str:
    address = result.stdout.strip()
    if result.return_code != 0:
        raise Task14ExecutionError(detail=detail)
    try:
        _ = ipaddress.ip_address(address)
    except ValueError as error:
        raise Task14ExecutionError(detail=detail) from error
    return address


async def _run(
    context: ScenarioContextProtocol, *, mode: Literal["search", "search-errors"]
) -> ScenarioReport:
    project = context.compose_project
    triton = f"{project}-task14-triton"
    postgres = f"{project}-task14-postgres"
    output = context.run_root / (
        "task-14-search.json" if mode == "search" else "task-14-errors.json"
    )
    crop_root = context.runtime_root / "task14" / "crops"
    worker_log = context.run_root / "task-14-worker.log"
    app_log = context.run_root / "task-14-app.log"
    driver_stdout = context.run_root / "task-14-driver.stdout"
    driver_stderr = context.run_root / "task-14-driver.stderr"
    cleanup_path = context.run_root / "task-14-cleanup.json"
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
            raise Task14ExecutionError(detail="fixture RTSP project failed to start")
        rtsp_host = _checked_ip(
            await fixture_ip(context, project=project),
            detail="fixture RTSP address was unavailable",
        )
        if not await wait_fixture_streams(context, rtsp_host=rtsp_host):
            raise Task14ExecutionError(detail="fixture RTSP streams were not ready")
        if (await start_postgres(context, container=postgres)).return_code != 0:
            raise Task14ExecutionError(detail="disposable PostgreSQL failed to start")
        postgres_host = _checked_ip(
            await container_ip(context, container=postgres),
            detail="disposable PostgreSQL address was unavailable",
        )
        database_url = f"postgresql+asyncpg://postgres@{postgres_host}:5432/gods_watching_task11"
        if (
            await prepare_database(context, container=postgres, database_url=database_url)
        ).return_code != 0:
            raise Task14ExecutionError(detail="Task 14 migrations failed")
        triton_result = await start_triton(context, container=triton, models=_SEARCH_MODELS)
        if triton_result.return_code != 0 or not await wait_triton(
            f"127.0.0.1:{context.allocated_port}", _SEARCH_MODELS
        ):
            raise Task14ExecutionError(detail="detector and CLIP models were not ready")
        driver = await run_command(
            context,
            name=f"task14-{mode}-driver",
            command=(
                "/usr/bin/env",
                "PYTHONPATH=server/src",
                f"GW_TASK14_MODE={mode}",
                f"GW_TASK14_DATABASE_URL={database_url}",
                f"GW_TASK14_TRITON_URL=127.0.0.1:{context.allocated_port}",
                f"GW_TASK14_TRITON_CONTAINER={triton}",
                f"GW_TASK14_RTSP_HOST={rtsp_host}",
                f"GW_TASK14_CROP_ROOT={crop_root}",
                f"GW_TASK14_OUTPUT={output}",
                f"GW_TASK14_WORKER_LOG={worker_log}",
                f"GW_TASK14_APP_LOG={app_log}",
                "uv",
                "run",
                "--project",
                str(_REPOSITORY_ROOT),
                "python",
                "-m",
                "gods_watching.verification.scenarios.task14_driver",
            ),
        )
        _ = driver_stdout.write_text(driver.stdout, encoding="utf-8")
        _ = driver_stderr.write_text(driver.stderr, encoding="utf-8")
        if driver.return_code != 0 or not output.is_file():
            raise Task14ExecutionError(detail="Task 14 search driver failed")
        evidence = Task14DriverEvidence.model_validate_json(output.read_text(encoding="utf-8"))
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
        artifact_paths=(output, worker_log, app_log, driver_stdout, driver_stderr, cleanup_path),
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_search(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="search")


async def _run_errors(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="search-errors")


for scenario_name, runner in (("search", _run_search), ("search-errors", _run_errors)):
    register_scenario(
        ImplementedScenario(
            name=parse_scenario_name(scenario_name),
            runner=runner,
            required_commands=("docker", "ffmpeg", "ffprobe", "uv"),
            timeout_seconds=420.0,
        )
    )
