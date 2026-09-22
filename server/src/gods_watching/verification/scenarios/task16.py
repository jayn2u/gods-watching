"""Register the real-API person search UI scenario."""

import ipaddress
import json
from pathlib import Path
from typing import Final

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
from .task16_checks import build_checks
from .task16_errors import Task16ExecutionError
from .task16_models import Task16DriverEvidence

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_CLEANUP_STEP_SECONDS: Final = 20.0
_SEARCH_MODELS: Final = ("detector", "clip_image", "clip_text")


def _checked_ip(result: CommandResult, *, detail: str) -> str:
    address = result.stdout.strip()
    if result.return_code != 0:
        raise Task16ExecutionError(detail=detail)
    try:
        _ = ipaddress.ip_address(address)
    except ValueError as error:
        raise Task16ExecutionError(detail=detail) from error
    return address


async def _run(context: ScenarioContextProtocol) -> ScenarioReport:  # noqa: PLR0915
    project = context.compose_project
    triton = f"{project}-task16-triton"
    postgres = f"{project}-task16-postgres"
    output = context.run_root / "task-16-search-ui.json"
    crop_root = context.runtime_root / "task16" / "crops"
    e2e_root = context.run_root / "task-16-playwright"
    worker_log = context.run_root / "task-16-worker.log"
    app_log = context.run_root / "task-16-app.log"
    build_log = context.run_root / "task-16-web-build.log"
    driver_stdout = context.run_root / "task-16-driver.stdout"
    driver_stderr = context.run_root / "task-16-driver.stderr"
    cleanup_path = context.run_root / "task-16-cleanup.json"
    crop_root.parent.mkdir(parents=True, exist_ok=True)
    e2e_root.mkdir(parents=True, exist_ok=True)
    cleanup_codes: dict[str, int | None] = {
        "fixtures": None,
        "postgres": None,
        "triton": None,
        "inspect": None,
    }
    cleanup_succeeded = False
    web_build_succeeded = False
    try:
        build = await run_command(
            context,
            name="task16-web-build",
            command=("pnpm", "--dir", str(_REPOSITORY_ROOT / "web"), "run", "build"),
        )
        _ = build_log.write_text(build.stdout + build.stderr, encoding="utf-8")
        web_build_succeeded = build.return_code == 0
        if not web_build_succeeded:
            raise Task16ExecutionError(detail="production web build failed")
        if (await start_fixtures(context, project)).return_code != 0:
            raise Task16ExecutionError(detail="fixture RTSP project failed to start")
        rtsp_host = _checked_ip(
            await fixture_ip(context, project=project),
            detail="fixture RTSP address was unavailable",
        )
        rtsp_port = fixture_rtsp_port()
        if not await wait_fixture_streams(
            context, rtsp_host=rtsp_host, rtsp_port=rtsp_port
        ):
            raise Task16ExecutionError(detail="fixture RTSP streams were not ready")
        if (await start_postgres(context, container=postgres)).return_code != 0:
            raise Task16ExecutionError(detail="disposable PostgreSQL failed to start")
        postgres_host = _checked_ip(
            await container_ip(context, container=postgres),
            detail="disposable PostgreSQL address was unavailable",
        )
        database_url = f"postgresql+asyncpg://postgres@{postgres_host}:5432/gods_watching_task11"
        if (
            await prepare_database(context, container=postgres, database_url=database_url)
        ).return_code != 0:
            raise Task16ExecutionError(detail="Task 16 migrations failed")
        triton_result = await start_triton(context, container=triton, models=_SEARCH_MODELS)
        if triton_result.return_code != 0 or not await wait_triton(
            f"127.0.0.1:{context.allocated_port}", _SEARCH_MODELS
        ):
            raise Task16ExecutionError(detail="detector and CLIP models were not ready")
        driver = await run_command(
            context,
            name="task16-search-ui-driver",
            command=(
                "/usr/bin/env",
                "PYTHONPATH=server/src",
                f"GW_TASK16_DATABASE_URL={database_url}",
                f"GW_TASK16_TRITON_URL=127.0.0.1:{context.allocated_port}",
                f"GW_TASK16_TRITON_CONTAINER={triton}",
                f"GW_TASK16_RTSP_HOST={rtsp_host}",
                f"GW_TASK16_RTSP_PORT={rtsp_port}",
                f"GW_TASK16_CROP_ROOT={crop_root}",
                f"GW_TASK16_OUTPUT={output}",
                f"GW_TASK16_WORKER_LOG={worker_log}",
                f"GW_TASK16_APP_LOG={app_log}",
                f"GW_TASK16_E2E_ROOT={e2e_root}",
                "uv",
                "run",
                "--project",
                str(_REPOSITORY_ROOT),
                "python",
                "-m",
                "gods_watching.verification.scenarios.task16_driver",
            ),
        )
        _ = driver_stdout.write_text(driver.stdout, encoding="utf-8")
        _ = driver_stderr.write_text(driver.stderr, encoding="utf-8")
        if driver.return_code != 0 or not output.is_file():
            raise Task16ExecutionError(detail="Task 16 search UI driver failed")
        evidence = Task16DriverEvidence.model_validate_json(output.read_text(encoding="utf-8"))
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
        checks=build_checks(
            web_build_succeeded=web_build_succeeded,
            evidence=evidence,
            cleanup_succeeded=cleanup_succeeded,
        ),
        artifact_paths=(
            output,
            build_log,
            worker_log,
            app_log,
            driver_stdout,
            driver_stderr,
            e2e_root,
            cleanup_path,
        ),
        evidence_kind=EvidenceKind.REAL,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("search-ui"),
        runner=_run,
        required_commands=("docker", "ffmpeg", "ffprobe", "uv", "pnpm"),
        timeout_seconds=1_200.0,
    )
)
