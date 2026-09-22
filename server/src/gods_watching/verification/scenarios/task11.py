"""Register real active-publication and stale-completion scenarios."""

import ipaddress
import json
from typing import Literal

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
from .task11_checks import build_checks
from .task11_errors import Task11ExecutionError
from .task11_models import Task11DriverEvidence
from .task11_runtime import (
    Task11Resources,
    container_ip,
    inspect_containers,
    prepare_database,
    remove_container,
    run_driver,
    start_postgres,
    start_triton,
    wait_triton,
)


async def _run(  # noqa: C901, PLR0912, PLR0915
    context: ScenarioContextProtocol, *, mode: Literal["appearance", "appearance-stale"]
) -> ScenarioReport:
    project = context.compose_project
    triton = f"{project}-task11-triton"
    postgres = f"{project}-task11-postgres"
    output = context.run_root / (
        "task-11-versions.json" if mode == "appearance" else "task-11-recovery.json"
    )
    crop_root = context.runtime_root / "task11-crops"
    manifest = context.run_root / "task-11-resources.json"
    cleanup_path = context.run_root / "task-11-cleanup.json"
    driver_stdout = context.run_root / "task-11-driver.stdout"
    driver_stderr = context.run_root / "task-11-driver.stderr"
    triton_log = context.run_root / "task-11-triton.log"
    fixture_log = context.run_root / "task-11-fixtures.log"
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
    fixtures_owned = True
    postgres_owned = True
    triton_owned = True
    cleanup_codes: dict[str, int | None] = {
        "fixtures": None,
        "postgres": None,
        "triton": None,
        "inspect": None,
    }
    cleanup_succeeded = False
    try:
        fixtures = await start_fixtures(context, project)
        if fixtures.return_code != 0:
            raise Task11ExecutionError(detail="fixture RTSP project failed to start")
        fixture_address = await fixture_ip(context, project=project)
        if fixture_address.return_code != 0:
            raise Task11ExecutionError(detail="fixture RTSP address was unavailable")
        rtsp_host = fixture_address.stdout.strip()
        try:
            _ = ipaddress.ip_address(rtsp_host)
        except ValueError as error:
            raise Task11ExecutionError(detail="fixture RTSP address was invalid") from error
        if not await wait_fixture_streams(
            context, rtsp_host=rtsp_host, rtsp_port=fixture_rtsp_port()
        ):
            raise Task11ExecutionError(detail="fixture RTSP stream was not ready")

        postgres_result = await start_postgres(context, container=postgres)
        if postgres_result.return_code != 0:
            raise Task11ExecutionError(detail="disposable PostgreSQL failed to start")
        postgres_address = await container_ip(context, container=postgres)
        if postgres_address.return_code != 0:
            raise Task11ExecutionError(detail="disposable PostgreSQL address was unavailable")
        postgres_host = postgres_address.stdout.strip()
        try:
            _ = ipaddress.ip_address(postgres_host)
        except ValueError as error:
            raise Task11ExecutionError(
                detail="disposable PostgreSQL address was invalid"
            ) from error
        database_url = f"postgresql+asyncpg://postgres@{postgres_host}:5432/gods_watching_task11"
        migration = await prepare_database(context, container=postgres, database_url=database_url)
        if migration.return_code != 0:
            raise Task11ExecutionError(detail="Task 11 migrations failed")

        triton_result = await start_triton(context, container=triton)
        endpoint = f"127.0.0.1:{context.allocated_port}"
        if triton_result.return_code != 0 or not await wait_triton(endpoint):
            raise Task11ExecutionError(detail="detector and CLIP image models were not ready")
        resources = Task11Resources(
            triton=triton,
            postgres=postgres,
            database_url=database_url,
        )
        driver = await run_driver(
            context,
            mode=mode,
            resources=resources,
            rtsp_host=rtsp_host,
            output=output,
            crop_root=crop_root,
        )
        _ = driver_stdout.write_text(driver.stdout, encoding="utf-8")
        _ = driver_stderr.write_text(driver.stderr, encoding="utf-8")
        if driver.return_code != 0 or not output.is_file():
            raise Task11ExecutionError(detail="Task 11 production driver failed")
        evidence = Task11DriverEvidence.model_validate_json(output.read_text(encoding="utf-8"))
        logs = await compose_logs(context, project=project)
        _ = fixture_log.write_text(logs.stdout + logs.stderr, encoding="utf-8")
        triton_logs = await inspect_triton_log(context, container=triton)
        _ = triton_log.write_text(triton_logs.stdout + triton_logs.stderr, encoding="utf-8")
    finally:
        with context.suppress_interruptions():
            with anyio.CancelScope(shield=True):
                if triton_owned:
                    with anyio.move_on_after(20.0):
                        cleanup_codes["triton"] = (
                            await remove_container(context, container=triton)
                        ).return_code
                if postgres_owned:
                    with anyio.move_on_after(20.0):
                        cleanup_codes["postgres"] = (
                            await remove_container(context, container=postgres)
                        ).return_code
                if fixtures_owned:
                    with anyio.move_on_after(20.0):
                        cleanup_codes["fixtures"] = (
                            await stop_fixtures(context, project)
                        ).return_code
                task10_remaining = await inspect_resources(context, project=project, triton=triton)
                task11_remaining = await inspect_containers(context, containers=(triton, postgres))
                cleanup_codes["inspect"] = max(
                    task10_remaining.return_code, task11_remaining.return_code
                )
                remaining = task10_remaining.stdout.strip() + task11_remaining.stdout.strip()
                cleanup_succeeded = (
                    cleanup_codes["triton"] == 0
                    and cleanup_codes["postgres"] == 0
                    and cleanup_codes["fixtures"] == 0
                    and cleanup_codes["inspect"] == 0
                    and not remaining
                )
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
            driver_stdout,
            driver_stderr,
            fixture_log,
            triton_log,
            cleanup_path,
        ),
        evidence_kind=EvidenceKind.REAL,
    )


async def inspect_triton_log(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Capture the combined model server log before exact-name cleanup."""
    return await run_command(
        context, name="task11-triton-logs", command=("docker", "logs", container)
    )


async def _run_appearance(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="appearance")


async def _run_stale(context: ScenarioContextProtocol) -> ScenarioReport:
    return await _run(context, mode="appearance-stale")


for scenario_name, runner in (
    ("appearance", _run_appearance),
    ("appearance-stale", _run_stale),
):
    register_scenario(
        ImplementedScenario(
            name=parse_scenario_name(scenario_name),
            runner=runner,
            required_commands=("docker", "ffmpeg", "ffprobe", "uv"),
            timeout_seconds=300.0,
        )
    )
