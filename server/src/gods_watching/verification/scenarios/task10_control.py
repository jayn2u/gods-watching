"""Coordinate the installed Task 10 driver and its controlled outage."""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import anyio

from gods_watching.verification.models import ScenarioContextProtocol

from .task10_runtime import CommandResult, compose_command, run_command


@dataclass(frozen=True, slots=True)
class DriverRunConfig:
    """Group the immutable inputs for one installed driver process."""

    mode: str
    rtsp_host: str
    rtsp_port: int
    output_path: Path
    slow_signal: Path
    project: str
    source_control_path: Path
    triton_port: int
    repository_root: Path


async def _source_interruption(context: ScenarioContextProtocol, config: DriverRunConfig) -> None:
    started = datetime.now(UTC)
    waited = 0.0
    with anyio.move_on_after(12.0) as scope:
        while not config.slow_signal.exists():
            await anyio.sleep(0.1)
            waited += 0.1
    signal_seen = not scope.cancel_called
    stop_result = await run_command(
        context,
        name="task10-outage-source-stop",
        command=compose_command(config.project, "stop", "fixture-camera-2"),
    )
    await anyio.sleep(2.0)
    restart_result = await run_command(
        context,
        name="task10-outage-source-restart",
        command=compose_command(config.project, "up", "-d", "fixture-camera-2"),
    )
    _ = config.source_control_path.write_text(
        json.dumps(
            {
                "started_utc": started.isoformat(),
                "waited_seconds": waited,
                "slow_signal_seen": signal_seen,
                "stop_exit_code": stop_result.return_code,
                "restart_exit_code": restart_result.return_code,
                "source_service": "fixture-camera-2",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _driver_command(config: DriverRunConfig) -> tuple[str, ...]:
    return (
        "/usr/bin/env",
        "PYTHONPATH=server/src",
        f"GW_TASK10_MODE={config.mode}",
        f"GW_TASK10_RTSP_HOST={config.rtsp_host}",
        f"GW_TASK10_RTSP_PORT={config.rtsp_port}",
        f"GW_TASK10_TRITON_URL=127.0.0.1:{config.triton_port}",
        f"GW_TASK10_OUTPUT={config.output_path}",
        f"GW_TASK10_SLOW_SIGNAL={config.slow_signal}",
        f"GW_TASK10_REPOSITORY_ROOT={config.repository_root}",
        "GW_TASK10_DURATION=20",
        "uv",
        "run",
        "--project",
        str(config.repository_root),
        "python",
        "-m",
        "gods_watching.verification.scenarios.task10_driver",
    )


async def run_driver(context: ScenarioContextProtocol, config: DriverRunConfig) -> CommandResult:
    """Run the real driver and, in outage mode, restart only camera two."""
    config.slow_signal.unlink(missing_ok=True)
    command = _driver_command(config)
    if config.mode == "ingest-outage":
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_source_interruption, context, config)
            return await run_command(context, name="task10-outage-driver", command=command)
    return await run_command(context, name="task10-ingest-driver", command=command)


__all__ = ["DriverRunConfig", "run_driver"]
