"""Own Task 8 Docker processes and model repository mounts."""

import sysconfig
from pathlib import Path
from typing import Final

import anyio
from anyio.abc import ByteReceiveStream

from gods_watching.verification.models import ScenarioContextProtocol

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
IMAGE: Final = "gods-watching-triton:25.02"
MODEL_ROOT: Final = REPOSITORY_ROOT / "inference/models"
ASSET_ROOT: Final = REPOSITORY_ROOT / "runtime/assets/models"
QA_ROOT: Final = REPOSITORY_ROOT / "qa/clip"
SERVER_ROOT: Final = REPOSITORY_ROOT / "server/src"
CLIENT_SITE_PACKAGES: Final = Path(sysconfig.get_path("purelib"))
FIXTURE: Final = REPOSITORY_ROOT / "runtime/assets/fixtures/barcelona-boulevard.mp4"


async def run_command(
    context: ScenarioContextProtocol, *, name: str, command: tuple[str, ...]
) -> int:
    """Run one registered command and return its exact status."""
    async with context.process(name=name, command=command) as process:
        return await process.wait()


async def _read_stream(stream: ByteReceiveStream | None, chunks: list[bytes]) -> None:
    if stream is None:
        return
    chunks.extend([chunk async for chunk in stream])


async def run_command_output(
    context: ScenarioContextProtocol, *, name: str, command: tuple[str, ...]
) -> tuple[int, bytes]:
    """Run one registered command and retain stdout plus stderr."""
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    async with context.process(name=name, command=command) as process:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(_read_stream, process.stdout, stdout_chunks)
            task_group.start_soon(_read_stream, process.stderr, stderr_chunks)
        code = await process.wait()
    return code, b"".join((*stdout_chunks, *stderr_chunks))


async def _failure_container(
    context: ScenarioContextProtocol, *, suffix: str, models: Path, revision: str | None = None
) -> bool:
    container = f"{context.compose_project}-task8-{suffix}"
    started = await run_command(
        context,
        name=f"task8-{suffix}-container",
        command=docker_run(
            context, container=container, models=models, selected=("clip_text",), revision=revision
        ),
    )
    ready = await wait_ready(context, container) if started == 0 else 1
    log_path = context.run_root / f"{suffix}.log"
    log_code, log_bytes = await run_command_output(
        context,
        name=f"task8-{suffix}-logs",
        command=("docker", "logs", container),
    )
    _ = log_path.write_bytes(log_bytes)
    await cleanup(context, container)
    return (
        started == 0
        and ready != 0
        and log_code == 0
        and (
            b"clip_snapshot_missing" if suffix == "missing-snapshot" else b"clip_revision_mismatch"
        )
        in log_bytes
    )


failure_container = _failure_container


def docker_run(
    context: ScenarioContextProtocol,
    *,
    container: str,
    models: Path,
    selected: tuple[str, ...],
    revision: str | None = None,
) -> tuple[str, ...]:
    """Build an offline GPU container command for explicitly selected models."""
    command = [
        "docker",
        "run",
        "--detach",
        "--name",
        container,
        "--network",
        "none",
        "--gpus",
        "device=0",
        "--mount",
        f"type=bind,src={MODEL_ROOT},dst=/model-repository,readonly",
        "--mount",
        f"type=bind,src={models},dst=/models,readonly",
        "--mount",
        f"type=bind,src={context.run_root},dst=/evidence",
        "--mount",
        f"type=bind,src={QA_ROOT},dst=/qa,readonly",
        "--mount",
        f"type=bind,src={SERVER_ROOT},dst=/app,readonly",
        "--mount",
        f"type=bind,src={CLIENT_SITE_PACKAGES},dst=/client-site-packages,readonly",
    ]
    if revision is not None:
        command.extend(("--env", f"GW_CLIP_REVISION={revision}"))
    command.extend((IMAGE, "tritonserver", "--model-repository=/model-repository"))
    command.extend(("--model-control-mode=explicit", "--exit-on-error=true"))
    command.extend(f"--load-model={model}" for model in selected)
    return tuple(command)


async def wait_ready(context: ScenarioContextProtocol, container: str) -> int:
    """Wait within a bounded container-side readiness probe."""
    readiness_prefix = "for i in {1..300}; do curl -fsS localhost:8000/v2/health/ready "
    readiness = f"{readiness_prefix}&& exit 0; sleep 1; done; exit 1"
    return await run_command(
        context,
        name=f"{container}-readiness",
        command=("docker", "exec", container, "bash", "-c", readiness),
    )


async def cleanup(context: ScenarioContextProtocol, container: str) -> None:
    """Force-remove only the container registered by this scenario."""
    _ = await run_command(
        context,
        name=f"{container}-cleanup",
        command=("docker", "rm", "--force", container),
    )
