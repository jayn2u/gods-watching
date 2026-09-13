"""Own disposable PostgreSQL and combined Triton resources for Task 11."""

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import anyio
from tritonclient.grpc.aio import InferenceServerClient
from tritonclient.utils import InferenceServerException

from gods_watching.verification.models import ScenarioContextProtocol

from .task10_runtime import CommandResult, run_command

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_MODEL_ROOT: Final = _REPOSITORY_ROOT / "inference/models"
_ASSET_ROOT: Final = _REPOSITORY_ROOT / "runtime/assets/models"
_TRITON_IMAGE: Final = "gods-watching-triton:25.02"
_POSTGRES_IMAGE: Final = "pgvector/pgvector:0.8.1-pg17"


@dataclass(frozen=True, slots=True)
class Task11Resources:
    """Name the exact containers and database endpoint owned by one run."""

    triton: str
    postgres: str
    database_url: str


async def start_triton(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Start detector and image CLIP models on the admitted GPU."""
    return await run_command(
        context,
        name="task11-triton-up",
        command=(
            "docker",
            "run",
            "--detach",
            "--name",
            container,
            "--gpus",
            "device=0",
            "--shm-size=1g",
            "--network",
            "bridge",
            "-p",
            f"127.0.0.1:{context.allocated_port}:8001",
            "--mount",
            f"type=bind,src={_MODEL_ROOT},dst=/model-repository,readonly",
            "--mount",
            f"type=bind,src={_ASSET_ROOT},dst=/models,readonly",
            _TRITON_IMAGE,
            "tritonserver",
            "--model-repository=/model-repository",
            "--model-control-mode=explicit",
            "--load-model=detector",
            "--load-model=clip_image",
            "--strict-model-config=true",
            "--exit-on-error=true",
        ),
    )


async def wait_triton(endpoint: str) -> bool:
    """Wait until both production inference models report ready."""
    client = InferenceServerClient(url=endpoint)
    try:
        for _attempt in range(120):
            try:
                ready = await client.is_server_ready()
                detector = await client.is_model_ready("detector")
                clip_image = await client.is_model_ready("clip_image")
            except InferenceServerException:
                await anyio.sleep(0.25)
                continue
            if ready and detector and clip_image:
                return True
            await anyio.sleep(0.25)
        return False
    finally:
        await client.close()


async def start_postgres(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Start one disposable pgvector database with host authentication disabled."""
    return await run_command(
        context,
        name="task11-postgres-up",
        command=(
            "docker",
            "run",
            "--detach",
            "--name",
            container,
            "--network",
            "bridge",
            "--env",
            "POSTGRES_HOST_AUTH_METHOD=trust",
            "--env",
            "POSTGRES_DB=gods_watching_task11",
            _POSTGRES_IMAGE,
        ),
    )


async def container_ip(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Read a run-owned bridge address for host-side RTSP or PostgreSQL access."""
    return await run_command(
        context,
        name=f"{container}-ip",
        command=(
            "docker",
            "inspect",
            "--format",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            container,
        ),
    )


async def prepare_database(
    context: ScenarioContextProtocol, *, container: str, database_url: str
) -> CommandResult:
    """Wait for PostgreSQL and migrate the disposable schema to head."""
    readiness_prefix = "for i in {1..120}; do pg_isready -U postgres"
    readiness_command = (
        f"{readiness_prefix} -d gods_watching_task11 && exit 0; sleep .25; done; exit 1"
    )
    ready = await run_command(
        context,
        name="task11-postgres-ready",
        command=(
            "docker",
            "exec",
            container,
            "bash",
            "-c",
            readiness_command,
        ),
    )
    if ready.return_code != 0:
        return ready
    return await run_command(
        context,
        name="task11-migrations",
        command=(
            "/usr/bin/env",
            f"GW_DATABASE_URL={database_url}",
            "uv",
            "run",
            "--project",
            str(_REPOSITORY_ROOT),
            "alembic",
            "upgrade",
            "head",
        ),
    )


async def run_driver(  # noqa: PLR0913
    context: ScenarioContextProtocol,
    *,
    mode: Literal["appearance", "appearance-stale"],
    resources: Task11Resources,
    rtsp_host: str,
    output: Path,
    crop_root: Path,
) -> CommandResult:
    """Run the production adapters against the owned RTSP, Triton, and database stack."""
    return await run_command(
        context,
        name=f"task11-{mode}-driver",
        command=(
            "/usr/bin/env",
            "PYTHONPATH=server/src",
            f"GW_TASK11_MODE={mode}",
            f"GW_TASK11_DATABASE_URL={resources.database_url}",
            f"GW_TASK11_TRITON_URL=127.0.0.1:{context.allocated_port}",
            f"GW_TASK11_RTSP_URL=rtsp://{rtsp_host}:8554/camera-1",
            f"GW_TASK11_OUTPUT={output}",
            f"GW_TASK11_CROP_ROOT={crop_root}",
            "uv",
            "run",
            "--project",
            str(_REPOSITORY_ROOT),
            "python",
            "-m",
            "gods_watching.verification.scenarios.task11_driver",
        ),
    )


async def remove_container(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Force-remove one exact run-owned container."""
    return await run_command(
        context,
        name=f"{container}-cleanup",
        command=("docker", "rm", "--force", container),
    )


async def inspect_containers(
    context: ScenarioContextProtocol, *, containers: tuple[str, ...]
) -> CommandResult:
    """Return any exact-name Task 11 containers that survived cleanup."""
    filters = tuple(item for name in containers for item in ("--filter", f"name=^/{name}$"))
    return await run_command(
        context,
        name="task11-resource-inspect",
        command=("docker", "ps", "-a", *filters, "--format", "{{.Names}}"),
    )


__all__ = [
    "Task11Resources",
    "container_ip",
    "inspect_containers",
    "prepare_database",
    "remove_container",
    "run_driver",
    "start_postgres",
    "start_triton",
    "wait_triton",
]
