"""Own the isolated Triton process used by Task 7 verification."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anyio
from anyio import EndOfStream
from anyio.abc import ByteReceiveStream
from tritonclient.grpc.aio import InferenceServerClient
from tritonclient.utils import InferenceServerException

from gods_watching.verification.models import ScenarioContextProtocol

_IMAGE: Final = "gods-watching-triton:25.02"


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Capture a command's binary result without trusting its prose."""

    return_code: int
    stdout: str
    stderr: str


async def _drain(stream: ByteReceiveStream, chunks: list[bytes]) -> None:
    while True:
        try:
            chunks.append(await stream.receive())
        except EndOfStream:
            return


async def run_command(
    context: ScenarioContextProtocol, *, name: str, command: Sequence[str]
) -> CommandResult:
    """Run and fully drain one resource-ledger-owned subprocess."""
    stdout_chunks: list[bytes] = []
    stderr_chunks: list[bytes] = []
    return_code = -1
    async with (
        context.process(name=name, command=command) as process,
        anyio.create_task_group() as task_group,
    ):
        if process.stdout is not None:
            task_group.start_soon(_drain, process.stdout, stdout_chunks)
        if process.stderr is not None:
            task_group.start_soon(_drain, process.stderr, stderr_chunks)
        return_code = await process.wait()
    return CommandResult(
        return_code=return_code,
        stdout=b"".join(stdout_chunks).decode(errors="replace"),
        stderr=b"".join(stderr_chunks).decode(errors="replace"),
    )


def resource_path(context: ScenarioContextProtocol) -> Path:
    """Return the Task 7 container lifecycle artifact path."""
    return context.run_root / "task-7-resources.json"


def write_resource(context: ScenarioContextProtocol, *, container: str, state: str) -> None:
    """Persist the owned container state before and after acquisition."""
    _ = resource_path(context).write_text(
        json.dumps({"container": container, "state": state}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def start_detector(
    context: ScenarioContextProtocol,
    *,
    container: str,
    repository_root: Path,
    models_root: Path,
) -> CommandResult:
    """Start only the detector model in an isolated named container."""
    write_resource(context, container=container, state="registered")
    repository_mount = "".join(
        (
            f"type=bind,src={repository_root / 'inference/models/detector'},",
            "dst=/model-repository/detector,readonly",
        )
    )
    command = (
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
        repository_mount,
        "--mount",
        f"type=bind,src={models_root},dst=/models,readonly",
        _IMAGE,
        "tritonserver",
        "--model-repository=/model-repository",
        "--strict-model-config=true",
        "--exit-on-error=false",
    )
    result = await run_command(context, name=f"{container}-start", command=command)
    if result.return_code == 0:
        write_resource(context, container=container, state="started")
    return result


async def wait_for_endpoint(endpoint: str, *, model_ready: bool) -> bool:
    """Wait for Triton connectivity and the expected model readiness state."""
    client = InferenceServerClient(url=endpoint)
    try:
        for _attempt in range(120):
            try:
                server = await client.is_server_ready()
                model = await client.is_model_ready("detector")
            except InferenceServerException:
                await anyio.sleep(0.25)
                continue
            if server is model_ready and model is model_ready:
                return True
            if not model_ready and not server and not model:
                return True
            await anyio.sleep(0.25)
        return False
    finally:
        await client.close()


async def capture_logs(
    context: ScenarioContextProtocol, *, container: str, destination: Path
) -> CommandResult:
    """Capture current owned-container logs to an evidence artifact."""
    result = await run_command(
        context, name=f"{container}-logs", command=("docker", "logs", container)
    )
    _ = destination.write_text(result.stdout + result.stderr, encoding="utf-8")
    return result


async def remove_detector(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Force-remove only the scenario's uniquely named detector container."""
    result = await run_command(
        context, name=f"{container}-cleanup", command=("docker", "rm", "-f", container)
    )
    write_resource(context, container=container, state="cleaned")
    return result
