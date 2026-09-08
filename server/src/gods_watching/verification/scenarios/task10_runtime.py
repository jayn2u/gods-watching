"""Own the disposable fixture and Triton resources for Task 10 scenarios."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anyio
from anyio import EndOfStream
from anyio.abc import ByteReceiveStream

from gods_watching.verification.models import ScenarioContextProtocol

from .task07_runtime import wait_for_endpoint

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_COMPOSE_FILE: Final = _REPOSITORY_ROOT / "deploy/compose.fixtures.yaml"
_MODEL_REPOSITORY: Final = _REPOSITORY_ROOT / "inference/models/detector"
_MODEL_ASSETS: Final = _REPOSITORY_ROOT / "runtime/assets/models"
_TRITON_IMAGE: Final = "gods-watching-triton:25.02"
_FIXTURE_STREAM_COUNT: Final = 4
_FIXTURE_READY_TIMEOUT_SECONDS: Final = 20.0
_FIXTURE_PROBE_TIMEOUT_SECONDS: Final = 2.0
_FIXTURE_RETRY_DELAY_SECONDS: Final = 0.25
_FIXTURE_MAX_RETRY_DELAY_SECONDS: Final = 2.0


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Capture one resource-ledger-owned subprocess result."""

    return_code: int
    stdout: str
    stderr: str


async def _drain(stream: ByteReceiveStream | None, chunks: list[bytes]) -> None:
    if stream is None:
        return
    while True:
        try:
            chunks.append(await stream.receive())
        except EndOfStream:
            return


async def run_command(
    context: ScenarioContextProtocol, *, name: str, command: Sequence[str]
) -> CommandResult:
    """Run one bounded command and drain both output streams."""
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


def compose_command(project: str, *arguments: str) -> tuple[str, ...]:
    """Build the exact Compose command for one run-owned project."""
    return (
        "docker",
        "compose",
        "-p",
        project,
        "-f",
        str(_COMPOSE_FILE),
        *arguments,
    )


async def start_fixtures(context: ScenarioContextProtocol, project: str) -> CommandResult:
    """Start the four finite H.264 fixture publishers and MediaMTX."""
    return await run_command(
        context,
        name="task10-fixtures-up",
        command=compose_command(project, "--profile", "fixtures", "up", "-d"),
    )


async def stop_fixtures(context: ScenarioContextProtocol, project: str) -> CommandResult:
    """Stop and remove only the run-owned fixture Compose project."""
    return await run_command(
        context,
        name="task10-fixtures-down",
        command=compose_command(
            project, "--profile", "fixtures", "down", "--remove-orphans", "--timeout", "10"
        ),
    )


async def start_triton(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Start the pinned GPU Triton detector with read-only model mounts."""
    repository_mount = f"type=bind,src={_MODEL_REPOSITORY},dst=/model-repository/detector,readonly"
    assets_mount = f"type=bind,src={_MODEL_ASSETS},dst=/models,readonly"
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
        assets_mount,
        _TRITON_IMAGE,
        "tritonserver",
        "--model-repository=/model-repository",
        "--strict-model-config=true",
        "--exit-on-error=false",
    )
    return await run_command(context, name="task10-triton-up", command=command)


async def wait_triton(context: ScenarioContextProtocol) -> bool:
    """Wait until the run-owned Triton server and detector model are ready."""
    return await wait_for_endpoint(f"127.0.0.1:{context.allocated_port}", model_ready=True)


async def remove_triton(context: ScenarioContextProtocol, *, container: str) -> CommandResult:
    """Force-remove the uniquely named run-owned Triton container."""
    return await run_command(
        context, name="task10-triton-down", command=("docker", "rm", "--force", container)
    )


async def fixture_ip(context: ScenarioContextProtocol, *, project: str) -> CommandResult:
    """Read the MediaMTX container IP used by the host-side PyAV driver."""
    return await run_command(
        context,
        name="task10-fixtures-ip",
        command=(
            "docker",
            "inspect",
            "--format",
            "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
            f"{project}-fixture-mediamtx-1",
        ),
    )


async def wait_fixture_streams(context: ScenarioContextProtocol, *, rtsp_host: str) -> bool:
    """Wait until all finite publishers expose decodable H.264 RTSP streams."""
    pending = set(range(1, _FIXTURE_STREAM_COUNT + 1))
    deadline = anyio.current_time() + _FIXTURE_READY_TIMEOUT_SECONDS
    retry_delay = _FIXTURE_RETRY_DELAY_SECONDS
    while pending and anyio.current_time() < deadline:
        for camera_index in tuple(sorted(pending)):
            result = await _probe_fixture_stream(
                context, camera_index=camera_index, rtsp_host=rtsp_host
            )
            if result is not None and _is_h264_probe(result):
                pending.remove(camera_index)
            if not pending:
                return True
            if anyio.current_time() >= deadline:
                break
        if pending:
            remaining = deadline - anyio.current_time()
            await anyio.sleep(min(retry_delay, max(0.0, remaining)))
            retry_delay = min(
                retry_delay * 2.0,
                _FIXTURE_MAX_RETRY_DELAY_SECONDS,
            )
    return not pending


async def _probe_fixture_stream(
    context: ScenarioContextProtocol, *, camera_index: int, rtsp_host: str
) -> CommandResult | None:
    try:
        with anyio.fail_after(_FIXTURE_PROBE_TIMEOUT_SECONDS):
            return await run_command(
                context,
                name=f"task10-fixture-probe-{camera_index}",
                command=(
                    "ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-rw_timeout",
                    "1000000",
                    "-select_streams",
                    "v:0",
                    "-read_intervals",
                    "%+0.1",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "csv=p=0",
                    f"rtsp://{rtsp_host}:8554/camera-{camera_index}",
                ),
            )
    except TimeoutError:
        return None


def _is_h264_probe(result: CommandResult) -> bool:
    return result.return_code == 0 and result.stdout.strip().casefold() == "h264"


async def compose_logs(context: ScenarioContextProtocol, *, project: str) -> CommandResult:
    """Capture fixture gateway logs before Compose cleanup."""
    return await run_command(
        context,
        name="task10-fixtures-logs",
        command=compose_command(project, "logs", "--no-color", "fixture-mediamtx"),
    )


async def inspect_resources(
    context: ScenarioContextProtocol, *, project: str, triton: str
) -> CommandResult:
    """List this run's Compose resources and exact-name Triton resource."""
    common = (
        "docker",
        "ps",
        "-a",
        "--format",
        "{{.Names}} {{.Status}} {{.Image}}",
    )
    label_result, name_result = await _inspect_queries(
        context,
        label_command=(
            *common[:3],
            "--filter",
            f"label=com.docker.compose.project={project}",
            *common[3:],
        ),
        name_command=(*common[:3], "--filter", f"name=^/{triton}$", *common[3:]),
    )
    lines = list(
        dict.fromkeys((*label_result.stdout.splitlines(), *name_result.stdout.splitlines()))
    )
    return CommandResult(
        return_code=(
            label_result.return_code if label_result.return_code != 0 else name_result.return_code
        ),
        stdout="\n".join(lines) + ("\n" if lines else ""),
        stderr="\n".join(
            message for message in (label_result.stderr, name_result.stderr) if message
        ),
    )


async def _inspect_queries(
    context: ScenarioContextProtocol,
    *,
    label_command: Sequence[str],
    name_command: Sequence[str],
) -> tuple[CommandResult, CommandResult]:
    label_result = await run_command(
        context, name="task10-resource-inspect-label", command=label_command
    )
    name_result = await run_command(
        context, name="task10-resource-inspect-name", command=name_command
    )
    return label_result, name_result


__all__ = [
    "CommandResult",
    "compose_command",
    "compose_logs",
    "fixture_ip",
    "inspect_resources",
    "remove_triton",
    "run_command",
    "start_fixtures",
    "start_triton",
    "stop_fixtures",
    "wait_fixture_streams",
    "wait_triton",
]
