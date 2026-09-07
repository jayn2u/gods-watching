"""Launch the private media stack and real Chromium verification."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Literal
from uuid import uuid4

import anyio
from pydantic import SecretStr

from gods_watching.verification.models import ScenarioContextProtocol

from .task09_artifacts import PublisherArtifact, StackPaths
from .task09_probes import inspect_writes, probe_denied
from .task09_qa import MediaQaSettings
from .task09_runtime import (
    READER_USER,
    GatewayPorts,
    RuntimeSecrets,
    finalize_publisher,
    gateway_command,
    prepare_runtime,
    wait_for_gateway,
)


class BrowserVerificationError(RuntimeError):
    """Report a failed browser process without exposing its settings."""

    def __init__(self, mode: str) -> None:
        """Create a credential-free failure message."""
        super().__init__(f"Chromium {mode} verification failed")


async def run_stack(
    context: ScenarioContextProtocol,
    *,
    browser_mode: Literal["decode", "denied"],
    probe_rtsp_denial: bool,
) -> StackPaths:
    """Run one cleanup-owned publisher, gateway, proxy and browser stack."""
    media_path = f"test-publisher/{uuid4()!s}"
    runtime_secrets = prepare_runtime(context, media_path)
    browser_path = context.run_root / "task-9-webrtc-stats.json"
    writes_path = context.run_root / "task-9-writes.json"
    denied_path = context.run_root / "task-9-denied.json"
    publisher_path = context.run_root / "task-9-publisher-generations.json"
    async with _media_gateway(context):
        ports = await wait_for_gateway(context)
        await _wait_for_tcp(ports.rtsp)
        publishers = finalize_publisher(context, ports.rtsp)
        async with context.process(
            name="media-test-publisher-generation-1",
            command=("/bin/sh", str(publishers.first)),
        ) as first_publisher:
            await _wait_for_authenticated_path(ports, media_path, runtime_secrets, ready=True)
            first_returncode = await first_publisher.wait()
        await _wait_for_authenticated_path(ports, media_path, runtime_secrets, ready=False)
        async with context.process(
            name="media-test-publisher-generation-2",
            command=("/bin/sh", str(publishers.second)),
        ):
            await _wait_for_authenticated_path(ports, media_path, runtime_secrets, ready=True)
            publisher = PublisherArtifact(
                first_returncode=first_returncode,
                path_not_ready_between_generations=True,
                second_generation_ready=True,
                seamless_stream_loop_used=False,
            )
            _ = publisher_path.write_text(publisher.model_dump_json(indent=2) + "\n")
            settings_path = _write_qa_settings(context, ports, media_path, runtime_secrets)
            async with context.process(
                name="media-private-proxy",
                command=_proxy_command(context, settings_path),
            ):
                await _wait_for_tcp(context.allocated_port)
                try:
                    await _run_browser(context, settings_path, browser_path, browser_mode)
                except BrowserVerificationError:
                    await _capture_gateway_logs(context)
                    raise
                if probe_rtsp_denial:
                    denied = await probe_denied(
                        context, ports, media_path, browser_path, runtime_secrets
                    )
                    _ = denied_path.write_text(denied.model_dump_json(indent=2) + "\n")
            writes = await inspect_writes(context)
            _ = writes_path.write_text(writes.model_dump_json(indent=2) + "\n")
    return StackPaths(
        browser=browser_path,
        writes=writes_path,
        denied=denied_path,
        publisher=publisher_path,
    )


@asynccontextmanager
async def _media_gateway(context: ScenarioContextProtocol) -> AsyncIterator[None]:
    """Stop the exact owned container even when the attached Docker client exits first."""
    container_name = f"{context.compose_project}-media"
    try:
        async with context.process(name="media-gateway", command=gateway_command(context)):
            yield
    finally:
        with anyio.CancelScope(shield=True):
            _ = await anyio.run_process(
                (
                    "/usr/bin/docker",
                    "container",
                    "stop",
                    "--timeout",
                    "2",
                    container_name,
                ),
                check=False,
            )
            _ = await anyio.run_process(
                ("/usr/bin/docker", "container", "remove", container_name),
                check=False,
            )


async def _capture_gateway_logs(context: ScenarioContextProtocol) -> None:
    completed = await anyio.run_process(
        ("/usr/bin/docker", "logs", f"{context.compose_project}-media"),
        check=False,
    )
    _ = (context.run_root / "task-9-gateway-error.log").write_bytes(
        completed.stdout + completed.stderr
    )


def _write_qa_settings(
    context: ScenarioContextProtocol,
    ports: GatewayPorts,
    media_path: str,
    runtime_secrets: RuntimeSecrets,
) -> Path:
    settings = MediaQaSettings(
        gateway_host="127.0.0.1",
        gateway_port=ports.whep,
        gateway_username=READER_USER,
        gateway_password=SecretStr(runtime_secrets.reader_password),
        camera_id=uuid4(),
        media_path=media_path,
        qa_session=SecretStr(runtime_secrets.qa_session),
        origin=f"http://127.0.0.1:{context.allocated_port}",
    )
    path = context.runtime_root / "media-qa-settings.json"
    payload = settings.model_dump(mode="json")
    payload["gateway_password"] = runtime_secrets.reader_password
    payload["qa_session"] = runtime_secrets.qa_session
    _ = path.write_text(json.dumps(payload), encoding="utf-8")
    path.chmod(0o600)
    return path


def _proxy_command(context: ScenarioContextProtocol, settings_path: Path) -> tuple[str, ...]:
    repository_root = Path(__file__).parents[5]
    return (
        "/usr/bin/env",
        f"GW_MEDIA_QA_SETTINGS_PATH={settings_path}",
        "uv",
        "run",
        "--project",
        str(repository_root),
        "uvicorn",
        "gods_watching.verification.scenarios.task09_qa:create_qa_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(context.allocated_port),
        "--log-level",
        "warning",
    )


async def _run_browser(
    context: ScenarioContextProtocol,
    settings_path: Path,
    output_path: Path,
    mode: Literal["decode", "denied"],
) -> None:
    repository_root = Path(__file__).parents[5]
    command = (
        "/usr/bin/env",
        f"PLAYWRIGHT_BROWSERS_PATH={repository_root / 'runtime/playwright-browsers'}",
        "/usr/bin/node",
        str(repository_root / "qa/media/webrtc_decode.mjs"),
        str(settings_path),
        str(output_path),
        mode,
    )
    async with context.process(name=f"media-chromium-{mode}", command=command) as process:
        returncode = await process.wait()
        stderr = b""
        if process.stderr is not None:
            with suppress(anyio.EndOfStream):
                stderr = await process.stderr.receive()
    if returncode != 0:
        _ = output_path.with_name("task-9-browser-error.log").write_bytes(stderr)
        raise BrowserVerificationError(mode=mode)


async def _wait_for_tcp(port: int) -> None:
    with anyio.fail_after(10):
        while True:
            try:
                stream = await anyio.connect_tcp("127.0.0.1", port)
            except OSError:
                await anyio.sleep(0.05)
                continue
            await stream.aclose()
            return


async def _wait_for_authenticated_path(
    ports: GatewayPorts,
    media_path: str,
    runtime_secrets: RuntimeSecrets,
    *,
    ready: bool,
) -> None:
    source = (
        f"rtsp://{READER_USER}:{runtime_secrets.reader_password}"
        f"@127.0.0.1:{ports.rtsp}/{media_path}"
    )
    with anyio.fail_after(10):
        while True:
            completed = await anyio.run_process(
                (
                    "/usr/bin/ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-show_entries",
                    "stream=codec_name",
                    source,
                ),
                check=False,
            )
            if (completed.returncode == 0) is ready:
                return
            await anyio.sleep(0.05)
