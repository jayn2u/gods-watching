"""Runtime helpers for isolated task-9 MediaMTX verification."""

import base64
import hashlib
import secrets
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anyio

from gods_watching.verification.models import ScenarioContextProtocol

MEDIAMTX_IMAGE: Final = (
    "bluenviron/mediamtx@sha256:206139c58377b7544d6ef63f8af86bcf46b9e89262aaa6f7513e1347dbab7d36"
)
READER_USER: Final = "task9-media-reader"
PUBLISHER_USER: Final = "task9-test-publisher"


@dataclass(frozen=True, slots=True)
class RuntimeSecrets:
    """Hold verification-only credentials until the runtime root is removed."""

    reader_password: str
    publisher_password: str
    qa_session: str


@dataclass(frozen=True, slots=True)
class GatewayPorts:
    """Expose loopback-only RTSP and WHEP proxy targets."""

    rtsp: int
    whep: int


@dataclass(frozen=True, slots=True)
class PublisherScripts:
    """Locate the short first generation and full second generation scripts."""

    first: Path
    second: Path


def prepare_runtime(context: ScenarioContextProtocol, media_path: str) -> RuntimeSecrets:
    """Render ephemeral credentials without writing them to persistent evidence."""
    secrets_value = RuntimeSecrets(
        reader_password=secrets.token_urlsafe(32),
        publisher_password=secrets.token_urlsafe(32),
        qa_session=secrets.token_urlsafe(32),
    )
    repository_root = Path(__file__).parents[5]
    template = (repository_root / "deploy/mediamtx.yml").read_text(encoding="utf-8")
    rendered = (
        template.replace("__GW_MEDIA_READER_USER__", READER_USER)
        .replace(
            "__GW_MEDIA_READER_PASS_SHA256__",
            _credential_hash(secrets_value.reader_password),
        )
        .replace("__GW_TEST_PUBLISHER_USER__", PUBLISHER_USER)
        .replace(
            "__GW_TEST_PUBLISHER_PASS_SHA256__",
            _credential_hash(secrets_value.publisher_password),
        )
        .replace("__GW_MEDIA_CONTROL_USER__", "task9-media-control")
        .replace(
            "__GW_MEDIA_CONTROL_PASS_SHA256__",
            _credential_hash(secrets.token_urlsafe(32)),
        )
    )
    config_path = context.runtime_root / "mediamtx.yml"
    _ = config_path.write_text(rendered, encoding="utf-8")
    config_path.chmod(0o644)
    source = repository_root / "runtime/assets/fixtures/crosswalk.mp4"
    destination = (
        f"rtsp://{PUBLISHER_USER}:{secrets_value.publisher_password}"
        f"@127.0.0.1:__RTSP_PORT__/{media_path}"
    )
    command = (
        "/usr/bin/ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-re",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-profile:v",
        "baseline",
        "-level:v",
        "3.1",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        destination,
    )
    _write_publisher_script(
        context.runtime_root / "publisher-generation-1.sh",
        (*command[:-5], "-t", "1", *command[-5:]),
    )
    _write_publisher_script(context.runtime_root / "publisher-generation-2.sh", command)
    return secrets_value


def _write_publisher_script(path: Path, command: tuple[str, ...]) -> None:
    _ = path.write_text(
        "#!/bin/sh\nexec " + " ".join(shlex.quote(part) for part in command) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o700)


def gateway_command(context: ScenarioContextProtocol) -> tuple[str, ...]:
    """Build an isolated read-only gateway command with one loopback UDP port."""
    container_name = f"{context.compose_project}-media"
    return (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "--read-only",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "-e",
        f"MTX_WEBRTCLOCALUDPADDRESS=:{context.allocated_port}",
        "-e",
        "MTX_WEBRTCADDITIONALHOSTS=127.0.0.1",
        "-p",
        "127.0.0.1::8554",
        "-p",
        "127.0.0.1::8889",
        "-p",
        f"127.0.0.1:{context.allocated_port}:{context.allocated_port}/udp",
        "--volume",
        f"{context.runtime_root / 'mediamtx.yml'}:/mediamtx.yml:ro",
        MEDIAMTX_IMAGE,
    )


async def wait_for_gateway(context: ScenarioContextProtocol) -> GatewayPorts:
    """Read Docker-assigned loopback ports only after the named container is live."""
    container_name = f"{context.compose_project}-media"
    with anyio.fail_after(10):
        while True:
            rtsp = await _mapped_port(container_name, "8554/tcp")
            whep = await _mapped_port(container_name, "8889/tcp")
            if rtsp is not None and whep is not None:
                return GatewayPorts(rtsp=rtsp, whep=whep)
            await anyio.sleep(0.05)


async def _mapped_port(container_name: str, container_port: str) -> int | None:
    completed = await anyio.run_process(
        ("/usr/bin/docker", "port", container_name, container_port),
        check=False,
    )
    if completed.returncode != 0:
        return None
    value = completed.stdout.decode().strip().rsplit(":", 1)[-1]
    try:
        return int(value)
    except ValueError:
        return None


def finalize_publisher(context: ScenarioContextProtocol, rtsp_port: int) -> PublisherScripts:
    """Insert the runtime-assigned RTSP port into both cleanup-owned scripts."""
    paths = PublisherScripts(
        first=context.runtime_root / "publisher-generation-1.sh",
        second=context.runtime_root / "publisher-generation-2.sh",
    )
    for script in (paths.first, paths.second):
        value = script.read_text(encoding="utf-8").replace("__RTSP_PORT__", str(rtsp_port))
        _ = script.write_text(value, encoding="utf-8")
        script.chmod(0o700)
    return paths


def _credential_hash(password: str) -> str:
    return base64.b64encode(hashlib.sha256(password.encode()).digest()).decode("ascii")
