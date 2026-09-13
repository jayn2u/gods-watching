"""Security and filesystem probes for task-9 runtime verification."""

from pathlib import Path

import anyio

from gods_watching.verification.models import ScenarioContextProtocol

from .task09_artifacts import DeniedArtifact, DeniedBrowserArtifact, WriteArtifact
from .task09_runtime import GatewayPorts, RuntimeSecrets


async def probe_denied(
    context: ScenarioContextProtocol,
    ports: GatewayPorts,
    media_path: str,
    browser_path: Path,
    runtime_secrets: RuntimeSecrets,
) -> DeniedArtifact:
    """Attempt browser signaling and RTSP operations without credentials."""
    repository_root = Path(__file__).parents[5]
    target = f"rtsp://127.0.0.1:{ports.rtsp}/{media_path}"
    read = await anyio.run_process(
        ("/usr/bin/ffprobe", "-v", "error", "-rtsp_transport", "tcp", target),
        check=False,
    )
    publish = await anyio.run_process(
        (
            "/usr/bin/ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            str(repository_root / "runtime/assets/fixtures/crosswalk.mp4"),
            "-t",
            "0.2",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            target,
        ),
        check=False,
    )
    browser = DeniedBrowserArtifact.model_validate_json(browser_path.read_text(encoding="utf-8"))
    persisted = "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in context.run_root.iterdir()
        if path.is_file()
    )
    return DeniedArtifact(
        browser_status=browser.observation.status_code,
        rtsp_read_returncode=read.returncode,
        rtsp_publish_returncode=publish.returncode,
        credential_text_persisted=any(
            secret_value in persisted
            for secret_value in (
                runtime_secrets.reader_password,
                runtime_secrets.publisher_password,
                runtime_secrets.qa_session,
            )
        ),
    )


async def inspect_writes(context: ScenarioContextProtocol) -> WriteArtifact:
    """Inspect mounts and writable-layer changes while the gateway is live."""
    container_name = f"{context.compose_project}-media"
    mounts = await anyio.run_process(
        (
            "/usr/bin/docker",
            "inspect",
            "--format",
            "{{range .Mounts}}{{.Destination}}|rw={{.RW}}{{println}}{{end}}",
            container_name,
        )
    )
    rootfs = await anyio.run_process(
        (
            "/usr/bin/docker",
            "inspect",
            "--format",
            "{{.HostConfig.ReadonlyRootfs}}",
            container_name,
        )
    )
    changes = await anyio.run_process(("/usr/bin/docker", "diff", container_name))
    mount_lines = tuple(line for line in mounts.stdout.decode().splitlines() if line)
    diff_lines = tuple(line for line in changes.stdout.decode().splitlines() if line)
    return WriteArtifact(
        gateway_read_only=rootfs.stdout.decode().strip() == "true",
        mounts=mount_lines,
        docker_diff=diff_lines,
        recording_mount_present=any("record" in line.lower() for line in mount_lines),
    )
