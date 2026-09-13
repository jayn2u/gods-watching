"""Run the healthy detector stack and capture its typed evidence."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

import anyio

from gods_watching.verification.models import ScenarioContextProtocol

from .task07_models import HealthyRun, ModelConfig, ModelStats, ProbeEvidence
from .task07_runtime import (
    capture_logs,
    remove_detector,
    run_command,
    start_detector,
    wait_for_endpoint,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
ASSETS_ROOT: Final = REPOSITORY_ROOT / "runtime/assets/models"
IMAGE_ID: Final = "sha256:5e6e358bed19b978bbf0405062bc01b8f702113995b6cc2ec694bd8313ac3ff3"
WEIGHTS_SHA256: Final = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
_FIXTURE_SOURCE: Final = REPOSITORY_ROOT / "runtime/assets/fixtures/crosswalk.mp4"


@dataclass(frozen=True, slots=True)
class DetectorProbeError(RuntimeError):
    """Report a failed fixture extraction or gRPC probe command."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


async def healthy_run(
    context: ScenarioContextProtocol, *, suffix: str, probe_name: str
) -> HealthyRun:
    """Start, drive, capture, and clean one healthy detector container."""
    container = f"{context.compose_project}-detector-{suffix}"
    endpoint = f"127.0.0.1:{context.allocated_port}"
    frame = context.run_root / f"{suffix}-pedestrian-frame.jpg"
    empty = context.run_root / f"{suffix}-empty-frame.jpg"
    annotated = context.run_root / f"{suffix}-annotated-pedestrians.jpg"
    probe_path = context.run_root / probe_name
    config_path = context.run_root / f"{suffix}-model-config.json"
    stats_path = context.run_root / f"{suffix}-model-stats.json"
    log_path = context.run_root / f"{suffix}-container.log"
    started = await start_detector(
        context,
        container=container,
        repository_root=REPOSITORY_ROOT,
        models_root=ASSETS_ROOT,
    )
    try:
        ready = started.return_code == 0 and await wait_for_endpoint(endpoint, model_ready=True)
        frame_result = await run_command(
            context,
            name=f"{suffix}-extract-pedestrian-frame",
            command=(
                "/usr/bin/ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                "2",
                "-i",
                str(_FIXTURE_SOURCE),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(frame),
            ),
        )
        empty_result = await run_command(
            context,
            name=f"{suffix}-create-empty-frame",
            command=(
                "/usr/bin/ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=1280x720",
                "-frames:v",
                "1",
                "-q:v",
                "2",
                "-y",
                str(empty),
            ),
        )
        probe_result = await run_command(
            context,
            name=f"{suffix}-grpc-probe",
            command=(
                sys.executable,
                str(REPOSITORY_ROOT / "qa/detector/probe.py"),
                endpoint,
                str(frame),
                str(empty),
                str(probe_path),
                str(annotated),
            ),
        )
        if frame_result.return_code or empty_result.return_code or probe_result.return_code:
            raise DetectorProbeError(detail="detector fixture or gRPC probe failed")
        config_result = await run_command(
            context,
            name=f"{suffix}-effective-config",
            command=(
                "docker",
                "exec",
                container,
                "curl",
                "-fsS",
                "http://127.0.0.1:8000/v2/models/detector/config",
            ),
        )
        stats_result = await run_command(
            context,
            name=f"{suffix}-batch-stats",
            command=(
                "docker",
                "exec",
                container,
                "curl",
                "-fsS",
                "http://127.0.0.1:8000/v2/models/detector/stats",
            ),
        )
        image_result = await run_command(
            context,
            name=f"{suffix}-image-id",
            command=(
                "docker",
                "image",
                "inspect",
                "gods-watching-triton:25.02",
                "--format",
                "{{.Id}}",
            ),
        )
        _ = config_path.write_text(config_result.stdout, encoding="utf-8")
        _ = stats_path.write_text(stats_result.stdout, encoding="utf-8")
        _ = await capture_logs(context, container=container, destination=log_path)
        return HealthyRun(
            ready=ready,
            probe=ProbeEvidence.model_validate_json(probe_path.read_text()),
            config=ModelConfig.model_validate_json(config_result.stdout),
            stats=ModelStats.model_validate_json(stats_result.stdout),
            image_id=image_result.stdout.strip(),
            paths=(probe_path, annotated, config_path, stats_path, log_path),
        )
    finally:
        with anyio.CancelScope(shield=True):
            _ = await remove_detector(context, container=container)
