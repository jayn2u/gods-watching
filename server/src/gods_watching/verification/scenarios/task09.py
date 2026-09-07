"""Register real recording-free media transport verification scenarios."""

from http import HTTPStatus

from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task09_artifacts import (
    DecodeArtifact,
    DeniedArtifact,
    PublisherArtifact,
    WriteArtifact,
)
from .task09_stack import run_stack

_MIN_RISING_SAMPLES = 2


async def run_media(context: ScenarioContextProtocol) -> ScenarioReport:
    """Decode real H.264 in Chromium through the private WHEP proxy."""
    paths = await run_stack(context, browser_mode="decode", probe_rtsp_denial=False)
    decode = DecodeArtifact.model_validate_json(paths.browser.read_text(encoding="utf-8"))
    writes = WriteArtifact.model_validate_json(paths.writes.read_text(encoding="utf-8"))
    publisher = PublisherArtifact.model_validate_json(paths.publisher.read_text(encoding="utf-8"))
    samples = decode.observation.samples
    first_frames = samples[0].frames_decoded if samples else 0
    last_frames = samples[-1].frames_decoded if samples else 0
    return ScenarioReport(
        checks=(
            Check(
                name="chromium-frames-decoded-rise",
                passed=len(samples) >= _MIN_RISING_SAMPLES and last_frames > first_frames,
                detail=f"{first_frames}->{last_frames}",
            ),
            Check(
                name="chromium-negotiated-h264",
                passed=decode.observation.codec_mime_type.casefold() == "video/h264",
                detail=decode.observation.codec_mime_type,
            ),
            Check(
                name="receive-only-whep-teardown",
                passed=(
                    decode.observation.transceiver_direction == "recvonly"
                    and HTTPStatus.OK
                    <= decode.observation.delete_status
                    < HTTPStatus.MULTIPLE_CHOICES
                    and "/api/live/" in decode.observation.resource_location
                ),
                detail="browser DELETE closed application-owned resource",
            ),
            Check(
                name="recording-free-filesystem",
                passed=(
                    writes.gateway_read_only
                    and not writes.recording_mount_present
                    and not writes.docker_diff
                ),
                detail="read-only gateway, config-only mount, empty writable diff",
            ),
            Check(
                name="finite-publisher-reopen-boundary",
                passed=(
                    publisher.first_returncode == 0
                    and publisher.path_not_ready_between_generations
                    and publisher.second_generation_ready
                    and not publisher.seamless_stream_loop_used
                ),
                detail="finite exit, path not-ready, second generation ready",
            ),
        ),
        artifact_paths=(
            paths.browser,
            paths.browser.with_suffix(".png"),
            paths.writes,
            paths.publisher,
        ),
        evidence_kind=EvidenceKind.REAL,
    )


async def run_media_denied(context: ScenarioContextProtocol) -> ScenarioReport:
    """Prove missing app context and uncredentialed RTSP operations fail closed."""
    paths = await run_stack(context, browser_mode="denied", probe_rtsp_denial=True)
    denied = DeniedArtifact.model_validate_json(paths.denied.read_text(encoding="utf-8"))
    publisher = PublisherArtifact.model_validate_json(paths.publisher.read_text(encoding="utf-8"))
    return ScenarioReport(
        checks=(
            Check(
                name="absent-session-context-denied",
                passed=denied.browser_status == HTTPStatus.UNAUTHORIZED,
                detail=f"HTTP {denied.browser_status}",
            ),
            Check(
                name="uncredentialed-private-rtsp-denied",
                passed=denied.rtsp_read_returncode != 0 and denied.rtsp_publish_returncode != 0,
                detail="read and publish returned nonzero",
            ),
            Check(
                name="credentials-not-persisted",
                passed=not denied.credential_text_persisted,
                detail="persistent evidence contains no ephemeral credential",
            ),
            Check(
                name="finite-publisher-reopen-boundary",
                passed=(
                    publisher.first_returncode == 0
                    and publisher.path_not_ready_between_generations
                    and publisher.second_generation_ready
                    and not publisher.seamless_stream_loop_used
                ),
                detail="finite exit, path not-ready, second generation ready",
            ),
        ),
        artifact_paths=(paths.browser, paths.denied, paths.publisher),
        evidence_kind=EvidenceKind.REAL,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("media"),
        runner=run_media,
        required_commands=("docker", "ffmpeg", "ffprobe", "node", "uv"),
        timeout_seconds=45.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("media-denied"),
        runner=run_media_denied,
        required_commands=("docker", "ffmpeg", "ffprobe", "node", "uv"),
        timeout_seconds=45.0,
    )
)
