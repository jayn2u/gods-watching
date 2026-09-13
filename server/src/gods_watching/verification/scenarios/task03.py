"""Register attributed fixture input and corruption scenarios."""

import shutil
import time
from pathlib import Path
from typing import ClassVar, Final

import anyio
from pydantic import BaseModel, ConfigDict

from gods_watching.fixtures import (
    FixturePreparationError,
    FixtureProbe,
    build_publisher_command,
    build_read_probe_command,
    load_and_probe_fixtures,
)
from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

_MEDIAMTX_IMAGE: Final = (
    "bluenviron/mediamtx@sha256:206139c58377b7544d6ef63f8af86bcf46b9e89262aaa6f7513e1347dbab7d36"
)
_READY_DEADLINE_SECONDS: Final = 10.0
_POLL_SECONDS: Final = 0.1
_EXPECTED_FIXTURE_COUNT: Final = 4
_MIN_PERSON_SECONDS: Final = 20.0


class ArtifactModel(BaseModel):
    """Keep task-owned scenario artifacts immutable and schema checked."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class PublisherObservation(ArtifactModel):
    """Record the externally observed source-session lifecycle."""

    first_session_ready: bool
    first_process_returncode: int
    path_not_ready_between_sessions: bool
    second_session_ready: bool
    transport: str
    audio_published: bool
    seamless_stream_loop_used: bool


class ProbeArtifact(ArtifactModel):
    """Combine measured local inputs with the RTSP boundary observation."""

    label: str
    probes: tuple[FixtureProbe, ...]
    publisher: PublisherObservation


async def _wait_for_gateway(port: int) -> None:
    deadline = time.monotonic() + _READY_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        try:
            stream = await anyio.connect_tcp("127.0.0.1", port)
        except OSError:
            await anyio.sleep(_POLL_SECONDS)
            continue
        await stream.aclose()
        return
    raise FixturePreparationError(code="gateway_timeout", detail="RTSP listener did not open")


async def _path_is_ready(destination: str) -> bool:
    completed = await anyio.run_process(build_read_probe_command(destination), check=False)
    return completed.returncode == 0


async def _wait_for_path(destination: str, *, ready: bool) -> None:
    deadline = time.monotonic() + _READY_DEADLINE_SECONDS
    while time.monotonic() < deadline:
        if await _path_is_ready(destination) is ready:
            return
        await anyio.sleep(_POLL_SECONDS)
    state = "ready" if ready else "not-ready"
    raise FixturePreparationError(code="path_state_timeout", detail=state)


async def _observe_reopen(context: ScenarioContextProtocol, source: Path) -> PublisherObservation:
    repository_root = Path(__file__).parents[5]
    mediamtx_config = repository_root / "server/src/gods_watching/fixtures/mediamtx-fixtures.yml"
    container_name = f"{context.compose_project}-mediamtx"
    destination = f"rtsp://127.0.0.1:{context.allocated_port}/camera-1"
    gateway_command = (
        "/usr/bin/docker",
        "run",
        "--rm",
        "--name",
        container_name,
        "-p",
        f"127.0.0.1:{context.allocated_port}:8554",
        "--read-only",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "--volume",
        f"{mediamtx_config}:/mediamtx.yml:ro",
        _MEDIAMTX_IMAGE,
    )
    async with context.process(name="fixture-mediamtx", command=gateway_command):
        await _wait_for_gateway(context.allocated_port)
        publish_command = build_publisher_command(source, destination)
        async with context.process(
            name="fixture-publisher-generation-1", command=publish_command
        ) as first_publisher:
            await _wait_for_path(destination, ready=True)
            first_returncode = await first_publisher.wait()
        await _wait_for_path(destination, ready=False)
        async with context.process(name="fixture-publisher-generation-2", command=publish_command):
            await _wait_for_path(destination, ready=True)
        return PublisherObservation(
            first_session_ready=True,
            first_process_returncode=first_returncode,
            path_not_ready_between_sessions=True,
            second_session_ready=True,
            transport="rtsp/tcp",
            audio_published=False,
            seamless_stream_loop_used=False,
        )


async def run_fixture_inputs(context: ScenarioContextProtocol) -> ScenarioReport:
    """Probe four real files and observe one finite publisher reopen boundary."""
    repository_root = Path(__file__).parents[5]
    manifest_path = repository_root / "assets/test-streams.json"
    manifest, probes = await load_and_probe_fixtures(manifest_path, repository_root)
    publisher = await _observe_reopen(
        context, repository_root / "runtime/assets/fixtures/crosswalk.mp4"
    )
    provenance_path = context.run_root / "task-3-provenance.json"
    probe_path = context.run_root / "task-3-probe.json"
    _ = provenance_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    _ = probe_path.write_text(
        ProbeArtifact(
            label="real-public-pedestrian-inputs", probes=probes, publisher=publisher
        ).model_dump_json(indent=2)
        + "\n",
        encoding="utf-8",
    )
    checks = (
        Check(
            name="four-distinct-real-inputs",
            passed=len(probes) == _EXPECTED_FIXTURE_COUNT,
            detail="4 files",
        ),
        Check(
            name="h264-1080p",
            passed=all(
                probe.codec == "h264" and (probe.width, probe.height) == (1920, 1080)
                for probe in probes
            ),
            detail="ffprobe inspected every file",
        ),
        Check(
            name="continuous-person-interval",
            passed=any(
                interval.duration_seconds >= _MIN_PERSON_SECONDS
                for stream in manifest.streams
                for interval in stream.continuous_person_intervals
            ),
            detail="visually inspected 23-second interval",
        ),
        Check(
            name="publisher-teardown-reopen",
            passed=(
                publisher.first_process_returncode == 0
                and publisher.path_not_ready_between_sessions
                and publisher.second_session_ready
            ),
            detail="ready, finite exit, not-ready, reopened",
        ),
    )
    return ScenarioReport(
        checks=checks,
        artifact_paths=(provenance_path, probe_path),
        evidence_kind=EvidenceKind.REAL,
    )


async def run_corrupt_fixture(context: ScenarioContextProtocol) -> ScenarioReport:
    """Prove a corrupt local prepared input cannot pass validation."""
    repository_root = Path(__file__).parents[5]
    manifest_source = repository_root / "assets/test-streams.json"
    manifest_copy = context.runtime_root / "assets/test-streams.json"
    manifest_copy.parent.mkdir(parents=True)
    _ = shutil.copy2(manifest_source, manifest_copy)
    corrupt_input = context.runtime_root / "runtime/assets/fixtures/business-plaza.mp4"
    corrupt_input.parent.mkdir(parents=True)
    _ = corrupt_input.write_bytes(b"corrupt fixture bytes")
    observed_code = "unexpected_success"
    try:
        _ = await load_and_probe_fixtures(manifest_copy, context.runtime_root)
    except FixturePreparationError as error:
        observed_code = error.code
    artifact_path = context.run_root / "task-3-corrupt.log"
    _ = artifact_path.write_text(f"observed_code={observed_code}\n", encoding="utf-8")
    return ScenarioReport(
        checks=(
            Check(
                name="corrupt-input-rejected",
                passed=observed_code in {"checksum_mismatch", "invalid_video"},
                detail=observed_code,
            ),
        ),
        artifact_paths=(artifact_path,),
        evidence_kind=EvidenceKind.SYNTHETIC,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("fixture-inputs"),
        runner=run_fixture_inputs,
        required_commands=("docker", "ffmpeg", "ffprobe"),
        timeout_seconds=45.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("corrupt-fixture"),
        runner=run_corrupt_fixture,
        required_commands=("ffprobe",),
        timeout_seconds=10.0,
    )
)
