"""Unit contracts for the durable camera/settings service lane."""

from uuid import uuid4

import anyio
import pytest
from pydantic import ValidationError

from gods_watching.cameras import (
    CameraActivationReason,
    CameraActivationRequest,
    CameraCancellationReason,
    CameraGenerationId,
    CameraLifecyclePlan,
    CameraNameConflictError,
    ProbeFailureCode,
    ProbeResult,
    RtspProbeError,
    RtspSourceProbe,
    parse_rtsp_source,
)
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.settings import (
    DEFAULT_QUOTA_BYTES,
    DEFAULT_RETENTION_DAYS,
)


def test_parse_rtsp_source_rejects_unsafe_inputs_before_probe() -> None:
    # Given: values that are outside the trusted RTSP source boundary
    unsafe_values = (
        "file:///tmp/camera.mp4",
        "https://camera.example/live",
        "data:text/plain,camera",
        "rtsp:///missing-host",
        "rtsp://camera.example:0/live",
        f"rtsp://camera.example/{'x' * 2048}",
    )

    # When / Then: each input is rejected while it is still a string
    for value in unsafe_values:
        with pytest.raises(ValidationError):
            _ = parse_rtsp_source(value)


def test_probe_command_is_shell_free_and_rtsp_tcp_only() -> None:
    # Given: a parsed credential-bearing RTSP source
    source = parse_rtsp_source("rtsp://operator:secret@fixture:8554/lobby")

    # When: the bounded decoder command is constructed
    command = RtspSourceProbe.build_command(source)

    # Then: only the safe decoder surface is exposed
    assert command[0] == "/usr/bin/ffmpeg"
    assert "-protocol_whitelist" in command
    assert command[command.index("-protocol_whitelist") + 1] == "rtsp,tcp"
    assert command[command.index("-rtsp_transport") + 1] == "tcp"
    assert command[command.index("-frames:v") + 1] == "1"
    assert command[-1] == "-"
    assert command[command.index("-i") + 1] == source.url
    assert "secret" not in repr(source)


def test_probe_command_uses_rtsp_timeout_option_supported_by_runtime() -> None:
    # Given: a source that must be opened through the application's FFmpeg runtime
    source = parse_rtsp_source("rtsp://fixture:8554/lobby")

    # When: the one-frame probe command is constructed
    command = RtspSourceProbe.build_command(source)

    # Then: use FFmpeg's RTSP socket-I/O timeout option, in microseconds
    assert command[command.index("-timeout") + 1] == "10000000"
    assert "-stimeout" not in command


def test_probe_result_is_sanitized() -> None:
    # Given: metadata parsed from an actual H.264 1080p frame
    result = ProbeResult(
        source_host="fixture",
        source_port=8554,
        codec="h264",
        width=1920,
        height=1080,
    )

    # When: the result is rendered for a caller
    rendered = result.to_response()

    # Then: only non-secret stream metadata crosses the service boundary
    assert rendered.model_dump() == {
        "source_host": "fixture",
        "source_port": 8554,
        "codec": "h264",
        "width": 1920,
        "height": 1080,
    }


def test_camera_lifecycle_plan_is_typed_for_12c() -> None:
    # Given: one persisted camera generation and a source parsed at the boundary
    camera_id = CameraId(uuid4())
    session_id = CameraSessionId(uuid4())
    source = parse_rtsp_source("rtsp://fixture:8554/lobby")
    plan = CameraLifecyclePlan(
        activation=CameraActivationRequest(
            camera_id=camera_id,
            version=2,
            session_id=session_id,
            generation_id=CameraGenerationId(uuid4()),
            source=source,
            detection_threshold=0.5,
            reason=CameraActivationReason.CREATED,
        ),
        cancellation=None,
    )

    # When / Then: the plan carries only typed, actionable data
    assert plan.activation is not None
    assert plan.activation.camera_id == camera_id
    assert plan.activation.source.host == "fixture"
    assert plan.cancellation is None
    assert CameraCancellationReason.DELETED.value == "deleted"


def test_probe_failure_does_not_expose_source_url() -> None:
    # Given: a typed failure produced for a credential-bearing source
    failure = ProbeFailureCode.AUTHENTICATION_FAILED

    # When / Then: its public diagnostic is stable and secret-free
    assert "rtsp://" not in str(failure)
    assert "password" not in str(failure).lower()


def test_settings_defaults_are_decimal_and_durable_contract_values() -> None:
    # Given / When / Then: the service constants describe the approved defaults
    assert DEFAULT_RETENTION_DAYS == 7
    assert DEFAULT_QUOTA_BYTES == 100_000_000_000


def test_unreachable_probe_is_bounded() -> None:
    # Given: an injected process runner that exits as unreachable
    source = parse_rtsp_source("rtsp://127.0.0.1:1/offline")
    probe = RtspSourceProbe()

    # When / Then: the public error is typed and contains no URL
    with pytest.raises(RtspProbeError) as error:
        _ = anyio.run(probe.probe, source)
    assert "rtsp://" not in str(error.value)


def test_camera_service_has_name_conflict_error_type() -> None:
    # Given / When / Then: callers have a typed persistence error to map in 12c
    error = CameraNameConflictError(name="Lobby")
    assert str(error) == "camera name is already in use"
