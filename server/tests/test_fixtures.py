from pathlib import Path

import anyio
import pytest

from gods_watching.fixtures import (
    FixturePreparationError,
    build_publisher_command,
    load_and_probe_fixtures,
    load_fixture_manifest,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_real_manifest_probes_four_prepared_inputs() -> None:
    # Given: the attributed manifest and locally prepared public inputs
    manifest_path = REPOSITORY_ROOT / "assets/test-streams.json"
    # When: every input is parsed, checksummed, and inspected by real ffprobe
    manifest, probes = anyio.run(load_and_probe_fixtures, manifest_path, REPOSITORY_ROOT)
    # Then: exactly four distinct real H.264 1080p clips satisfy the manifest
    assert len(manifest.streams) == 4
    assert len(probes) == 4
    assert {probe.codec for probe in probes} == {"h264"}
    assert {(probe.width, probe.height) for probe in probes} == {(1920, 1080)}
    assert len({stream.scene for stream in manifest.streams}) == 4
    assert any(
        interval.duration_seconds >= 20
        for stream in manifest.streams
        for interval in stream.continuous_person_intervals
    )


def test_missing_input_fails_preparation(tmp_path: Path) -> None:
    # Given: a valid manifest copied beside an empty fixture root
    manifest = load_fixture_manifest(REPOSITORY_ROOT / "assets/test-streams.json")
    manifest_path = tmp_path / "assets/test-streams.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    # When: preparation attempts to probe the absent local inputs
    with pytest.raises(FixturePreparationError) as error:
        _ = anyio.run(load_and_probe_fixtures, manifest_path, tmp_path)
    # Then: the error identifies the missing boundary without downloading anything
    assert error.value.code == "missing_fixture"


def test_corrupt_input_fails_preparation(tmp_path: Path) -> None:
    # Given: a manifest whose first in-root input contains non-video bytes
    manifest = load_fixture_manifest(REPOSITORY_ROOT / "assets/test-streams.json")
    first_path = tmp_path / manifest.streams[0].prepared_path
    first_path.parent.mkdir(parents=True)
    _ = first_path.write_bytes(b"not a video")
    manifest_path = tmp_path / "assets/test-streams.json"
    manifest_path.parent.mkdir(exist_ok=True)
    _ = manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    # When: preparation runs the real media probe
    with pytest.raises(FixturePreparationError) as error:
        _ = anyio.run(load_and_probe_fixtures, manifest_path, tmp_path)
    # Then: corrupt media remains a binary failure
    assert error.value.code in {"checksum_mismatch", "invalid_video"}


def test_path_escape_is_rejected_before_probe(tmp_path: Path) -> None:
    # Given: an otherwise complete manifest with a repository escape path
    manifest = load_fixture_manifest(REPOSITORY_ROOT / "assets/test-streams.json")
    escaped_stream = manifest.streams[0].model_copy(update={"prepared_path": "../outside.mp4"})
    escaped_manifest = manifest.model_copy(
        update={"streams": (escaped_stream, *manifest.streams[1:])}
    )
    manifest_path = tmp_path / "assets/test-streams.json"
    manifest_path.parent.mkdir()
    _ = manifest_path.write_text(escaped_manifest.model_dump_json(), encoding="utf-8")
    # When: preparation resolves the untrusted manifest path
    with pytest.raises(FixturePreparationError) as error:
        _ = anyio.run(load_and_probe_fixtures, manifest_path, tmp_path)
    # Then: the escape is rejected without invoking ffprobe outside the fixture root
    assert error.value.code == "unsafe_fixture_path"


def test_publisher_command_has_one_session_and_no_seamless_loop() -> None:
    # Given: one validated local input and one local RTSP destination
    source = REPOSITORY_ROOT / "runtime/assets/fixtures/crosswalk.mp4"
    # When: the publisher command is built
    command = build_publisher_command(source, "rtsp://127.0.0.1:8554/camera-1")
    # Then: it publishes video-only H.264 over TCP exactly once per process
    assert command[0] == "/usr/bin/ffmpeg"
    assert "-rtsp_transport" in command
    assert command[command.index("-rtsp_transport") + 1] == "tcp"
    assert "-an" in command
    assert "-stream_loop" not in command
    assert command[-1] == "rtsp://127.0.0.1:8554/camera-1"


@pytest.mark.parametrize(
    "destination",
    [("file:///tmp/output",), (f"rtsp://127.0.0.1/{'a' * 2048}",)],
)
def test_publisher_rejects_unbounded_or_non_rtsp_destination(destination: str) -> None:
    # Given: a local input and an invalid destination boundary
    source = REPOSITORY_ROOT / "runtime/assets/fixtures/crosswalk.mp4"
    # When: a publisher command is requested
    with pytest.raises(FixturePreparationError) as error:
        _ = build_publisher_command(source, destination)
    # Then: invalid configuration never reaches an FFmpeg process
    assert error.value.code == "invalid_rtsp_destination"
