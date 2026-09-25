"""Behavioral tests for local fixture video preparation."""

import hashlib
import json
from pathlib import Path

import pytest

from gods_watching.journeys.fixtures import FixtureDownloadError, ensure_fixture_videos


def write_manifest(
    root: Path,
    *,
    prepared_path: str = "runtime/assets/fixtures/camera.mp4",
    direct_url: str = "https://fixture.example/camera.mp4",
    digest: str,
) -> Path:
    """Write the smallest supported test-stream manifest."""
    manifest = root / "assets" / "test-streams.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    _ = manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "streams": [
                    {
                        "stream_id": "camera-1",
                        "prepared_path": prepared_path,
                        "direct_url": direct_url,
                        "sha256": digest,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_valid_fixture_file_is_left_untouched(tmp_path: Path) -> None:
    """An existing file with the expected SHA-256 does not trigger a download."""
    payload = b"prepared camera video"
    destination = tmp_path / "runtime/assets/fixtures/camera.mp4"
    destination.parent.mkdir(parents=True)
    _ = destination.write_bytes(payload)
    manifest = write_manifest(tmp_path, digest=hashlib.sha256(payload).hexdigest())
    original_stat = destination.stat()

    def download(url: str, target: Path) -> None:
        del url, target
        failure_message = "unexpected fixture download"
        raise AssertionError(failure_message)

    ensure_fixture_videos(manifest, tmp_path, download=download)

    assert destination.read_bytes() == payload
    assert destination.stat().st_mtime_ns == original_stat.st_mtime_ns


def test_missing_fixture_is_downloaded_to_a_verified_file(tmp_path: Path) -> None:
    """A missing stream is downloaded, hashed, and atomically installed."""
    payload = b"downloaded fixture payload"
    manifest = write_manifest(tmp_path, digest=hashlib.sha256(payload).hexdigest())
    calls: list[tuple[str, Path]] = []

    def download(url: str, target: Path) -> None:
        calls.append((url, target))
        _ = target.write_bytes(payload)

    ensure_fixture_videos(manifest, tmp_path, download=download)

    destination = tmp_path / "runtime/assets/fixtures/camera.mp4"
    assert destination.read_bytes() == payload
    assert calls[0][0] == "https://fixture.example/camera.mp4"
    assert calls[0][1].parent == destination.parent
    assert calls[0][1] != destination
    assert list(destination.parent.glob("*.tmp")) == []


def test_bad_download_hash_leaves_no_partial_destination(tmp_path: Path) -> None:
    """A mismatched download is removed without installing partial fixture bytes."""
    manifest = write_manifest(tmp_path, digest=hashlib.sha256(b"expected").hexdigest())

    def download(url: str, target: Path) -> None:
        del url
        _ = target.write_bytes(b"wrong video")

    with pytest.raises(FixtureDownloadError):
        ensure_fixture_videos(manifest, tmp_path, download=download)

    destination = tmp_path / "runtime/assets/fixtures/camera.mp4"
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


def test_empty_stale_bind_mount_directory_is_replaced_with_a_file(tmp_path: Path) -> None:
    """An empty directory at a prepared video path is repaired before download."""
    payload = b"fixture bytes"
    destination = tmp_path / "runtime/assets/fixtures/camera.mp4"
    destination.mkdir(parents=True)
    manifest = write_manifest(tmp_path, digest=hashlib.sha256(payload).hexdigest())

    def download(url: str, target: Path) -> None:
        del url
        _ = target.write_bytes(payload)

    ensure_fixture_videos(manifest, tmp_path, download=download)

    assert destination.is_file()
    assert destination.read_bytes() == payload


def test_nonempty_stale_bind_mount_directory_is_preserved(tmp_path: Path) -> None:
    """A directory containing operator data is never removed for a fixture path."""
    destination = tmp_path / "runtime/assets/fixtures/camera.mp4"
    destination.mkdir(parents=True)
    marker = destination / "keep.txt"
    _ = marker.write_text("operator data", encoding="utf-8")
    manifest = write_manifest(tmp_path, digest=hashlib.sha256(b"video").hexdigest())

    with pytest.raises(FixtureDownloadError):
        ensure_fixture_videos(manifest, tmp_path, download=lambda _url, _path: None)

    assert marker.read_text(encoding="utf-8") == "operator data"
