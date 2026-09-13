"""Checksum and inspect prepared local fixture media."""

import hashlib
from pathlib import Path
from typing import Final

import anyio
from pydantic import ValidationError

from .errors import FixturePreparationError
from .manifest import load_fixture_manifest
from .models import FFProbeDocument, FixtureManifest, FixtureProbe, FixtureStream

_FIXTURE_ROOT: Final = Path("runtime/assets/fixtures")
_MAX_FIXTURE_BYTES: Final = 512_000_000
_PROBE_TIMEOUT_SECONDS: Final = 10.0
_DURATION_TOLERANCE_SECONDS: Final = 0.01


def _fixture_path(repository_root: Path, stream: FixtureStream) -> Path:
    fixture_root = (repository_root / _FIXTURE_ROOT).resolve()
    candidate = repository_root / stream.prepared_path
    if Path(stream.prepared_path).is_absolute() or ".." in Path(stream.prepared_path).parts:
        raise FixturePreparationError(code="unsafe_fixture_path", detail=stream.stream_id)
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise FixturePreparationError(code="missing_fixture", detail=stream.stream_id) from error
    if not resolved.is_relative_to(fixture_root) or candidate.is_symlink():
        raise FixturePreparationError(code="unsafe_fixture_path", detail=stream.stream_id)
    size_bytes = resolved.stat().st_size
    if size_bytes < 1 or size_bytes > _MAX_FIXTURE_BYTES:
        raise FixturePreparationError(code="fixture_size", detail=stream.stream_id)
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fixture:
        while chunk := fixture.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


async def _probe(stream: FixtureStream, path: Path) -> FixtureProbe:
    command = (
        "/usr/bin/ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate:format=duration,size",
        "-of",
        "json",
        str(path),
    )
    try:
        with anyio.fail_after(_PROBE_TIMEOUT_SECONDS):
            completed = await anyio.run_process(command, check=False)
    except TimeoutError as error:
        raise FixturePreparationError(code="probe_timeout", detail=stream.stream_id) from error
    except OSError as error:
        raise FixturePreparationError(code="probe_unavailable", detail=stream.stream_id) from error
    if completed.returncode != 0:
        raise FixturePreparationError(code="invalid_video", detail=stream.stream_id)
    try:
        document = FFProbeDocument.model_validate_json(completed.stdout)
    except ValidationError as error:
        raise FixturePreparationError(code="invalid_video", detail=stream.stream_id) from error
    if len(document.streams) != 1:
        raise FixturePreparationError(code="invalid_video", detail=stream.stream_id)
    video = document.streams[0]
    return FixtureProbe(
        stream_id=stream.stream_id,
        path=stream.prepared_path,
        sha256=_sha256(path),
        duration_seconds=float(document.format.duration),
        fps=video.r_frame_rate,
        width=video.width,
        height=video.height,
        codec=video.codec_name,
        size_bytes=int(document.format.size),
    )


def _require_manifest_match(stream: FixtureStream, probe: FixtureProbe) -> None:
    if probe.sha256 != stream.sha256:
        raise FixturePreparationError(code="checksum_mismatch", detail=stream.stream_id)
    observed = (probe.codec, probe.width, probe.height, probe.fps)
    expected = (stream.codec, stream.width, stream.height, stream.fps)
    if observed != expected:
        raise FixturePreparationError(code="probe_mismatch", detail=stream.stream_id)
    if abs(probe.duration_seconds - stream.duration_seconds) > _DURATION_TOLERANCE_SECONDS:
        raise FixturePreparationError(code="duration_mismatch", detail=stream.stream_id)


async def load_and_probe_fixtures(
    manifest_path: Path, repository_root: Path
) -> tuple[FixtureManifest, tuple[FixtureProbe, ...]]:
    """Validate every prepared input against one complete attributed manifest."""
    manifest = load_fixture_manifest(manifest_path)
    probes: list[FixtureProbe] = []
    for stream in manifest.streams:
        probe = await _probe(stream, _fixture_path(repository_root, stream))
        _require_manifest_match(stream, probe)
        probes.append(probe)
    return manifest, tuple(probes)
