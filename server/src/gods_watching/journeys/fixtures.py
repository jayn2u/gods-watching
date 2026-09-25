"""Verify and atomically prepare manifest-listed Journey fixture videos."""

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Final, TypeGuard, cast, override

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_HASH_CHUNK_BYTES: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FixtureDownloadError(RuntimeError):
    """Report invalid fixture metadata, unsafe paths, or failed integrity checks."""

    path: Path
    reason: str

    @override
    def __str__(self) -> str:
        """Describe the fixture failure without including source credentials."""
        return f"fixture video at {self.path} rejected: {self.reason}"


def ensure_fixture_videos(
    manifest_path: Path,
    repository_root: Path,
    *,
    download: Callable[[str, Path], None],
) -> None:
    """Verify each prepared fixture or download it to a same-directory temp file."""
    try:
        manifest = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FixtureDownloadError(manifest_path, "manifest cannot be read") from error
    if not _is_json_object(manifest):
        raise FixtureDownloadError(manifest_path, "manifest must be a JSON object")
    streams = manifest.get("streams")
    if not _is_json_array(streams):
        raise FixtureDownloadError(manifest_path, "manifest streams must be a list")
    root = repository_root.resolve()
    for stream in streams:
        if not _is_json_object(stream):
            raise FixtureDownloadError(manifest_path, "stream entry must be a JSON object")
        prepared_path = stream.get("prepared_path")
        direct_url = stream.get("direct_url")
        expected_hash = stream.get("sha256")
        if not isinstance(prepared_path, str) or not isinstance(direct_url, str):
            raise FixtureDownloadError(manifest_path, "stream path and direct_url must be strings")
        if not isinstance(expected_hash, str) or _SHA256_PATTERN.fullmatch(expected_hash) is None:
            raise FixtureDownloadError(manifest_path, "stream sha256 must be a lowercase SHA-256")
        destination = (root / prepared_path).resolve()
        if not destination.is_relative_to(root):
            raise FixtureDownloadError(destination, "prepared path escapes the repository root")
        _ensure_fixture(destination, expected_hash, direct_url, download)


def _ensure_fixture(
    destination: Path,
    expected_hash: str,
    direct_url: str,
    download: Callable[[str, Path], None],
) -> None:
    if destination.is_dir():
        try:
            destination.rmdir()
        except OSError as error:
            raise FixtureDownloadError(
                destination,
                "stale fixture directory is not empty",
            ) from error
    if destination.is_file() and _sha256(destination) == expected_hash:
        return

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise FixtureDownloadError(destination, "fixture directory could not be created") from error
    temporary_path: Path | None = None
    actual_hash: str | None = None
    try:
        with NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        download(direct_url, temporary_path)
        actual_hash = _sha256(temporary_path)
        if actual_hash == expected_hash:
            _ = temporary_path.replace(destination)
            temporary_path = None
    except Exception as error:
        raise FixtureDownloadError(
            destination,
            f"fixture download or installation failed: {type(error).__name__}: {error}",
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if actual_hash != expected_hash:
        raise FixtureDownloadError(destination, "downloaded fixture SHA-256 does not match")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as video_file:
        while chunk := video_file.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    json_object = cast("dict[object, object]", value)
    return all(isinstance(key, str) for key in json_object)


def _is_json_array(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


__all__ = ["FixtureDownloadError", "ensure_fixture_videos"]
