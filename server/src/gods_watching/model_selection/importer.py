"""Safe local CLIP package publication."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .imported_manifest import ClipPackageImportError, ImportedClipManifest, parse_source_manifest


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def import_clip_package(source: Path, assets_root: Path) -> ImportedClipManifest:
    """Verify a local package and atomically register its immutable copy."""
    source = Path(source)
    if not source.is_dir() or source.is_symlink():
        raise ClipPackageImportError("invalid_package_source")
    try:
        manifest = parse_source_manifest(source)
        imported = Path(assets_root) / "imported"
        imported.mkdir(parents=True, exist_ok=True)
        with (imported / ".import.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for existing in imported.iterdir():
                if not existing.is_dir() or existing.name.startswith("."):
                    continue
                manifest_file = existing / "manifest.json"
                if not manifest_file.is_file():
                    raise ClipPackageImportError("invalid_installed_package")
                try:
                    record = json.loads(manifest_file.read_text())
                except (OSError, json.JSONDecodeError) as error:
                    raise ClipPackageImportError("invalid_installed_package") from error
                if record.get("model_id") == manifest.model_id:
                    if record != manifest.to_dict() or existing.name != manifest.package_sha256:
                        raise ClipPackageImportError("model_id_conflict")
                    return manifest
            destination = imported / manifest.package_sha256
            if destination.exists():
                raise ClipPackageImportError("package_hash_conflict")
            with tempfile.TemporaryDirectory(prefix=".clip-import-", dir=imported) as temporary:
                staging = Path(temporary)
                for file in manifest.files:
                    original = source / file.path
                    target = staging / file.path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    checksum = hashlib.sha256()
                    size = 0
                    with original.open("rb") as reader, target.open("xb") as writer:
                        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                            writer.write(chunk)
                            checksum.update(chunk)
                            size += len(chunk)
                        writer.flush()
                        os.fsync(writer.fileno())
                    if (
                        original.is_symlink()
                        or size != file.size
                        or checksum.hexdigest() != file.sha256
                    ):
                        raise ClipPackageImportError("package_hash_mismatch")
                published = staging / "manifest.json"
                with published.open("x", encoding="utf-8") as stream:
                    json.dump(manifest.to_dict(), stream, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(staging)
                os.rename(staging, destination)
                _fsync_directory(imported)
        return manifest
    except ClipPackageImportError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise ClipPackageImportError("package_import_failed") from error
