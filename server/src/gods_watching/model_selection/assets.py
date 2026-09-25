"""Offline availability checks for immutable prepared model packages.

The application and worker poll this module while rendering model readiness.  It
intentionally contains no inference dependencies (and, in particular, no
``torch`` import).  Expensive file hashing is cached behind file metadata so a
poll does not reread multi-gigabyte checkpoints when an immutable cache has not
changed.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

from pydantic import ValidationError

from gods_watching.model_selection.models import PreparedModelStatus
from gods_watching.setup.models import LockedModel, ModelsLock, load_models_lock

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .registry import ClipModelPackage, ClipModelRegistry

IDENTITY_MARKER_NAME: Final = "gods-watching-model.json"
"""The sidecar name consumed by the CLIP Triton Python backends."""

_DEFAULT_ASSETS_ROOT: Final = Path("/models")
_LOCK_MISMATCH: Final = "lock_mismatch"
_HASH_CHUNK_SIZE: Final = 1024 * 1024


@dataclass(frozen=True, slots=True)
class _PathMetadata:
    """Metadata sufficient to invalidate a cached digest safely."""

    exists: bool
    is_file: bool
    is_symlink: bool
    size: int = 0
    mtime_ns: int = 0
    ctime_ns: int = 0
    inode: int = 0


@dataclass(frozen=True, slots=True)
class _DigestCacheEntry:
    metadata: _PathMetadata
    digest: str | None


@dataclass(frozen=True, slots=True)
class _MarkerCacheEntry:
    metadata: _PathMetadata
    marker: Mapping[str, object] | None
    reason: str | None


class PreparedModelCatalog:
    """Report lock- and sidecar-backed availability for registered packages.

    ``status`` is synchronous because callers use it during startup and in
    lightweight request handlers.  It performs no network, framework, or ML
    imports.  Every locked file is checked for existence, size, and a streamed
    SHA-256 digest before a package is considered prepared.  A package also
    needs an identity marker written by model preparation after all of those
    checks pass.
    """

    def __init__(
        self,
        registry: ClipModelRegistry,
        lock_path: Path,
        assets_root: Path = _DEFAULT_ASSETS_ROOT,
    ) -> None:
        """Create a catalog over a registry, lock file, and mounted asset root."""
        self.registry: ClipModelRegistry = registry
        self.lock_path: Path = Path(lock_path)
        self.assets_root: Path = Path(assets_root)
        self._lock: threading.RLock = threading.RLock()
        self._lock_metadata: _PathMetadata | None = None
        self._loaded_lock: ModelsLock | None = None
        self._lock_reason: str | None = None
        self._digests: dict[Path, _DigestCacheEntry] = {}
        self._markers: dict[Path, _MarkerCacheEntry] = {}
        self._imported: dict[
            Path, tuple[tuple[tuple[str, _PathMetadata], ...], PreparedModelStatus]
        ] = {}

    def status(self, package: ClipModelPackage) -> PreparedModelStatus:
        """Return immutable local readiness for ``package``.

        A missing or malformed lock is reported as unavailable rather than
        escaping into a request handler.  This keeps an unprepared deployment
        fail-closed while preserving an actionable reason for the settings
        catalog and doctor command.
        """
        with self._lock:
            if package.snapshot_path.parts[:3] == ("/", "models", "imported"):
                return self._imported_status(package)
            lock = self._refresh_lock()
            reason: str | None = None
            if lock is None:
                reason = self._lock_reason or "lock_unreadable"
            else:
                locked_model = self._locked_model(lock, package)
                if (
                    locked_model is None
                    or locked_model.revision != package.revision
                    or not self._paths_agree(locked_model, package)
                ):
                    reason = _LOCK_MISMATCH
                else:
                    for locked_file in locked_model.files:
                        candidate = self._asset_path(locked_file.path)
                        reason = self._validate_file(
                            candidate, locked_file.sha256, locked_file.size
                        )
                        if reason is not None:
                            break
                if reason is None:
                    marker_path = self._snapshot_path(package) / IDENTITY_MARKER_NAME
                    marker, reason = self._read_marker(marker_path)
                    expected = {
                        "model_id": package.model_id,
                        "revision": package.revision,
                        "dimension": package.dimension,
                        "processor": package.processor,
                        "runtime": package.runtime,
                    }
                    if reason is None and marker != expected:
                        reason = "identity_marker_mismatch"
            return PreparedModelStatus(prepared=reason is None, reason=reason)

    def invalidate(self) -> None:
        """Drop cached metadata after an external preparation operation."""
        with self._lock:
            self._lock_metadata = None
            self._loaded_lock = None
            self._lock_reason = None
            self._digests.clear()
            self._markers.clear()
            self._imported.clear()

    def _imported_status(self, package: ClipModelPackage) -> PreparedModelStatus:
        from gods_watching.setup.model_preparation import (  # noqa: PLC0415
            GpuProof,
            PreparedManifest,
            validate_gpu_proof,
        )

        from .registry import _load_installed_manifest  # noqa: PLC0415

        directory = self._snapshot_path(package)
        proof_path = self.assets_root / "prepared-manifest.json"
        fingerprint = (
            *(
                (str(path.relative_to(self.assets_root)), _metadata(path))
                for path in sorted(directory.rglob("*"))
            ),
            ("prepared-manifest.json", _metadata(proof_path)),
            ("models.lock.json", _metadata(self.lock_path)),
        )
        cached = self._imported.get(directory)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        try:
            manifest = _load_installed_manifest(directory)
            if (
                manifest["model_id"] != package.model_id
                or manifest["revision"] != package.revision
                or manifest["dimension"] != package.dimension
            ):
                result = PreparedModelStatus(prepared=False, reason="imported_package_invalid")
            elif not proof_path.is_file():
                result = PreparedModelStatus(prepared=False, reason="imported_proof_missing")
            else:
                record = PreparedManifest.model_validate_json(
                    proof_path.read_text(encoding="utf-8")
                )
                matching = tuple(
                    proof for proof in record.model_proofs if proof.model_id == package.model_id
                )
                if (
                    len(matching) != 1
                    or not record.image_id
                    or record.lock_sha256 != _stream_sha256(self.lock_path)
                ):
                    result = PreparedModelStatus(prepared=False, reason="imported_proof_invalid")
                else:
                    proof = GpuProof(
                        python_abi=record.python_abi,
                        torch_version=record.torch_version,
                        torchvision_version=record.torchvision_version,
                        cuda_version=record.cuda_version,
                        cuda_device=record.cuda_device,
                        cuda_available=record.cuda_available,
                        cuda_operation=record.cuda_operation,
                        processor=record.processor,
                        clip_class=record.clip_class,
                        yolo_class=record.yolo_class,
                        detector_resident=record.detector_resident,
                        models=matching,
                    )
                    validate_gpu_proof(proof, (package,))
                    result = PreparedModelStatus(prepared=True)
        except (OSError, UnicodeError, ValueError, RuntimeError):
            result = PreparedModelStatus(prepared=False, reason="imported_proof_invalid")
        self._imported[directory] = (fingerprint, result)
        return result

    def _refresh_lock(self) -> ModelsLock | None:
        metadata = _metadata(self.lock_path)
        if metadata == self._lock_metadata:
            return self._loaded_lock
        self._lock_metadata = metadata
        self._loaded_lock = None
        self._lock_reason = None
        if not metadata.exists or not metadata.is_file or metadata.is_symlink:
            self._lock_reason = "lock_unreadable"
            return None
        try:
            self._loaded_lock = load_models_lock(self.lock_path)
        except (OSError, UnicodeError):
            self._lock_reason = "lock_unreadable"
        except (TypeError, ValidationError, ValueError):
            self._lock_reason = "lock_invalid"
        return self._loaded_lock

    def _locked_model(self, lock: ModelsLock, package: ClipModelPackage) -> LockedModel | None:
        matches = tuple(model for model in lock.models if model.model_id == package.model_id)
        if len(matches) != 1:
            return None
        return matches[0]

    def _paths_agree(self, model: LockedModel, package: ClipModelPackage) -> bool:
        snapshot_path = self._snapshot_path(package)
        if not model.files:
            return False
        for locked_file in model.files:
            path = Path(locked_file.path)
            if path.is_absolute() or ".." in path.parts:
                return False
            candidate = self._asset_path(path)
            try:
                _ = candidate.relative_to(snapshot_path)
            except ValueError:
                return False
        return True

    def _snapshot_path(self, package: ClipModelPackage) -> Path:
        configured = package.snapshot_path
        try:
            _ = configured.relative_to(self.assets_root)
        except ValueError:
            try:
                relative = configured.relative_to(_DEFAULT_ASSETS_ROOT)
            except ValueError:
                relative = Path(package.model_id.rsplit("/", maxsplit=1)[-1])
            configured = self.assets_root / relative
        return configured

    def _asset_path(self, relative: Path) -> Path:
        return self.assets_root / relative

    def _validate_file(self, path: Path, expected_digest: str, expected_size: int) -> str | None:
        metadata = _metadata(path)
        cached = self._digests.get(path)
        if cached is not None and cached.metadata == metadata:
            digest = cached.digest
        else:
            digest = _stream_sha256(path) if metadata.exists and metadata.is_file else None
            self._digests[path] = _DigestCacheEntry(metadata=metadata, digest=digest)
        if metadata.is_symlink or not metadata.exists or not metadata.is_file:
            return "asset_missing"
        if metadata.size != expected_size or digest != expected_digest:
            return "asset_corrupt"
        return None

    def _read_marker(self, path: Path) -> tuple[Mapping[str, object] | None, str | None]:
        metadata = _metadata(path)
        cached = self._markers.get(path)
        if cached is not None and cached.metadata == metadata:
            return cached.marker, cached.reason
        if metadata.is_symlink or not metadata.exists or not metadata.is_file:
            result = _MarkerCacheEntry(metadata, None, "identity_marker_missing")
            self._markers[path] = result
            return result.marker, result.reason
        try:
            value = cast("object", json.loads(path.read_text(encoding="utf-8")))
        except (OSError, UnicodeError, ValueError):
            result = _MarkerCacheEntry(metadata, None, "identity_marker_invalid")
            self._markers[path] = result
            return result.marker, result.reason
        if not isinstance(value, dict):
            result = _MarkerCacheEntry(metadata, None, "identity_marker_invalid")
            self._markers[path] = result
            return result.marker, result.reason
        raw_marker = cast("dict[object, object]", value)
        if any(not isinstance(key, str) for key in raw_marker):
            result = _MarkerCacheEntry(metadata, None, "identity_marker_invalid")
            self._markers[path] = result
            return result.marker, result.reason
        marker: dict[str, object] = {
            key: marker_value for key, marker_value in raw_marker.items() if isinstance(key, str)
        }
        result = _MarkerCacheEntry(metadata, marker, None)
        self._markers[path] = result
        return result.marker, result.reason


def _metadata(path: Path) -> _PathMetadata:
    try:
        stat_result = path.lstat()
    except OSError:
        return _PathMetadata(exists=False, is_file=False, is_symlink=False)
    return _PathMetadata(
        exists=True,
        is_file=path.is_file(),
        is_symlink=path.is_symlink(),
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
        inode=stat_result.st_ino,
    )


def _stream_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


__all__ = ["IDENTITY_MARKER_NAME", "PreparedModelCatalog", "PreparedModelStatus"]
