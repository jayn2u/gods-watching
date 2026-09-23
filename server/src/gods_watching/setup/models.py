"""Prepare model assets and validate them without network access."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final, override

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry

_HASH_CHUNK_SIZE: Final = 1024 * 1024


class LockedModelFile(BaseModel):
    """One immutable file in a prepared model snapshot."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    path: Path
    sha256: str
    size: int


class LockedModel(BaseModel):
    """The exact source revision and files for one model."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    revision: str
    license: str
    source: str
    notice: str = ""
    files: tuple[LockedModelFile, ...]


class LockedContainer(BaseModel):
    """Immutable Triton build inputs."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    base_image: str
    base_digest: str
    built_image: str
    python_abi: str
    notice: str = ""


class ModelsLock(BaseModel):
    """The committed model and container trust boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    container: LockedContainer
    models: tuple[LockedModel, ...]


@dataclass(frozen=True, slots=True)
class AssetValidationError(RuntimeError):
    """A stable offline validation failure."""

    code: str
    path: Path

    @override
    def __str__(self) -> str:
        return json.dumps({"code": self.code, "path": str(self.path)}, separators=(",", ":"))


class LockRegistryMismatchError(ValueError):
    """Report a lock entry that cannot describe the immutable model registry."""

    mismatches: tuple[str, ...]

    def __init__(self, mismatches: tuple[str, ...]) -> None:
        """Capture all mismatches so preparation can show one actionable error."""
        self.mismatches = mismatches
        super().__init__("model lock and registry disagree: " + ", ".join(mismatches))


def load_models_lock(path: Path) -> ModelsLock:
    """Parse the committed model lock."""
    return ModelsLock.model_validate_json(path.read_text(encoding="utf-8"))


def validate_model_assets(lock: ModelsLock, assets_root: Path) -> tuple[LockedModelFile, ...]:
    """Hash every locked file locally without attempting a download."""
    validated: list[LockedModelFile] = []
    for model in lock.models:
        for locked_file in model.files:
            if locked_file.path.is_absolute() or ".." in locked_file.path.parts:
                raise AssetValidationError(code="asset_invalid_path", path=locked_file.path)
            candidate = assets_root / locked_file.path
            try:
                metadata = candidate.lstat()
            except OSError:
                raise AssetValidationError(code="asset_missing", path=locked_file.path) from None
            if candidate.is_symlink() or not candidate.is_file():
                raise AssetValidationError(code="asset_missing", path=locked_file.path)
            digest = stream_sha256(candidate)
            if digest != locked_file.sha256 or metadata.st_size != locked_file.size:
                raise AssetValidationError(code="asset_corrupt", path=locked_file.path)
            validated.append(locked_file)
    return tuple(validated)


def stream_sha256(path: Path) -> str:
    """Return a file's SHA-256 digest using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_registry_agreement(
    lock: ModelsLock,
    registry: ClipModelRegistry,
    *,
    require_complete: bool = True,
) -> None:
    """Ensure registered CLIP identities and lock revisions/paths are aligned.

    The legacy synthetic locks used by offline unit tests may contain no
    registered model at all.  Those remain valid when ``require_complete`` is
    enabled.  Once any built-in package appears, every built-in package must be
    represented so a partial lock cannot advertise a ready catalog.
    """
    lock_by_id = {model.model_id: model for model in lock.models}
    registered_ids = tuple(package.model_id for package in registry.packages)
    present_ids = tuple(model_id for model_id in registered_ids if model_id in lock_by_id)
    if not present_ids:
        return
    mismatches: list[str] = []
    if require_complete:
        mismatches.extend(
            f"missing:{model_id}" for model_id in registered_ids if model_id not in lock_by_id
        )
    for package in registry.packages:
        model = lock_by_id.get(package.model_id)
        if model is None:
            continue
        if model.revision != package.revision:
            mismatches.append(f"revision:{package.model_id}")
        if _has_registry_path_mismatch(package, model):
            mismatches.append(f"path:{package.model_id}")
    if mismatches:
        raise LockRegistryMismatchError(tuple(dict.fromkeys(mismatches)))


def _has_registry_path_mismatch(package: ClipModelPackage, model: LockedModel) -> bool:
    """Return whether a lock's relative files escape its package root."""
    expected_root = package.snapshot_path.relative_to(Path("/models"))
    for locked_file in model.files:
        path = Path(locked_file.path)
        if path.is_absolute() or ".." in path.parts:
            return True
        try:
            _ = path.relative_to(expected_root)
        except ValueError:
            return True
    return False


__all__ = [
    "AssetValidationError",
    "LockRegistryMismatchError",
    "LockedContainer",
    "LockedModel",
    "LockedModelFile",
    "ModelsLock",
    "load_models_lock",
    "stream_sha256",
    "validate_model_assets",
    "validate_registry_agreement",
]
