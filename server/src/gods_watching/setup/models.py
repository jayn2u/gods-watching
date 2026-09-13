"""Prepare model assets and validate them without network access."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

from pydantic import BaseModel, ConfigDict


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


def load_models_lock(path: Path) -> ModelsLock:
    """Parse the committed model lock."""
    return ModelsLock.model_validate_json(path.read_text(encoding="utf-8"))


def validate_model_assets(lock: ModelsLock, assets_root: Path) -> tuple[LockedModelFile, ...]:
    """Hash every locked file locally without attempting a download."""
    validated: list[LockedModelFile] = []
    for model in lock.models:
        for locked_file in model.files:
            candidate = assets_root / locked_file.path
            if not candidate.is_file() or candidate.is_symlink():
                raise AssetValidationError(code="asset_missing", path=locked_file.path)
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != locked_file.sha256 or candidate.stat().st_size != locked_file.size:
                raise AssetValidationError(code="asset_corrupt", path=locked_file.path)
            validated.append(locked_file)
    return tuple(validated)
