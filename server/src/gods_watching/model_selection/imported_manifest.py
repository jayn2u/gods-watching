"""Strict, content-addressed metadata for offline CLIP packages."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CLIP_DIMENSION = 512
CLIP_PATCH_SIZE = 16
MIN_SAFETENSORS_HEADER_SIZE = 2


class ClipPackageImportError(ValueError):
    """An import failure with a stable, path-free public code."""

    def __init__(self, code: str) -> None:
        """Store a stable error code without source paths."""
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class ImportedFile:
    """One verified payload file."""

    path: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportedClipManifest:
    """The published immutable CLIP package record."""

    model_id: str
    revision: str
    display_name: str
    base_model_id: str
    dimension: int
    files: tuple[ImportedFile, ...]
    package_sha256: str
    cuhk_report: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe manifest metadata."""
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "display_name": self.display_name,
            "base_model_id": self.base_model_id,
            "dimension": self.dimension,
            "files": [vars_file(f) for f in self.files],
            "package_sha256": self.package_sha256,
            "cuhk_report": self.cuhk_report,
        }


def vars_file(item: ImportedFile) -> dict[str, Any]:
    """Return JSON-safe file metadata."""
    return {"path": item.path, "size": item.size, "sha256": item.sha256}


def _safe_relative(name: str) -> Path:
    path = Path(name)
    if (
        not name
        or path.is_absolute()
        or any(part in ("", ".", "..") for part in path.parts)
        or "\\" in name
    ):
        code = "invalid_package_path"
        raise ClipPackageImportError(code)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        code = "invalid_package_metadata"
        raise ClipPackageImportError(code) from error
    if not isinstance(value, dict):
        code = "invalid_package_metadata"
        raise ClipPackageImportError(code)
    return value


def parse_source_manifest(source: Path) -> ImportedClipManifest:  # noqa: C901, PLR0912, PLR0915
    """Validate metadata and every payload before producing its immutable identity."""
    meta = _read_json(source / "package.json")
    if set(meta) != {
        "model_id",
        "display_name",
        "base_model_id",
        "dimension",
        "files",
        "cuhk_report",
    }:
        code = "invalid_package_metadata"
        raise ClipPackageImportError(code)
    if not all(
        isinstance(meta[k], str) and meta[k].strip()
        for k in ("model_id", "display_name", "base_model_id", "cuhk_report")
    ):
        code = "invalid_package_metadata"
        raise ClipPackageImportError(code)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", meta["model_id"]) or ".." in meta[
        "model_id"
    ].split("/"):
        code = "invalid_model_id"
        raise ClipPackageImportError(code)
    if (
        meta["base_model_id"] != "openai/clip-vit-base-patch16"
        or type(meta["dimension"]) is not int
        or meta["dimension"] != CLIP_DIMENSION
    ):
        code = "unsupported_clip_architecture"
        raise ClipPackageImportError(code)
    raw_files = meta["files"]
    if not isinstance(raw_files, list) or not raw_files:
        code = "invalid_package_metadata"
        raise ClipPackageImportError(code)
    files: list[ImportedFile] = []
    seen: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "size", "sha256"}:
            code = "invalid_package_metadata"
            raise ClipPackageImportError(code)
        name, size, digest = raw["path"], raw["size"], raw["sha256"]
        if (
            not isinstance(name, str)
            or type(size) is not int
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch("[0-9a-f]{64}", digest)
        ):
            code = "invalid_package_metadata"
            raise ClipPackageImportError(code)
        _safe_relative(name)
        if name in seen or name in {"package.json", "manifest.json"}:
            code = "invalid_package_metadata"
            raise ClipPackageImportError(code)
        seen.add(name)
        files.append(ImportedFile(name, size, digest))
    names = {item.path for item in files}
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if not p.is_dir()}
    if actual != names | {"package.json"}:
        code = "unlisted_package_file"
        raise ClipPackageImportError(code)
    required = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
        meta["cuhk_report"],
    }
    if not required <= names or not meta["cuhk_report"].endswith(".json"):
        code = "missing_clip_file"
        raise ClipPackageImportError(code)
    if any(name.endswith((".bin", ".pkl", ".pickle", ".py", ".pt", ".pth")) for name in names):
        code = "unsafe_package_file"
        raise ClipPackageImportError(code)
    if any(name not in required for name in names):
        code = "unsupported_package_file"
        raise ClipPackageImportError(code)
    if (source / "package.json").is_symlink() or (source / "model.safetensors").is_symlink():
        code = "invalid_package_file"
        raise ClipPackageImportError(code)
    config = _read_json(source / "config.json")
    if (
        config.get("model_type") != "clip"
        or config.get("architectures") != ["CLIPModel"]
        or config.get("projection_dim") != CLIP_DIMENSION
        or config.get("vision_config", {}).get("patch_size") != CLIP_PATCH_SIZE
        or "auto_map" in config
        or "trust_remote_code" in config
    ):
        code = "unsupported_clip_architecture"
        raise ClipPackageImportError(code)
    if _read_json(source / "tokenizer_config.json").get("tokenizer_class") != "CLIPTokenizer":
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    try:
        with (source / "model.safetensors").open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not MIN_SAFETENSORS_HEADER_SIZE <= header_size <= 16 * 1024 * 1024:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            header = json.loads(stream.read(header_size))
            if not isinstance(header, dict) or not header:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
    except (OSError, ValueError, struct.error, json.JSONDecodeError) as error:
        code = "invalid_safetensors"
        raise ClipPackageImportError(code) from error
    for file in files:
        path = source / _safe_relative(file.path)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(
                part.is_symlink()
                for part in path.parents
                if part != source and source in part.parents
            )
        ):
            code = "invalid_package_file"
            raise ClipPackageImportError(code)
        checksum = hashlib.sha256()
        count = 0
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    checksum.update(chunk)
                    count += len(chunk)
        except OSError as error:
            code = "invalid_package_file"
            raise ClipPackageImportError(code) from error
        if count != file.size or checksum.hexdigest() != file.sha256:
            code = "package_hash_mismatch"
            raise ClipPackageImportError(code)
    canonical = json.dumps(
        [vars_file(f) for f in sorted(files, key=lambda f: f.path)],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    return ImportedClipManifest(
        meta["model_id"],
        digest,
        meta["display_name"],
        meta["base_model_id"],
        512,
        tuple(files),
        digest,
        meta["cuhk_report"],
    )
