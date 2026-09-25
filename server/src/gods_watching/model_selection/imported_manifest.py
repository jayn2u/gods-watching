"""Strict, content-addressed metadata for offline CLIP packages."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .registry import BUILTIN_CLIP_MODELS

CLIP_DIMENSION = 512
CLIP_PATCH_SIZE = 16
BASE_VISION_CONFIG = {
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "image_size": 224,
    "patch_size": 16,
}
BASE_TEXT_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 2048,
    "num_hidden_layers": 12,
    "num_attention_heads": 8,
    "vocab_size": 49408,
    "max_position_embeddings": 77,
}
MIN_SAFETENSORS_HEADER_SIZE = 2
PAIR_SIZE = 2


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


def _require_regular(source: Path, path: Path) -> None:
    """Reject links and special files before any payload is opened."""
    try:
        relative = path.relative_to(source)
        current = source
        for component in relative.parts:
            current = current / component
            mode = current.lstat().st_mode
            if current == path:
                if not stat.S_ISREG(mode):
                    code = "invalid_package_file"
                    raise ClipPackageImportError(code)
            elif not stat.S_ISDIR(mode):
                code = "invalid_package_file"
                raise ClipPackageImportError(code)
    except (OSError, ValueError) as error:
        code = "invalid_package_file"
        raise ClipPackageImportError(code) from error


def _validate_safetensors(path: Path) -> None:  # noqa: C901
    """Check the local container layout and presence of both encoder namespaces."""
    try:
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if not MIN_SAFETENSORS_HEADER_SIZE <= header_size <= 16 * 1024 * 1024:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            header = json.loads(stream.read(header_size))
            payload_size = path.stat().st_size - 8 - header_size
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        if (
            not isinstance(header, dict)
            or not tensors
            or not any(k.startswith("text_model.") for k in tensors)
            or not any(k.startswith("vision_model.") for k in tensors)
        ):
            code = "invalid_safetensors"
            raise ClipPackageImportError(code)
        spans: list[tuple[int, int]] = []
        for value in tensors.values():
            if not isinstance(value, dict) or set(value) != {"dtype", "shape", "data_offsets"}:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            offsets = value["data_offsets"]
            shape = value["shape"]
            if (
                value["dtype"] not in {"F16", "F32", "BF16"}
                or not isinstance(shape, list)
                or any(type(d) is not int or d < 1 for d in shape)
                or not isinstance(offsets, list)
                or len(offsets) != PAIR_SIZE
                or any(type(n) is not int for n in offsets)
                or not 0 <= offsets[0] < offsets[1] <= payload_size
            ):
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            bytes_per_element = {"F16": 2, "BF16": 2, "F32": 4}[value["dtype"]]
            if offsets[1] - offsets[0] != math.prod(shape) * bytes_per_element:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            spans.append((offsets[0], offsets[1]))
        position = 0
        for start, end in sorted(spans):
            if start != position:
                code = "invalid_safetensors"
                raise ClipPackageImportError(code)
            position = end
        if position != payload_size:
            code = "invalid_safetensors"
            raise ClipPackageImportError(code)
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        struct.error,
        json.JSONDecodeError,
    ) as error:
        code = "invalid_safetensors"
        raise ClipPackageImportError(code) from error


def _validate_support_files(source: Path, report_name: str) -> None:  # noqa: C901
    """Check local processor, tokenizer, and provenance document shapes."""
    processor = _read_json(source / "preprocessor_config.json")
    if processor.get("do_resize") is not True or not isinstance(processor.get("size"), (int, dict)):
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    tokens = _read_json(source / "special_tokens_map.json")
    unknown_token = tokens.get("unk_token")
    if isinstance(unknown_token, str):
        valid_token = bool(unknown_token)
    elif isinstance(unknown_token, dict):
        allowed = {"content", "single_word", "lstrip", "rstrip", "normalized", "special", "__type"}
        valid_token = (
            set(unknown_token) <= allowed
            and isinstance(unknown_token.get("content"), str)
            and bool(unknown_token["content"])
            and all(
                type(value) is bool
                for key, value in unknown_token.items()
                if key not in {"content", "__type"}
            )
            and unknown_token.get("__type", "AddedToken") == "AddedToken"
        )
    else:
        valid_token = False
    if not valid_token:
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    vocab = _read_json(source / "vocab.json")
    if not vocab or any(
        not isinstance(k, str) or type(v) is not int or v < 0 for k, v in vocab.items()
    ):
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    try:
        merges = (source / "merges.txt").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code) from error
    if (
        not merges
        or not merges[0].startswith("#version:")
        or not any(len(line.split()) == PAIR_SIZE for line in merges[1:] if line.strip())
    ):
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    report = _read_json(source / report_name)
    required_text = (
        "dataset_split",
        "protocol",
        "source_checkpoint",
        "source_checkpoint_revision",
        "candidate_weights_sha256",
        "evaluation_code_revision",
        "metric_definition",
    )
    if any(
        not isinstance(report.get(k), str) or not report[k].strip() for k in required_text
    ) or any(
        type(report.get(k)) not in (float, int) for k in ("baseline_score", "candidate_score")
    ):
        code = "invalid_cuhk_report"
        raise ClipPackageImportError(code)
    if not re.fullmatch(r"[0-9a-f]{64}", report["candidate_weights_sha256"]):
        code = "invalid_cuhk_report"
        raise ClipPackageImportError(code)
    digest = hashlib.sha256()
    with (source / "model.safetensors").open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
    if report["candidate_weights_sha256"] != digest.hexdigest():
        code = "cuhk_candidate_weights_mismatch"
        raise ClipPackageImportError(code)


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


def parse_source_manifest(  # noqa: C901, PLR0912, PLR0915
    source: Path, *, installed_manifest: ImportedClipManifest | None = None
) -> ImportedClipManifest:
    """Validate metadata and every payload before producing its immutable identity."""
    if installed_manifest is None:
        _require_regular(source, source / "package.json")
        meta = _read_json(source / "package.json")
    else:
        _require_regular(source, source / "manifest.json")
        meta = {
            "model_id": installed_manifest.model_id,
            "display_name": installed_manifest.display_name,
            "base_model_id": installed_manifest.base_model_id,
            "dimension": installed_manifest.dimension,
            "files": [vars_file(item) for item in installed_manifest.files],
            "cuhk_report": installed_manifest.cuhk_report,
        }
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
    if meta["model_id"] in {model.model_id for model in BUILTIN_CLIP_MODELS}:
        code = "reserved_model_id"
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
    metadata_name = "manifest.json" if installed_manifest is not None else "package.json"
    if actual != names | {metadata_name}:
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
    for file in files:
        _require_regular(source, source / _safe_relative(file.path))
    config = _read_json(source / "config.json")
    if (
        not isinstance(config.get("vision_config"), dict)
        or not isinstance(config.get("text_config"), dict)
        or config.get("model_type") != "clip"
        or config.get("architectures") != ["CLIPModel"]
        or config.get("projection_dim") != CLIP_DIMENSION
        or config.get("vision_config", {}).get("patch_size") != CLIP_PATCH_SIZE
        or any(
            config["vision_config"].get(key) != value for key, value in BASE_VISION_CONFIG.items()
        )
        or any(config["text_config"].get(key) != value for key, value in BASE_TEXT_CONFIG.items())
        or "auto_map" in config
        or "trust_remote_code" in config
    ):
        code = "unsupported_clip_architecture"
        raise ClipPackageImportError(code)
    if _read_json(source / "tokenizer_config.json").get("tokenizer_class") != "CLIPTokenizer":
        code = "unsupported_clip_processor"
        raise ClipPackageImportError(code)
    _validate_support_files(source, meta["cuhk_report"])
    _validate_safetensors(source / "model.safetensors")
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
        tuple(sorted(files, key=lambda f: f.path)),
        digest,
        meta["cuhk_report"],
    )
