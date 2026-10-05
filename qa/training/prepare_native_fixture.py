"""Prepare a fixed, original-split CUHK-PEDES subset for isolated runtime QA."""

# This executable is QA-only and is not part of the application package.
# ruff: noqa: INP001, TRY003, EM101, EM102

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, BinaryIO, cast

from gods_watching.training.dataset import DatasetManifest, DatasetValidationError, validate_cuhk

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

_FIXTURE_NAME = "cuhk-pedes-native-smoke"
_TRAIN_IMAGES = 1024
_TARGET_IMAGES: Mapping[str, int] = {"train": _TRAIN_IMAGES, "val": 16, "test": 16}
_COPY_CHUNK_SIZE = 1024 * 1024
_FINGERPRINT_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "output" / "training" / "runtime-data"


class FixturePreparationError(RuntimeError):
    """Report unsafe sources or a source dataset that cannot form the fixed fixture."""


@dataclass(slots=True)
class _Selection:
    chosen: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {split: [] for split in _TARGET_IMAGES}
    )
    seen_ids: dict[str, set[int]] = field(
        default_factory=lambda: {split: set() for split in _TARGET_IMAGES}
    )
    seen_paths: set[str] = field(default_factory=set)
    rows: list[dict[str, object]] = field(default_factory=list)


def _resolve_source_root(source_root: Path) -> Path:
    try:
        root = source_root.resolve(strict=True)
    except OSError as error:
        raise FixturePreparationError("original dataset root is unavailable") from error
    if not root.is_dir():
        raise FixturePreparationError("original dataset root is not a directory")
    return root


def _read_annotations(source_root: Path) -> tuple[bytes, list[object]]:
    annotation_path = source_root / "reid_raw.json"
    try:
        details = annotation_path.lstat()
    except OSError as error:
        raise FixturePreparationError("original reid_raw.json is unavailable") from error
    if not stat.S_ISREG(details.st_mode):
        raise FixturePreparationError("original reid_raw.json must be a regular non-symlink file")
    try:
        payload = annotation_path.read_bytes()
        value = cast("object", json.loads(payload))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FixturePreparationError("original reid_raw.json is unreadable or invalid") from error
    if not isinstance(value, list):
        raise FixturePreparationError("original reid_raw.json must contain an annotation list")
    return payload, cast("list[object]", value)


def _lstat(path: Path, relative_path: str) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as error:
        raise FixturePreparationError(
            f"selected original image is unavailable: {relative_path}"
        ) from error


def _safe_relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise FixturePreparationError("selected image path must be a POSIX relative path")
    pure_path = PurePosixPath(value)
    parts = value.split("/")
    if (
        pure_path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise FixturePreparationError("selected image path escapes the original imgs directory")
    return "/".join(parts)


def _source_image_path(source_root: Path, relative_path: str) -> Path:
    image_root = source_root / "imgs"
    root_details = _lstat(image_root, relative_path)
    if not stat.S_ISDIR(root_details.st_mode):
        raise FixturePreparationError("original imgs root must be a real directory")
    candidate = image_root
    parts = relative_path.split("/")
    for index, part in enumerate(parts):
        candidate = candidate / part
        details = _lstat(candidate, relative_path)
        if stat.S_ISLNK(details.st_mode):
            raise FixturePreparationError("selected image path contains a symlink")
        if index < len(parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise FixturePreparationError("selected image parent is not a directory")
        if index == len(parts) - 1 and not stat.S_ISREG(details.st_mode):
            raise FixturePreparationError("selected source image is not a regular file")
    return candidate


def _append_distinct_row(
    row: dict[str, object],
    split: str,
    selection: _Selection,
) -> None:
    if len(selection.chosen[split]) >= _TARGET_IMAGES[split]:
        return
    person_id = row.get("id")
    if type(person_id) is not int or person_id < 1:
        raise FixturePreparationError("selected annotation identity must be a positive integer")
    if person_id in selection.seen_ids[split]:
        return
    relative_path = _safe_relative_path(row.get("file_path"))
    if relative_path in selection.seen_paths:
        raise FixturePreparationError("selected annotations contain a duplicate image path")
    selection.chosen[split].append(row)
    selection.seen_ids[split].add(person_id)
    selection.seen_paths.add(relative_path)
    selection.rows.append(row)


def _select_native_rows(rows: list[object]) -> tuple[dict[str, object], ...]:
    selection = _Selection()

    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise FixturePreparationError("original annotation rows must be objects")
        row = cast("dict[str, object]", raw_row)
        split = row.get("split")
        if not isinstance(split, str) or split not in _TARGET_IMAGES:
            raise FixturePreparationError("original annotation has an unsupported split")
        _append_distinct_row(row, split, selection)
        if all(len(selection.chosen[name]) == _TARGET_IMAGES[name] for name in _TARGET_IMAGES):
            break

    missing = {
        split: {"images": _TARGET_IMAGES[split], "distinct_ids": len(selection.chosen[split])}
        for split in _TARGET_IMAGES
        if len(selection.chosen[split]) != _TARGET_IMAGES[split]
    }
    if missing:
        raise FixturePreparationError(
            "not enough distinct train identities or native split rows: " + str(missing)
        )
    all_ids = [person_id for split_ids in selection.seen_ids.values() for person_id in split_ids]
    if len(set(all_ids)) != len(all_ids):
        raise FixturePreparationError("selected person IDs overlap across original splits")
    return tuple(selection.rows)


def _copy_image_exact(source: Path, destination: Path) -> str:
    digest = hashlib.sha256()
    try:
        with source.open("rb") as original, destination.open("xb") as fixture:
            while chunk := original.read(_COPY_CHUNK_SIZE):
                digest.update(chunk)
                _write_chunk(fixture, chunk)
            fixture.flush()
            os.fsync(fixture.fileno())
    except OSError as error:
        raise FixturePreparationError(f"could not copy selected image: {source.name}") from error
    destination.chmod(0o644)
    return digest.hexdigest()


def _write_chunk(destination: BinaryIO, chunk: bytes) -> None:
    if destination.write(chunk) != len(chunk):
        raise OSError("short write while copying image bytes")


def _copy_selected_images(
    source_root: Path,
    fixture_root: Path,
    selected_rows: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    copy_hashes: dict[str, str] = {}
    for row in selected_rows:
        relative_path = _safe_relative_path(row.get("file_path"))
        source = _source_image_path(source_root, relative_path)
        destination = fixture_root / "imgs" / Path(*relative_path.split("/"))
        _ = destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        copy_hashes[relative_path] = _copy_image_exact(source, destination)
    return copy_hashes


def _manifest_split_counts(manifest: DatasetManifest) -> dict[str, dict[str, int]]:
    return {
        split: {
            "images": counts.images,
            "captions": counts.captions,
            "identities": counts.identities,
        }
        for split, counts in manifest.split_counts.items()
    }


def _publish_permissions(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)
    root.chmod(0o755)


def _validate_fixture(
    fixture_root: Path,
    copy_hashes: Mapping[str, str],
) -> tuple[DatasetManifest, dict[str, int]]:
    try:
        manifest = validate_cuhk(fixture_root)
    except DatasetValidationError as error:
        raise FixturePreparationError("native subset failed the CUHK-PEDES validator") from error
    expected_counts = dict(_TARGET_IMAGES)
    actual_images = {split: counts.images for split, counts in manifest.split_counts.items()}
    actual_identities = {
        split: counts.identities for split, counts in manifest.split_counts.items()
    }
    if actual_images != expected_counts or actual_identities != expected_counts:
        raise FixturePreparationError(
            "validated fixture counts differ from the fixed native targets"
        )
    fixture_hashes = {sample.relative_path: sample.image_sha256 for sample in manifest.samples}
    if fixture_hashes != copy_hashes or len(fixture_hashes) != sum(_TARGET_IMAGES.values()):
        raise FixturePreparationError("copied fixture images do not match their source byte hashes")
    train_identities = {sample.person_id for sample in manifest.samples if sample.split == "train"}
    if len(train_identities) != _TRAIN_IMAGES:
        raise FixturePreparationError("fixture must contain 1024 distinct train identities")
    return manifest, actual_identities


def _write_fixture_manifest(
    fixture_root: Path,
    original_fingerprint: str,
    original_annotation_sha256: str,
    copy_hashes: Mapping[str, str],
) -> None:
    manifest, actual_identities = _validate_fixture(fixture_root, copy_hashes)
    fixture_annotation_sha256 = hashlib.sha256(
        (fixture_root / "reid_raw.json").read_bytes()
    ).hexdigest()
    fixture_record = {
        "schema_version": 1,
        "purpose": "qa-native-cuhk-pedes-runtime-smoke",
        "original_fingerprint": original_fingerprint,
        "original_fingerprint_source": ("previous authenticated full-dataset readiness proof"),
        "original_annotation_sha256": original_annotation_sha256,
        "fixture_fingerprint": manifest.fingerprint,
        "fixture_annotation_sha256": fixture_annotation_sha256,
        "selection": {
            "policy": "first annotation row per person ID, retained in source annotation order",
            "image_counts": dict(_TARGET_IMAGES),
            "train_distinct_identities": actual_identities["train"],
            "source_images_modified": False,
        },
        "fixture": {
            "dataset_id": manifest.dataset_id,
            "protocol": manifest.protocol,
            "fingerprint": manifest.fingerprint,
            "image_count": manifest.image_count,
            "caption_count": manifest.caption_count,
            "identity_count": manifest.identity_count,
            "split_counts": _manifest_split_counts(manifest),
        },
        "image_sha256": dict(sorted(copy_hashes.items())),
    }
    record_path = fixture_root / "manifest.json"
    _ = record_path.write_text(json.dumps(fixture_record, sort_keys=True, indent=2) + "\n")
    record_path.chmod(0o644)


def _assert_destination_available(destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FixturePreparationError(f"fixture destination already exists: {destination}")


def prepare_native_fixture(
    *,
    source_root: Path,
    output_root: Path,
    original_fingerprint: str,
) -> Path:
    """Copy a fixed distinct-ID native-split fixture without scanning the full source."""
    if _FINGERPRINT_PATTERN.fullmatch(original_fingerprint) is None:
        raise FixturePreparationError("original fingerprint must be a lowercase SHA-256 digest")
    source = _resolve_source_root(source_root)
    output_parent = output_root.resolve()
    destination = output_parent / _FIXTURE_NAME
    _assert_destination_available(destination)
    if (
        destination == source
        or destination.is_relative_to(source)
        or source.is_relative_to(destination)
    ):
        raise FixturePreparationError(
            "fixture destination must remain outside the original dataset"
        )

    annotation_bytes, rows = _read_annotations(source)
    selected_rows = _select_native_rows(rows)
    original_annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    _ = output_parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{_FIXTURE_NAME}-", dir=output_parent))
    published = False
    try:
        copy_hashes = _copy_selected_images(source, staging, selected_rows)
        annotation_path = staging / "reid_raw.json"
        _ = annotation_path.write_text(
            json.dumps(selected_rows, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        annotation_path.chmod(0o644)
        _publish_permissions(staging)
        _assert_destination_available(destination)
        _ = staging.rename(destination)
        published = True
        _write_fixture_manifest(
            destination,
            original_fingerprint,
            original_annotation_sha256,
            copy_hashes,
        )
    except Exception:
        if published and destination.exists():
            shutil.rmtree(destination)
        elif staging.exists():
            shutil.rmtree(staging)
        raise
    else:
        return destination


@dataclass(frozen=True, slots=True)
class _Arguments:
    source_root: Path
    original_fingerprint: str
    output_root: Path


@dataclass(slots=True)
class _ParsedArguments(argparse.Namespace):
    source_root: Path | None = None
    original_fingerprint: str | None = None
    output_root: Path = _DEFAULT_OUTPUT_ROOT


def _arguments() -> _Arguments:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source-root", type=Path, required=True)
    _ = parser.add_argument("--original-fingerprint", required=True)
    _ = parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
    )
    parsed = _ParsedArguments()
    _ = parser.parse_args(namespace=parsed)
    if parsed.source_root is None or parsed.original_fingerprint is None:
        raise FixturePreparationError("source root and original fingerprint are required")
    return _Arguments(
        source_root=parsed.source_root,
        original_fingerprint=parsed.original_fingerprint,
        output_root=parsed.output_root,
    )


def _read_manifest_record(fixture_root: Path) -> dict[str, object]:
    manifest_path = fixture_root / "manifest.json"
    try:
        value = cast("object", json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as error:
        raise FixturePreparationError("fixture provenance manifest is unreadable") from error
    if not isinstance(value, dict):
        raise FixturePreparationError("fixture provenance manifest must be an object")
    return cast("dict[str, object]", value)


def main() -> int:
    """Run the fixed QA-only fixture generator and print its safe fingerprints."""
    arguments = _arguments()
    try:
        fixture_root = prepare_native_fixture(
            source_root=arguments.source_root,
            output_root=arguments.output_root,
            original_fingerprint=arguments.original_fingerprint,
        )
    except (FixturePreparationError, OSError) as error:
        _ = sys.stderr.write(f"native fixture preparation failed: {error}\n")
        return 2
    manifest = _read_manifest_record(fixture_root)
    fixture_record = manifest.get("fixture")
    if not isinstance(fixture_record, dict):
        raise FixturePreparationError("fixture provenance omitted validated counts")
    split_counts = cast("dict[str, object]", fixture_record).get("split_counts")
    original_fingerprint = manifest.get("original_fingerprint")
    fixture_fingerprint = manifest.get("fixture_fingerprint")
    if (
        not isinstance(split_counts, dict)
        or not isinstance(original_fingerprint, str)
        or not isinstance(fixture_fingerprint, str)
    ):
        raise FixturePreparationError("fixture provenance omitted fingerprints or split counts")
    _ = sys.stdout.write(
        json.dumps(
            {
                "fixture_root": str(fixture_root),
                "original_fingerprint": original_fingerprint,
                "fixture_fingerprint": fixture_fingerprint,
                "split_counts": cast("dict[str, object]", split_counts),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
