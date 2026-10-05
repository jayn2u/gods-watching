"""Validate and fingerprint the fixed CUHK-PEDES train, validation, and test data."""

# ruff: noqa: TRY003, EM101, EM102

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Mapping

DatasetSplit = Literal["train", "val", "test"]
_SPLITS: tuple[DatasetSplit, ...] = ("train", "val", "test")
_SPLIT_SET = frozenset(_SPLITS)
_DATASET_ID = "cuhk-pedes"
_PROTOCOL = "cuhk-pedes-original-splits-v1"
_HASH_CHUNK_SIZE = 1024 * 1024
_CACHE_MAX_ENTRIES = 4
_MIN_TRAIN_IDENTITIES = 2


class DatasetValidationError(ValueError):
    """Report unsafe, malformed, or incomplete CUHK-PEDES inputs."""


@dataclass(frozen=True, slots=True)
class DatasetSplitCounts:
    """Summarize the validated images, captions, and people in one split."""

    images: int
    captions: int
    identities: int


@dataclass(frozen=True, slots=True)
class TrainingSample:
    """One verified image and its split-safe identity and caption choices."""

    split: DatasetSplit
    relative_path: str
    image_path: Path = field(repr=False)
    person_id: int
    captions: tuple[str, ...] = field(repr=False)
    image_sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Internal image sample list plus a safe summary for durable/API snapshots."""

    dataset_id: str
    fingerprint: str
    protocol: str
    root: Path = field(repr=False)
    samples: tuple[TrainingSample, ...] = field(repr=False)
    split_counts: Mapping[str, DatasetSplitCounts]

    @property
    def image_count(self) -> int:
        """Return the total number of verified images."""
        return sum(counts.images for counts in self.split_counts.values())

    @property
    def caption_count(self) -> int:
        """Return the observed caption count rather than a README estimate."""
        return sum(counts.captions for counts in self.split_counts.values())

    @property
    def identity_count(self) -> int:
        """Return the total split-disjoint identity count."""
        return sum(counts.identities for counts in self.split_counts.values())

    def samples_for(self, split: DatasetSplit) -> tuple[TrainingSample, ...]:
        """Return only samples from one existing benchmark split."""
        return tuple(sample for sample in self.samples if sample.split == split)

    def public_snapshot(self) -> dict[str, object]:
        """Return durable metadata without image paths or source captions."""
        return {
            "dataset_id": self.dataset_id,
            "fingerprint": self.fingerprint,
            "protocol": self.protocol,
            "split_counts": {
                split: {
                    "images": counts.images,
                    "captions": counts.captions,
                    "identities": counts.identities,
                }
                for split, counts in self.split_counts.items()
            },
            "image_count": self.image_count,
            "caption_count": self.caption_count,
            "identity_count": self.identity_count,
        }


@dataclass(frozen=True, slots=True)
class _PendingSample:
    split: DatasetSplit
    relative_path: str
    image_path: Path
    person_id: int
    captions: tuple[str, ...]
    file_signature: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _DatasetSource:
    root: Path
    imgs_root: Path
    annotation_path: Path
    annotation_signature: tuple[int, int, int, int, int]
    annotation_bytes: bytes
    samples: tuple[_PendingSample, ...]
    cache_key: str


_MANIFEST_CACHE: OrderedDict[str, DatasetManifest] = OrderedDict()


def _reject_non_finite_constant(token: str) -> None:
    raise DatasetValidationError(f"annotation contains non-finite numeric value: {token}")


def _stat_signature(path: Path) -> tuple[int, int, int, int, int]:
    """Return a content-change-sensitive signature without following symlinks."""
    details = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode):
        raise DatasetValidationError(f"dataset input is not a regular file: {path.name}")
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _safe_image_path(
    imgs_root: Path, raw_path: object
) -> tuple[str, Path, tuple[int, int, int, int, int]]:
    """Resolve one annotation path while rejecting traversal and symlink components."""
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path or "\\" in raw_path:
        raise DatasetValidationError("image path must be a non-empty POSIX relative path")
    pure_path = PurePosixPath(raw_path)
    parts = raw_path.split("/")
    if (
        pure_path.is_absolute()
        or raw_path.startswith("/")
        or len(parts) == 0
        or any(part in {"", ".", ".."} for part in parts)
        or PureWindowsPath(raw_path).is_absolute()
    ):
        raise DatasetValidationError("image path escapes the configured imgs directory")

    candidate = imgs_root
    for index, part in enumerate(parts):
        candidate = candidate / part
        try:
            details = candidate.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise DatasetValidationError(
                f"image file is missing: {PurePosixPath(*parts).name}"
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise DatasetValidationError(
                f"image path contains a symlink: {PurePosixPath(*parts).name}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(details.st_mode):
            raise DatasetValidationError("image path parent is not a directory")
        if index == len(parts) - 1 and not stat.S_ISREG(details.st_mode):
            raise DatasetValidationError(
                f"image file is not a regular file: {PurePosixPath(*parts).name}"
            )

    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(imgs_root):
        raise DatasetValidationError("image path escapes the configured imgs directory")
    signature = _stat_signature(resolved)
    return "/".join(parts), resolved, signature


def _parse_annotations(raw: object, imgs_root: Path) -> tuple[_PendingSample, ...]:  # noqa: C901, PLR0912
    if not isinstance(raw, list) or not raw:
        raise DatasetValidationError("reid_raw.json must contain a non-empty annotation list")

    samples: list[_PendingSample] = []
    paths: dict[str, int] = {}
    identity_splits: dict[int, set[DatasetSplit]] = defaultdict(set)
    split_presence: Counter[str] = Counter()

    rows = cast("list[object]", raw)
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            raise DatasetValidationError("each annotation must be an object")
        row = cast("dict[str, object]", raw_row)
        split_value = row.get("split")
        if not isinstance(split_value, str) or split_value not in _SPLIT_SET:
            raise DatasetValidationError("annotation split must be train, val, or test")
        split = split_value

        person_id = row.get("id")
        if type(person_id) is not int or person_id < 1:
            raise DatasetValidationError("annotation identity must be a positive integer")

        raw_captions = row.get("captions")
        if not isinstance(raw_captions, list) or not raw_captions:
            raise DatasetValidationError("each image must have at least one caption")
        caption_values = cast("list[object]", raw_captions)
        captions: list[str] = []
        for caption in caption_values:
            if not isinstance(caption, str) or not caption.strip():
                raise DatasetValidationError("caption must be non-empty text")
            captions.append(caption)

        relative_path, image_path, file_signature = _safe_image_path(
            imgs_root, row.get("file_path")
        )
        if relative_path in paths:
            if paths[relative_path] != person_id:
                raise DatasetValidationError("duplicate image path has conflicting identity")
            raise DatasetValidationError("duplicate image path in reid_raw.json")
        paths[relative_path] = person_id

        identity_splits[person_id].add(split)
        samples.append(
            _PendingSample(
                split=split,
                relative_path=relative_path,
                image_path=image_path,
                person_id=person_id,
                captions=tuple(captions),
                file_signature=file_signature,
            )
        )
        split_presence[split] += 1

    if any(len(splits) != 1 for splits in identity_splits.values()):
        raise DatasetValidationError("identity appears in multiple splits")
    if any(split_presence[split] == 0 for split in _SPLITS):
        raise DatasetValidationError("train, val, and test splits must all contain images")
    if (
        len(identity_splits) < _MIN_TRAIN_IDENTITIES
        or len({sample.person_id for sample in samples if sample.split == "train"})
        < _MIN_TRAIN_IDENTITIES
    ):
        raise DatasetValidationError("training split needs at least two distinct identities")
    return tuple(samples)


def _cache_key(root: Path, annotation_bytes: bytes, samples: tuple[_PendingSample, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_PROTOCOL.encode("ascii"))
    digest.update(b"\0")
    digest.update(root.as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(annotation_bytes).digest())
    for sample in samples:
        digest.update(sample.relative_path.encode("utf-8"))
        digest.update(b"\0")
        for item in sample.file_signature:
            digest.update(item.to_bytes(16, "big", signed=False))
    return digest.hexdigest()


def _assert_sources_unchanged(
    annotation_path: Path,
    annotation_signature: tuple[int, int, int, int, int],
    imgs_root: Path,
    samples: tuple[_PendingSample, ...],
) -> None:
    """Recheck every original input before returning or caching a manifest."""
    try:
        current_annotation = _stat_signature(annotation_path)
    except OSError as exc:
        raise DatasetValidationError("dataset source changed during validation") from exc
    if current_annotation != annotation_signature:
        raise DatasetValidationError("reid_raw.json changed during validation")
    for sample in samples:
        try:
            _relative_path, image_path, signature = _safe_image_path(
                imgs_root,
                sample.relative_path,
            )
        except (DatasetValidationError, OSError) as exc:
            raise DatasetValidationError("dataset source changed during validation") from exc
        if image_path != sample.image_path or signature != sample.file_signature:
            raise DatasetValidationError("dataset image changed during validation")


def _hash_and_decode(sample: _PendingSample) -> tuple[str, int, int]:
    """Hash and fully decode one stable image without following a final symlink."""
    initial = _stat_signature(sample.image_path)
    if initial != sample.file_signature:
        raise DatasetValidationError("dataset image changed during validation")

    digest = hashlib.sha256()
    try:
        file_descriptor = os.open(
            sample.image_path,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        with os.fdopen(file_descriptor, "rb") as source:
            for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
            _ = source.seek(0)
            with Image.open(source) as image:
                _ = image.verify()
            _ = source.seek(0)
            with Image.open(source) as image:
                width, height = image.size
                _ = image.load()
    except Exception as exc:
        raise DatasetValidationError(
            f"cannot decode image: {sample.relative_path.rsplit('/', 1)[-1]}"
        ) from exc
    if width <= 0 or height <= 0:
        raise DatasetValidationError("image dimensions must be positive")
    if _stat_signature(sample.image_path) != initial:
        raise DatasetValidationError("dataset image changed during validation")
    return digest.hexdigest(), width, height


def _build_split_counts(samples: tuple[TrainingSample, ...]) -> Mapping[str, DatasetSplitCounts]:
    image_counts: Counter[DatasetSplit] = Counter()
    caption_counts: Counter[DatasetSplit] = Counter()
    identities: dict[DatasetSplit, set[int]] = {split: set() for split in _SPLITS}
    for sample in samples:
        image_counts[sample.split] += 1
        caption_counts[sample.split] += len(sample.captions)
        identities[sample.split].add(sample.person_id)
    counts = {
        split: DatasetSplitCounts(
            images=image_counts[split],
            captions=caption_counts[split],
            identities=len(identities[split]),
        )
        for split in _SPLITS
    }
    if any(
        value < 0
        for record in counts.values()
        for value in (record.images, record.captions, record.identities)
    ):
        raise DatasetValidationError("dataset split counts must be finite non-negative integers")
    return MappingProxyType(counts)


def _fingerprint(
    annotation_bytes: bytes,
    samples: tuple[TrainingSample, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(_DATASET_ID.encode("ascii"))
    digest.update(b"\0")
    digest.update(_PROTOCOL.encode("ascii"))
    digest.update(b"\0")
    digest.update(hashlib.sha256(annotation_bytes).digest())
    for sample in samples:
        digest.update(sample.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.image_sha256.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _read_dataset_source(root: Path) -> _DatasetSource:
    """Parse the fixed annotation and capture all referenced-file signatures."""
    requested_root = Path(root).expanduser()
    if requested_root.is_symlink():
        raise DatasetValidationError("configured dataset root must not be a symlink")
    try:
        dataset_root = requested_root.resolve(strict=True)
    except OSError as exc:
        raise DatasetValidationError("configured dataset root does not exist") from exc
    if not dataset_root.is_dir():
        raise DatasetValidationError("configured dataset root is not a directory")

    imgs_root = dataset_root / "imgs"
    annotation_path = dataset_root / "reid_raw.json"
    if imgs_root.is_symlink() or annotation_path.is_symlink():
        raise DatasetValidationError("dataset imgs and reid_raw.json must not be symlinks")
    if not imgs_root.is_dir():
        raise DatasetValidationError("dataset imgs directory is missing")
    imgs_root = imgs_root.resolve(strict=True)
    try:
        annotation_signature = _stat_signature(annotation_path)
        annotation_bytes = annotation_path.read_bytes()
        if _stat_signature(annotation_path) != annotation_signature:
            raise DatasetValidationError("reid_raw.json changed during validation")
    except OSError as exc:
        raise DatasetValidationError("reid_raw.json is missing or unreadable") from exc

    try:
        raw = cast(
            "object",
            json.loads(
                annotation_bytes.decode("utf-8"),
                parse_constant=_reject_non_finite_constant,
            ),
        )
    except DatasetValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError("reid_raw.json is not valid UTF-8 JSON") from exc

    pending_samples = _parse_annotations(raw, imgs_root)
    key = _cache_key(dataset_root, annotation_bytes, pending_samples)
    return _DatasetSource(
        root=dataset_root,
        imgs_root=imgs_root,
        annotation_path=annotation_path,
        annotation_signature=annotation_signature,
        annotation_bytes=annotation_bytes,
        samples=pending_samples,
        cache_key=key,
    )


def validate_cached_cuhk(root: Path) -> DatasetManifest | None:
    """Recheck every source signature and return only an unchanged cached manifest."""
    source = _read_dataset_source(root)
    cached = _MANIFEST_CACHE.get(source.cache_key)
    _assert_sources_unchanged(
        source.annotation_path,
        source.annotation_signature,
        source.imgs_root,
        source.samples,
    )
    if cached is None:
        return None
    _MANIFEST_CACHE.move_to_end(source.cache_key)
    return cached


def validate_cuhk(root: Path) -> DatasetManifest:
    """Validate the full base annotation and every referenced CUHK image."""
    source = _read_dataset_source(root)
    pending_samples = source.samples
    key = source.cache_key
    cached = _MANIFEST_CACHE.get(key)
    if cached is not None:
        _assert_sources_unchanged(
            source.annotation_path,
            source.annotation_signature,
            source.imgs_root,
            pending_samples,
        )
        _MANIFEST_CACHE.move_to_end(key)
        return cached

    samples = tuple(
        TrainingSample(
            split=pending.split,
            relative_path=pending.relative_path,
            image_path=pending.image_path,
            person_id=pending.person_id,
            captions=pending.captions,
            image_sha256=sha256,
            width=width,
            height=height,
        )
        for pending in pending_samples
        for sha256, width, height in (_hash_and_decode(pending),)
    )
    _assert_sources_unchanged(
        source.annotation_path,
        source.annotation_signature,
        source.imgs_root,
        pending_samples,
    )
    counts = _build_split_counts(samples)
    manifest = DatasetManifest(
        dataset_id=_DATASET_ID,
        fingerprint=_fingerprint(source.annotation_bytes, samples),
        protocol=_PROTOCOL,
        root=source.root,
        samples=samples,
        split_counts=counts,
    )
    _MANIFEST_CACHE[key] = manifest
    _MANIFEST_CACHE.move_to_end(key)
    while len(_MANIFEST_CACHE) > _CACHE_MAX_ENTRIES:
        _ = _MANIFEST_CACHE.popitem(last=False)
    return manifest


__all__ = [
    "DatasetManifest",
    "DatasetSplit",
    "DatasetSplitCounts",
    "DatasetValidationError",
    "TrainingSample",
    "validate_cached_cuhk",
    "validate_cuhk",
]
