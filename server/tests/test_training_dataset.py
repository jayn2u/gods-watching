# pyright: reportPrivateUsage=false

import json
import os
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

import gods_watching.training.dataset as training_dataset
from gods_watching.training.dataset import DatasetValidationError, validate_cuhk
from gods_watching.training.repository import _json_value
from gods_watching.training.settings import TrainingSettings


def _write_image(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (3, 2), color).save(path, format="PNG")


def _record(
    split: str,
    person_id: int,
    name: str,
    captions: tuple[str, ...] = ("first description", "second description"),
) -> dict[str, object]:
    return {
        "split": split,
        "captions": list(captions),
        "file_path": f"{split}/{name}.png",
        "id": person_id,
    }


def _dataset_root(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    root = tmp_path / "CUHK-PEDES"
    (root / "imgs").mkdir(parents=True)
    for record in records:
        file_path = record["file_path"]
        assert isinstance(file_path, str)
        image_path = root / "imgs" / file_path
        if "missing" not in image_path.name:
            _write_image(image_path)
    _ = (root / "reid_raw.json").write_text(json.dumps(records), encoding="utf-8")
    return root


def _valid_records() -> list[dict[str, object]]:
    return [
        _record("train", 1, "person-1-a"),
        _record("train", 2, "person-2-a"),
        _record("val", 3, "person-3-a"),
        _record("test", 4, "person-4-a"),
    ]


def test_dataset_rejects_escape_symlink_missing_image(tmp_path: Path) -> None:
    for index, (file_path, create_symlink) in enumerate(
        (
            ("../outside.png", False),
            ("train/link.png", True),
            ("train/missing.png", False),
        )
    ):
        records = _valid_records()
        records[0]["file_path"] = file_path
        root = _dataset_root(tmp_path / str(index), records)
        if create_symlink:
            outside = tmp_path / "outside.png"
            _write_image(outside)
            link = root / "imgs" / file_path
            link.parent.mkdir(parents=True, exist_ok=True)
            link.unlink()
            link.symlink_to(outside)

        with pytest.raises(DatasetValidationError):
            _ = validate_cuhk(root)


def test_identity_overlap_rejected(tmp_path: Path) -> None:
    records = _valid_records()
    records[2]["id"] = records[0]["id"]
    root = _dataset_root(tmp_path, records)

    with pytest.raises(DatasetValidationError, match="identity appears in multiple splits"):
        _ = validate_cuhk(root)


def test_duplicate_relative_path_with_conflicting_identity_is_rejected(
    tmp_path: Path,
) -> None:
    records = _valid_records()
    records[1]["file_path"] = records[0]["file_path"]
    root = _dataset_root(tmp_path, records)

    with pytest.raises(DatasetValidationError, match="duplicate image path"):
        _ = validate_cuhk(root)


def test_dataset_rejects_non_image_and_fifo_inputs(tmp_path: Path) -> None:
    records = _valid_records()
    root = _dataset_root(tmp_path, records)
    image_path = root / "imgs" / "train/person-1-a.png"
    _ = image_path.write_bytes(b"not an image")
    with pytest.raises(DatasetValidationError, match="cannot decode image"):
        _ = validate_cuhk(root)

    image_path.unlink()
    os.mkfifo(image_path)
    with pytest.raises(DatasetValidationError, match="not a regular file"):
        _ = validate_cuhk(root)


def test_dataset_rejects_empty_caption(tmp_path: Path) -> None:
    records = _valid_records()
    records[0]["captions"] = ["  "]
    root = _dataset_root(tmp_path, records)

    with pytest.raises(DatasetValidationError, match="caption must be non-empty"):
        _ = validate_cuhk(root)


def test_fingerprint_changes_with_image_bytes(tmp_path: Path) -> None:
    root = _dataset_root(tmp_path, _valid_records())
    first = validate_cuhk(root)
    image_path = root / "imgs" / "train/person-1-a.png"
    _write_image(image_path, color=(220, 30, 50))

    second = validate_cuhk(root)

    assert first.fingerprint != second.fingerprint


def test_actual_caption_count_is_reported_and_augmented_file_is_ignored(
    tmp_path: Path,
) -> None:
    records = _valid_records()
    records[1]["captions"] = ["a", "b", "c", "d"]
    root = _dataset_root(tmp_path, records)
    _ = (root / "reid_raw_diverse_color.json").write_text(
        "not the base protocol",
        encoding="utf-8",
    )

    manifest = validate_cuhk(root)

    assert manifest.caption_count == 10
    assert manifest.split_counts["train"].captions == 6
    assert manifest.split_counts["train"].images == 2


def test_dataset_public_snapshot_excludes_paths_and_captions(tmp_path: Path) -> None:
    manifest = validate_cuhk(_dataset_root(tmp_path, _valid_records()))

    snapshot = manifest.public_snapshot()
    durable_snapshot = _json_value(manifest)

    assert snapshot["dataset_id"] == "cuhk-pedes"
    assert "fingerprint" in snapshot
    assert "samples" not in snapshot
    assert "root" not in snapshot
    assert "first description" not in repr(snapshot)
    assert "person-1-a.png" not in repr(snapshot)
    assert durable_snapshot == snapshot
    assert "first description" not in repr(durable_snapshot)


def test_validation_rejects_an_image_changed_during_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset_root(tmp_path, _valid_records())
    image_path = root / "imgs" / "train/person-1-a.png"
    original_decode = training_dataset._hash_and_decode  # noqa: SLF001
    decode_count = 0

    def change_earlier_image(sample: training_dataset._PendingSample) -> tuple[str, int, int]:
        nonlocal decode_count
        decode_count += 1
        if decode_count == 2:
            _write_image(image_path, color=(200, 40, 90))
        return original_decode(sample)

    monkeypatch.setattr(training_dataset, "_hash_and_decode", change_earlier_image)

    with pytest.raises(DatasetValidationError, match="changed during validation"):
        _ = validate_cuhk(root)


def test_cached_manifest_is_not_returned_after_a_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _dataset_root(tmp_path, _valid_records())
    _ = validate_cuhk(root)
    image_path = root / "imgs" / "train/person-1-a.png"
    original_cache_key = training_dataset._cache_key  # noqa: SLF001

    def change_after_signature(
        cache_root: Path,
        annotation_bytes: bytes,
        pending_samples: tuple[training_dataset._PendingSample, ...],
    ) -> str:
        key = original_cache_key(cache_root, annotation_bytes, pending_samples)
        _write_image(image_path, color=(200, 40, 90))
        return key

    monkeypatch.setattr(training_dataset, "_cache_key", change_after_signature)

    with pytest.raises(DatasetValidationError, match="changed during validation"):
        _ = validate_cuhk(root)


def test_training_settings_use_only_an_absolute_environment_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GW_TRAINING_DATASET_ROOT", str(tmp_path))

    settings = TrainingSettings()

    assert settings.dataset_root == tmp_path
    with pytest.raises(ValidationError):
        _ = TrainingSettings(dataset_root=Path("relative/dataset"))
