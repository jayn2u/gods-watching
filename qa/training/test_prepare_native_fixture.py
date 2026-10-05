# ruff: noqa: INP001, D100, D103, S101, PLR2004

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import cast

import pytest
from PIL import Image
from qa.training.prepare_native_fixture import (
    FixturePreparationError,
    prepare_native_fixture,
)

_ORIGINAL_FINGERPRINT = "e62741f90efc9edde37fde3f3ef5c3ed7a774bdd3d7a7fc7c71d6475ed7fd333"
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753de"
) + bytes.fromhex("0000000c49444154789c63606060000000040001f61738550000000049454e44ae426082")


def _source_dataset(
    root: Path, *, duplicate_train_identity: bool = False
) -> list[dict[str, object]]:
    imgs = root / "imgs"
    imgs.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    identity = 1
    if duplicate_train_identity:
        rows.extend(
            [
                {
                    "split": "train",
                    "id": identity,
                    "file_path": "train/duplicate-a.png",
                    "captions": ["train person one"],
                },
                {
                    "split": "train",
                    "id": identity,
                    "file_path": "train/duplicate-b.png",
                    "captions": ["train person one alternate"],
                },
            ]
        )
        identity += 1
    train_identities = 1024
    for index in range(train_identities):
        person_id = identity + index
        rows.append(
            {
                "split": "train",
                "id": person_id,
                "file_path": f"train/person-{person_id}.png",
                "captions": [f"train person {person_id}"],
                "source_row": index,
            }
        )
    for split, starting_id in (("val", 10_000), ("test", 20_000)):
        for index in range(16):
            person_id = starting_id + index
            rows.append(
                {
                    "split": split,
                    "id": person_id,
                    "file_path": f"{split}/person-{person_id}.png",
                    "captions": [f"{split} person {person_id}"],
                    "source_row": index,
                }
            )

    for row in rows:
        image_path = imgs / str(row["file_path"])
        image_path.parent.mkdir(parents=True, exist_ok=True)
        _ = image_path.write_bytes(_PNG)
    _ = (root / "reid_raw.json").write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _json_value(path: Path) -> object:
    return cast("object", json.loads(path.read_text(encoding="utf-8")))


def _record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast("dict[str, object]", value)


def _rows(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [_record(row) for row in cast("list[object]", value)]


def test_prepare_fixture_preserves_native_rows_and_exact_image_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source_rows = _source_dataset(source, duplicate_train_identity=True)
    source_annotations = (source / "reid_raw.json").read_bytes()
    output = tmp_path / "runtime-data"

    fixture_root = prepare_native_fixture(
        source_root=source,
        output_root=output,
        original_fingerprint=_ORIGINAL_FINGERPRINT,
    )

    fixture_rows = _rows(_json_value(fixture_root / "reid_raw.json"))
    manifest = _record(_json_value(fixture_root / "manifest.json"))
    fixture_record = _record(manifest["fixture"])
    image_hashes = _record(manifest["image_sha256"])
    assert fixture_root.stat().st_mode & 0o777 == 0o755
    assert (fixture_root / "reid_raw.json").stat().st_mode & 0o777 == 0o644
    assert len(fixture_rows) == 1056
    assert manifest["original_fingerprint"] == _ORIGINAL_FINGERPRINT
    assert manifest["fixture_fingerprint"] == fixture_record["fingerprint"]
    assert fixture_record["split_counts"] == {
        "train": {"images": 1024, "captions": 1024, "identities": 1024},
        "val": {"images": 16, "captions": 16, "identities": 16},
        "test": {"images": 16, "captions": 16, "identities": 16},
    }
    assert len({row["id"] for row in fixture_rows if row["split"] == "train"}) == 1024
    assert {row["id"] for row in fixture_rows if row["split"] == "train"}.isdisjoint(
        row["id"] for row in fixture_rows if row["split"] != "train"
    )
    assert fixture_rows == [
        row
        for row in source_rows
        if row["file_path"] not in {"train/duplicate-b.png", "train/person-1025.png"}
    ]
    for row in fixture_rows:
        relative_path = Path("imgs") / str(row["file_path"])
        source_bytes = (source / relative_path).read_bytes()
        fixture_bytes = (fixture_root / relative_path).read_bytes()
        assert fixture_bytes == source_bytes
        assert sha256(fixture_bytes).hexdigest() == image_hashes[str(row["file_path"])]
    assert (source / "reid_raw.json").read_bytes() == source_annotations
    with Image.open(fixture_root / "imgs" / "train" / "person-2.png") as image:
        assert image.size == (1, 1)


def test_prepare_fixture_rejects_insufficient_distinct_train_identities(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    rows = _source_dataset(source)
    for row in rows:
        if row["split"] == "train":
            row["id"] = 1
    _ = (source / "reid_raw.json").write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "runtime-data"

    with pytest.raises(FixturePreparationError, match="distinct train identities"):
        _ = prepare_native_fixture(
            source_root=source,
            output_root=output,
            original_fingerprint=_ORIGINAL_FINGERPRINT,
        )

    assert not (output / "cuhk-pedes-native-smoke").exists()
