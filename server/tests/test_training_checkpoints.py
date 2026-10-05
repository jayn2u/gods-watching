from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from gods_watching.training.checkpoints import (
    CheckpointIdentityError,
    CheckpointSpaceError,
    load_checkpoint_verified,
    save_checkpoint_atomic,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import BinaryIO

_SNAPSHOT = {
    "config_snapshot": {"epochs": 4, "micro_batch_size": 2},
    "dataset_fingerprint": "a" * 64,
    "source_fingerprint": "b" * 64,
}


def _write_json(state: Mapping[str, object], destination: BinaryIO) -> None:
    _ = destination.write(json.dumps(state, sort_keys=True).encode("utf-8"))


def _read_json(path: Path) -> object:
    return cast("object", json.loads(path.read_text(encoding="utf-8")))


def _state() -> dict[str, object]:
    return {
        "identity": _SNAPSHOT,
        "model": {"weight": [1.0, 2.0]},
        "progress": {"epoch": 1, "optimizer_step": 3},
    }


def test_checkpoint_round_trip_is_atomic_and_snapshot_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"

    save_checkpoint_atomic(_state(), checkpoint, serializer=_write_json)
    loaded = load_checkpoint_verified(checkpoint, _SNAPSHOT, loader=_read_json)

    assert loaded["identity"] == _SNAPSHOT
    assert loaded["progress"] == {"epoch": 1, "optimizer_step": 3}
    assert not list(tmp_path.glob("*.partial"))


def test_checkpoint_refuses_changed_config_dataset_or_source(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    save_checkpoint_atomic(_state(), checkpoint, serializer=_write_json)

    changed = dict(_SNAPSHOT)
    changed["dataset_fingerprint"] = "c" * 64
    with pytest.raises(CheckpointIdentityError):
        _ = load_checkpoint_verified(checkpoint, changed, loader=_read_json)


def test_interrupted_save_keeps_previous_complete_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "last.pt"
    save_checkpoint_atomic(_state(), checkpoint, serializer=_write_json)
    previous = checkpoint.read_bytes()

    def interrupted(_state: Mapping[str, object], destination: BinaryIO) -> None:
        _ = destination.write(b"partial checkpoint")
        message = "simulated interrupted write"
        raise OSError(message)

    with pytest.raises(OSError, match="simulated interrupted write"):
        save_checkpoint_atomic(_state(), checkpoint, serializer=interrupted)

    assert checkpoint.read_bytes() == previous
    assert not list(tmp_path.glob("*.partial"))


def test_checkpoint_checks_temporary_space_before_serializing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "last.pt"
    save_checkpoint_atomic(_state(), checkpoint, serializer=_write_json)
    previous = checkpoint.read_bytes()
    def _one_byte_disk_usage(_path: Path) -> object:
        return type("Usage", (), {"free": 1})()

    monkeypatch.setattr(
        "gods_watching.training.checkpoints.shutil.disk_usage",
        _one_byte_disk_usage,
    )

    with pytest.raises(CheckpointSpaceError):
        save_checkpoint_atomic(_state(), checkpoint, serializer=_write_json)

    assert checkpoint.read_bytes() == previous
