from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, cast

import pytest

from gods_watching.retention.filesystem import safe_scan
from gods_watching.storage import CropObjectStore, physical_usage
from gods_watching.storage.physical_usage import PhysicalUsageError, physical_crop_bytes

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def test_physical_crop_bytes_counts_regular_jpeg_and_temp_files_at_any_depth(
    tmp_path: Path,
) -> None:
    root = tmp_path / "crops"
    store = CropObjectStore(root)
    crop = store.write(b"generated crop")
    root_jpeg = root / "existing.jpg"
    _ = root_jpeg.write_bytes(b"root level")
    nested = root / "unmanaged" / "deeper"
    nested.mkdir(parents=True)
    temporary = nested / "partial.tmp"
    _ = temporary.write_bytes(b"partial")
    text_file = nested / "notes.txt"
    _ = text_file.write_bytes(b"ignore")

    outside = tmp_path / "outside.jpg"
    _ = outside.write_bytes(b"must not follow")
    linked_file = root / "unmanaged" / "outside.jpg"
    linked_file.symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_nested = outside_dir / "deep"
    outside_nested.mkdir(parents=True)
    _ = (outside_nested / "linked.tmp").write_bytes(b"must not follow")
    linked_dir = root / "linked-dir"
    linked_dir.symlink_to(outside_dir, target_is_directory=True)

    assert physical_crop_bytes(root) == crop.byte_size + len(b"root level") + len(b"partial")
    assert outside.read_bytes() == b"must not follow"
    assert linked_file.is_symlink()
    assert linked_dir.is_symlink()


def test_physical_crop_bytes_uses_no_shell_find_with_a_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "crops"
    root.mkdir()
    captured_command: list[str] = []
    captured_options: dict[str, object] = {}

    def capture_run(
        command: list[str],
        **options: object,
    ) -> subprocess.CompletedProcess[bytes]:
        captured_command.extend(command)
        captured_options.update(options)
        return subprocess.CompletedProcess(command, 0, b"17\0", b"")

    monkeypatch.setattr("gods_watching.storage.physical_usage.subprocess.run", capture_run)

    assert physical_crop_bytes(root) == 17
    assert captured_command
    assert captured_command[0].endswith("/find")
    assert captured_command[1] == "-P"
    assert "-type" in captured_command
    assert "-name" in captured_command
    assert "*.jpg" in captured_command
    assert "*.tmp" in captured_command
    assert captured_options["shell"] is False
    assert captured_options["timeout"] == 10.0


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "malformed"])
def test_physical_crop_bytes_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    root = tmp_path / "crops"
    root.mkdir()

    def failed_run(
        command: list[str],
        **_options: object,
    ) -> subprocess.CompletedProcess[bytes]:
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 10)
        if failure == "nonzero":
            return subprocess.CompletedProcess(command, 1, b"17\0", b"find failed")
        return subprocess.CompletedProcess(command, 0, b"17", b"")

    monkeypatch.setattr("gods_watching.storage.physical_usage.subprocess.run", failed_run)

    with pytest.raises(PhysicalUsageError):
        _ = physical_crop_bytes(root)


def test_physical_crop_bytes_rejects_symlinked_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="crop root"):
        _ = physical_crop_bytes(linked_root)


def test_managed_crop_bytes_matches_generated_safe_scan_grammar(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    store = CropObjectStore(root)
    crop = store.write(b"generated jpeg")
    temporary_key = "aa/bb/.00000000-0000-4000-8000-000000000001.tmp"
    temporary = root / temporary_key
    temporary.parent.mkdir(parents=True, exist_ok=True)
    _ = temporary.write_bytes(b"partial")
    _ = (root / "existing.jpg").write_bytes(b"root level")
    unmanaged = root / "aa" / "bb" / "unmanaged.jpg"
    _ = unmanaged.write_bytes(b"unmanaged suffix")
    invalid_prefix = root / "GG" / "bb" / "00000000-0000-4000-8000-000000000002.jpg"
    invalid_prefix.parent.mkdir(parents=True)
    _ = invalid_prefix.write_bytes(b"invalid prefix")
    outside = tmp_path / "outside.jpg"
    _ = outside.write_bytes(b"linked file")
    linked_file = root / "aa" / "bb" / "00000000-0000-4000-8000-000000000003.jpg"
    linked_file.symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_file = outside_dir / "bb" / "00000000-0000-4000-8000-000000000004.jpg"
    outside_file.parent.mkdir(parents=True)
    _ = outside_file.write_bytes(b"linked directory file")
    linked_dir = root / "cc"
    linked_dir.symlink_to(outside_dir, target_is_directory=True)

    managed_crop_bytes = vars(physical_usage).get("managed_crop_bytes")
    assert callable(managed_crop_bytes)
    managed_crop_bytes = cast("Callable[[Path], int]", managed_crop_bytes)

    scanned = safe_scan(root)
    assert {item.object_key for item in scanned} == {crop.object_key, temporary_key}
    assert managed_crop_bytes(root) == sum(item.byte_size for item in scanned)
    assert linked_file.is_symlink()
    assert linked_dir.is_symlink()
