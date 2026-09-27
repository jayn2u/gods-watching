from __future__ import annotations

from pathlib import Path

import pytest

from gods_watching.retention.filesystem import (
    ManagedFileKind,
    safe_scan,
    safe_unlink,
)
from gods_watching.retention.models import (
    DEFAULT_QUOTA_BYTES,
    DEFAULT_RETENTION_DAYS,
    RetentionSettings,
    StorageAccounting,
)
from gods_watching.storage import CropObjectStore


def test_settings_clamp_malformed_positive_values_to_safe_minimums() -> None:
    settings = RetentionSettings.from_values(retention_days=0, quota_bytes=-5)

    assert settings.retention_days == 1
    assert settings.quota_bytes == 1


def test_settings_default_to_scope_values_when_missing() -> None:
    settings = RetentionSettings.from_values(retention_days=None, quota_bytes=None)

    assert settings.retention_days == DEFAULT_RETENTION_DAYS
    assert settings.quota_bytes == DEFAULT_QUOTA_BYTES


def test_managed_bytes_include_pending_gc_and_relation_bloat() -> None:
    accounting = StorageAccounting(
        physical_crop_bytes=80,
        pending_gc_bytes=20,
        relation_bytes=900,
        filesystem_free_bytes=10_000,
        quota_bytes=1_000,
    )

    assert accounting.managed_bytes == 980
    assert accounting.cleanup_required


def test_safe_scan_and_unlink_leave_symlinked_outside_data_untouched(tmp_path: Path) -> None:
    root = tmp_path / "crops"
    store = CropObjectStore(root)
    crop = store.write(b"jpeg")
    temporary = root / "aa" / "bb" / ".00000000-0000-4000-8000-000000000001.tmp"
    temporary.parent.mkdir(parents=True, exist_ok=True)
    _ = temporary.write_bytes(b"partial")
    outside = tmp_path / "outside.jpg"
    _ = outside.write_bytes(b"keep")
    symlink = root / "aa" / "bb" / "00000000-0000-4000-8000-000000000099.jpg"
    symlink.symlink_to(outside)

    files = safe_scan(root)
    for item in files:
        if item.kind is ManagedFileKind.TEMPORARY:
            _ = safe_unlink(root, item.object_key)
    _ = safe_unlink(root, crop.object_key)

    assert not (root / crop.object_key).exists()
    assert not temporary.exists()
    assert outside.read_bytes() == b"keep"
    assert symlink.is_symlink()


def test_safe_scan_rejects_a_symlinked_crop_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="crop root"):
        _ = safe_scan(linked_root)


def test_safe_scan_ignores_unmanaged_paths_without_per_file_path_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "crops"
    store = CropObjectStore(root)
    crop = store.write(b"jpeg")
    temporary_key = "aa/bb/.00000000-0000-4000-8000-000000000001.tmp"
    temporary = root / temporary_key
    temporary.parent.mkdir(parents=True, exist_ok=True)
    _ = temporary.write_bytes(b"partial")

    invalid_name = root / "aa" / "bb" / "not-a-generated-object.txt"
    _ = invalid_name.write_bytes(b"ignore")
    invalid_prefix = root / "GG" / "bb" / "00000000-0000-4000-8000-000000000002.jpg"
    invalid_prefix.parent.mkdir(parents=True)
    _ = invalid_prefix.write_bytes(b"ignore")

    outside = tmp_path / "outside.jpg"
    _ = outside.write_bytes(b"keep")
    linked_file = root / "aa" / "bb" / "00000000-0000-4000-8000-000000000003.jpg"
    linked_file.symlink_to(outside)
    outside_dir = tmp_path / "outside-dir"
    outside_crop = outside_dir / "dd" / "00000000-0000-4000-8000-000000000004.jpg"
    outside_crop.parent.mkdir(parents=True)
    _ = outside_crop.write_bytes(b"keep")
    linked_dir = root / "cc"
    linked_dir.symlink_to(outside_dir, target_is_directory=True)

    relative_to_calls = 0
    original_relative_to = Path.relative_to

    def count_relative_to(path: Path, other: str | Path) -> Path:
        nonlocal relative_to_calls
        relative_to_calls += 1
        return original_relative_to(path, other)

    monkeypatch.setattr(Path, "relative_to", count_relative_to)
    files = safe_scan(root)

    assert {
        (item.object_key, item.kind, item.byte_size)
        for item in files
    } == {
        (crop.object_key, ManagedFileKind.JPEG, crop.byte_size),
        (temporary_key, ManagedFileKind.TEMPORARY, len(b"partial")),
    }
    assert relative_to_calls == 0
    assert outside.read_bytes() == b"keep"
    assert linked_file.is_symlink()
    assert linked_dir.is_symlink()
