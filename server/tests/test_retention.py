from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from pathlib import Path


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
