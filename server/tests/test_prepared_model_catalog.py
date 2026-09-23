import hashlib
import json
import os
from pathlib import Path

from gods_watching.model_selection import (
    IDENTITY_MARKER_NAME,
    PreparedModelCatalog,
    PreparedModelStatus,
)
from gods_watching.model_selection.registry import ClipModelPackage, ClipModelRegistry
from gods_watching.setup.models import load_models_lock


def _write_fixture_package(
    tmp_path: Path,
) -> tuple[ClipModelRegistry, Path, Path, ClipModelPackage]:
    assets_root = tmp_path / "models"
    snapshot = assets_root / "clip"
    payload = b"immutable checkpoint fixture"
    file_path = snapshot / "weights.bin"
    file_path.parent.mkdir(parents=True)
    _ = file_path.write_bytes(payload)
    package = ClipModelPackage(
        model_id="openai/clip-vit-base-patch16",
        revision="revision-1",
        snapshot_path=Path("/models/clip"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    registry = ClipModelRegistry((package,))
    lock_path = tmp_path / "models.lock.json"
    _ = lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:" + "0" * 64,
                    "built_image": "test:image",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": package.model_id,
                        "revision": package.revision,
                        "license": "MIT",
                        "source": "https://example.invalid",
                        "files": [
                            {
                                "path": "clip/weights.bin",
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "size": len(payload),
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "model_id": package.model_id,
        "revision": package.revision,
        "dimension": package.dimension,
        "processor": package.processor,
        "runtime": package.runtime,
    }
    _ = (snapshot / IDENTITY_MARKER_NAME).write_text(json.dumps(marker), encoding="utf-8")
    return registry, lock_path, assets_root, package


def test_catalog_reports_prepared_package_and_reuses_unchanged_digest(tmp_path: Path) -> None:
    registry, lock_path, assets_root, package = _write_fixture_package(tmp_path)
    catalog = PreparedModelCatalog(registry, lock_path, assets_root)

    first = catalog.status(package)
    second = catalog.status(package)

    assert first == PreparedModelStatus(prepared=True)
    assert second == first


def test_catalog_rejects_corrupt_or_partial_package_without_marker(tmp_path: Path) -> None:
    registry, lock_path, assets_root, package = _write_fixture_package(tmp_path)
    target = assets_root / "clip/weights.bin"
    _ = target.write_bytes(b"partial")
    catalog = PreparedModelCatalog(registry, lock_path, assets_root)

    assert catalog.status(package) == PreparedModelStatus(prepared=False, reason="asset_corrupt")

    _ = target.write_bytes(b"immutable checkpoint fixture")
    (assets_root / "clip" / IDENTITY_MARKER_NAME).unlink()
    assert catalog.status(package) == PreparedModelStatus(
        prepared=False, reason="identity_marker_missing"
    )


def test_catalog_detects_registry_and_lock_revision_mismatch(tmp_path: Path) -> None:
    registry, lock_path, assets_root, package = _write_fixture_package(tmp_path)
    lock = load_models_lock(lock_path)
    record = lock.models[0]
    _ = lock_path.write_text(
        json.dumps(
            {
                **lock.model_dump(mode="json"),
                "models": [
                    {
                        **record.model_dump(mode="json"),
                        "revision": "different-revision",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    catalog = PreparedModelCatalog(registry, lock_path, assets_root)

    assert catalog.status(package) == PreparedModelStatus(prepared=False, reason="lock_mismatch")


def test_catalog_cache_invalidates_on_file_metadata_change(tmp_path: Path) -> None:
    registry, lock_path, assets_root, package = _write_fixture_package(tmp_path)
    catalog = PreparedModelCatalog(registry, lock_path, assets_root)
    assert catalog.status(package).prepared

    target = assets_root / "clip/weights.bin"
    _ = target.write_bytes(b"bad")
    _ = os.utime(target, None)
    assert catalog.status(package) == PreparedModelStatus(prepared=False, reason="asset_corrupt")
