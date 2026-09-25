import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from gods_watching.model_selection.registry import (
    B16_REVISION,
    B32_REVISION,
    DEFAULT_CLIP_MODEL_ID,
    L14_REVISION,
    ClipModelPackage,
    ClipModelRegistry,
    UnknownClipModelError,
    load_clip_registry,
)


def _installed_package(root: Path, model_id: str, content: bytes) -> Path:
    """Publish a minimal content-addressed fixture for catalog discovery."""
    item = {
        "path": "model.safetensors",
        "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    files = [item]
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    directory = root / "imported" / digest
    directory.mkdir(parents=True)
    (directory / item["path"]).write_bytes(content)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "revision": digest,
                "display_name": model_id,
                "base_model_id": DEFAULT_CLIP_MODEL_ID,
                "dimension": 512,
                "files": files,
                "package_sha256": digest,
                "cuhk_report": "report.json",
            }
        )
    )
    return directory


def test_imported_catalog_is_deterministic_and_checks_bytes(tmp_path: Path) -> None:
    first = _installed_package(tmp_path, "local/first", b"first")
    second = _installed_package(tmp_path, "local/second", b"second")
    initial = load_clip_registry(tmp_path)
    restarted = load_clip_registry(tmp_path)
    assert initial.packages == restarted.packages
    assert initial.require("local/first").revision != initial.require("local/second").revision
    assert initial.require("local/first").snapshot_path == Path("/models/imported") / first.name
    (second / "model.safetensors").write_bytes(b"broken")
    with pytest.raises(ValueError, match="invalid_installed_package"):
        load_clip_registry(tmp_path)


def test_duplicate_imported_model_id_is_rejected(tmp_path: Path) -> None:
    _installed_package(tmp_path, "local/shared", b"first")
    _installed_package(tmp_path, "local/shared", b"second")
    with pytest.raises(ValueError, match="duplicate_clip_model"):
        load_clip_registry(tmp_path)


def test_default_registry_contains_the_three_immutable_clip_packages() -> None:
    registry = ClipModelRegistry()

    assert registry.default.model_id == DEFAULT_CLIP_MODEL_ID
    assert registry.default.revision == B16_REVISION
    assert registry.default.snapshot_path == Path("/models/clip")
    assert registry.default.dimension == 512
    assert registry.default.processor == "CLIPProcessor"
    assert registry.default.runtime == "transformers"
    assert tuple(package.model_id for package in registry.packages) == (
        "openai/clip-vit-base-patch16",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-large-patch14",
    )
    assert registry.require("openai/clip-vit-base-patch32").revision == B32_REVISION
    assert registry.require("openai/clip-vit-large-patch14").dimension == 768
    assert registry.require("openai/clip-vit-large-patch14").revision == L14_REVISION


def test_registry_rejects_unknown_models_and_package_mutation() -> None:
    registry = ClipModelRegistry()

    with pytest.raises(UnknownClipModelError, match="unknown_clip_model"):
        _ = registry.require("example.invalid/unknown")
    with pytest.raises(FrozenInstanceError):
        registry.default.revision = "changed"  # pyright: ignore[reportAttributeAccessIssue]


def test_registry_rejects_duplicate_model_ids() -> None:
    package = ClipModelPackage(
        model_id="example.invalid/clip",
        revision="revision",
        snapshot_path=Path("/models/example"),
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )

    with pytest.raises(ValueError, match="duplicate_clip_model"):
        _ = ClipModelRegistry((package, package))
