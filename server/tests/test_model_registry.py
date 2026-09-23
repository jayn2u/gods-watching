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
)


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
