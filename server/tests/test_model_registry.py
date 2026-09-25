import hashlib
import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import anyio
import pytest

from gods_watching.api import production
from gods_watching.model_selection.importer import import_clip_package
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
from gods_watching.pipeline_worker import app as worker_app
from test_clip_backends import REPOSITORY_ROOT, _load_backend
from test_model_package_import import package as source_package


def test_api_and_worker_compose_same_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _installed_package(tmp_path, "local/shared", b"s")
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("ok")
    seen: list[tuple[ClipModelPackage, ...]] = []

    class StopCompositionError(Exception):
        pass

    def capture(root: Path) -> ClipModelRegistry:
        registry = load_clip_registry(root)
        seen.append(registry.packages)
        raise StopCompositionError

    class DatabaseStub:
        @staticmethod
        def connect(_url: object) -> object:
            return object()

    for module in (production, worker_app):
        monkeypatch.setattr(module, "Database", DatabaseStub)
        monkeypatch.setattr(module, "StorageRepository", lambda *_args: object())
        monkeypatch.setattr(module, "CredentialCipher", lambda *_args: object())
        monkeypatch.setattr(module, "load_clip_registry", capture)
    monkeypatch.setattr(production, "TritonClipTransport", lambda *_args: object())
    monkeypatch.setattr(worker_app, "CropObjectStore", lambda *_args: object())
    monkeypatch.setattr(worker_app, "TritonGrpcDetectorTransport", lambda **_kwargs: object())
    monkeypatch.setattr(worker_app, "DetectorClient", lambda **_kwargs: object())

    api_settings = production._ProductionSettings.model_construct(  # noqa: SLF001
        web_root=web,
        model_assets_root=tmp_path,
        camera_cipher_key="secret",
        database_url="sqlite://",
        triton_grpc_url="localhost:8001",
    )
    with pytest.raises(StopCompositionError):
        anyio.run(production._build_production_app, api_settings)  # noqa: SLF001
    worker_settings = worker_app.PipelineWorkerSettings.model_construct(
        model_assets_root=tmp_path,
        camera_cipher_key="secret",
        database_url="sqlite://",
        triton_grpc_url="localhost:8001",
        crops_root=tmp_path,
    )

    async def run_worker() -> None:
        await worker_app.run_pipeline_worker(
            worker_settings, lock_path=tmp_path / "lock", stop_event=anyio.Event()
        )

    with pytest.raises(StopCompositionError):
        anyio.run(run_worker)
    assert len(seen) == 2
    assert seen[0] == seen[1]


def _installed_package(root: Path, model_id: str, content: bytes) -> Path:
    """Publish a structurally valid package through the actual importer."""
    scratch = root / content.hex()
    scratch.mkdir(parents=True)
    source = source_package(scratch)
    weight = source / "model.safetensors"
    payload = weight.read_bytes()
    weight.write_bytes(payload[:-1] + content[:1])
    metadata = json.loads((source / "package.json").read_text())
    metadata["model_id"] = model_id
    for item in metadata["files"]:
        if item["path"] == "model.safetensors":
            item["sha256"] = hashlib.sha256(weight.read_bytes()).hexdigest()
    (source / "package.json").write_text(json.dumps(metadata))
    manifest = import_clip_package(source, root)
    return root / "imported" / manifest.package_sha256


def test_manual_hash_valid_but_structurally_invalid_install_is_rejected(tmp_path: Path) -> None:
    directory = _installed_package(tmp_path, "local/valid", b"v")
    (directory / "cuhk-report.json").unlink()
    with pytest.raises(ValueError, match="invalid_installed_package"):
        load_clip_registry(tmp_path)


@pytest.mark.parametrize("modality", ["image", "text"])
@pytest.mark.parametrize("diagnostic", ["missing_keys", "unexpected_keys", "mismatched_keys"])
def test_imported_backend_rejects_partial_checkpoint(
    modality: str, diagnostic: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = _load_backend(
        REPOSITORY_ROOT / f"inference/models/clip_{modality}/1/model.py", monkeypatch
    )

    class FakeClip:
        @staticmethod
        def from_pretrained(
            *_args: object, **kwargs: object
        ) -> tuple[object, dict[str, list[str]]]:
            assert kwargs["trust_remote_code"] is False
            assert kwargs["output_loading_info"] is True
            return object(), {diagnostic: ["projection.weight"]}

    monkeypatch.setattr(backend, "CLIPModel", FakeClip)
    settings = backend.RuntimeSettings(
        model_id="local/clip",
        snapshot_path=Path("/models/imported/hash"),
        revision="hash",
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )
    with pytest.raises(RuntimeError, match="clip_checkpoint_incomplete"):
        backend._load_model(settings)  # noqa: SLF001


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
    other = tmp_path / "other"
    second = _installed_package(other, "local/shared", b"second")
    shutil.copytree(second, tmp_path / "imported" / second.name)
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
