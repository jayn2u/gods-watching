import hashlib
import json
import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import TYPE_CHECKING, cast

import anyio
import pytest

from gods_watching.api import production
from gods_watching.model_selection import registry as registry_module
from gods_watching.model_selection.assets import PreparedModelStatus
from gods_watching.model_selection.importer import import_clip_package
from gods_watching.model_selection.models import ModelNotPreparedError
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
from gods_watching.model_selection.service import ClipImageEmbeddingPort, ModelSelectionService
from gods_watching.pipeline_worker import app as worker_app
from gods_watching.pipeline_worker.settings import PipelineWorkerSettings
from test_clip_backends import REPOSITORY_ROOT, _load_backend
from test_model_package_import import package as source_package


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.model_selection.coordinator import TransitionCoordinator
    from gods_watching.model_selection.repository import TransitionRepository
    from gods_watching.model_selection.service import (
        ClipFactoryPort,
        ClipRuntimePort,
        PipelineLifecyclePort,
    )
    from gods_watching.storage import CropObjectStore, Database


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

    def object_factory(*_args: object, **_kwargs: object) -> object:
        return object()

    for module in (production, worker_app):
        monkeypatch.setattr(module, "Database", DatabaseStub)
        monkeypatch.setattr(module, "StorageRepository", object_factory)
        monkeypatch.setattr(module, "CredentialCipher", object_factory)
        monkeypatch.setattr(module, "load_clip_registry", capture)
    monkeypatch.setattr(production, "TritonClipTransport", object_factory)
    monkeypatch.setattr(worker_app, "CropObjectStore", object_factory)
    monkeypatch.setattr(worker_app, "TritonGrpcDetectorTransport", object_factory)
    monkeypatch.setattr(worker_app, "DetectorClient", object_factory)

    api_settings = production._ProductionSettings.model_construct(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
        web_root=web,
        model_assets_root=tmp_path,
        camera_cipher_key="secret",
        database_url="sqlite://",
        triton_grpc_url="localhost:8001",
    )
    with pytest.raises(StopCompositionError):
        _ = anyio.run(production._build_production_app, api_settings)  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    worker_settings = PipelineWorkerSettings.model_construct(
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
    report = source / "cuhk-report.json"
    report_data = json.loads(report.read_text())
    report_data["candidate_weights_sha256"] = hashlib.sha256(weight.read_bytes()).hexdigest()
    report.write_text(json.dumps(report_data))
    metadata = json.loads((source / "package.json").read_text())
    metadata["model_id"] = model_id
    for item in metadata["files"]:
        if item["path"] in {"model.safetensors", "cuhk-report.json"}:
            file = source / item["path"]
            item["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            item["size"] = file.stat().st_size
    (source / "package.json").write_text(json.dumps(metadata))
    manifest = import_clip_package(source, root)
    return root / "imported" / manifest.package_sha256


def test_manual_hash_valid_but_structurally_invalid_install_is_rejected(tmp_path: Path) -> None:
    directory = _installed_package(tmp_path, "local/valid", b"v")
    (directory / "cuhk-report.json").unlink()
    with pytest.raises(ValueError, match="invalid_installed_package"):
        _ = load_clip_registry(tmp_path)


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
        _ = load_clip_registry(tmp_path)


def test_refresh_adds_new_package_once_and_keeps_last_valid_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = _installed_package(tmp_path, "local/first", b"first")
    registry = load_clip_registry(tmp_path)
    default = registry.default
    original = registry.require("local/first")
    validations: list[str] = []
    load_manifest = registry_module._load_installed_manifest  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]

    def track(directory: Path) -> dict[str, object]:
        validations.append(directory.name)
        return load_manifest(directory)

    monkeypatch.setattr(registry_module, "_load_installed_manifest", track)
    assert not registry.refresh_from(tmp_path)
    assert validations == []

    second = _installed_package(tmp_path, "local/second", b"second")
    assert registry.refresh_from(tmp_path)
    assert registry.require("local/first") == original
    assert registry.require("local/second").revision == second.name
    assert registry.default == default
    assert validations == [second.name]

    corrupt = _installed_package(tmp_path, "local/corrupt", b"corrupt")
    _ = (corrupt / "model.safetensors").write_bytes(b"tampered")
    assert not registry.refresh_from(tmp_path)
    assert registry.get("local/corrupt") is None
    assert registry.require("local/first") == original
    assert registry.require("local/second").revision == second.name
    assert validations == [second.name, corrupt.name]
    assert not registry.refresh_from(tmp_path)
    assert validations == [second.name, corrupt.name]


def test_refresh_never_replaces_registered_identity(tmp_path: Path) -> None:
    original_directory = _installed_package(tmp_path, "local/active", b"original")
    registry = load_clip_registry(tmp_path)
    original = registry.require("local/active")
    default = registry.default

    other_root = tmp_path / "other"
    replacement = _installed_package(other_root, "local/active", b"replacement")
    _ = shutil.copytree(replacement, tmp_path / "imported" / replacement.name)

    assert not registry.refresh_from(tmp_path)
    assert registry.require("local/active") == original
    assert registry.require("local/active").revision == original_directory.name
    assert registry.default == default


@pytest.mark.anyio
async def test_api_apply_refreshes_before_candidate_identity_lookup(tmp_path: Path) -> None:
    registry = load_clip_registry(tmp_path)

    class Prepared:
        def status(self, package: ClipModelPackage) -> PreparedModelStatus:
            del package
            return PreparedModelStatus(prepared=True)

    service = ModelSelectionService(
        database=cast("Database", object()),
        registry=registry,
        prepared=Prepared(),
        imported_assets_root=tmp_path / "imported",
        quality_evidence_root=tmp_path / "quality-evidence",
    )
    candidate = _installed_package(tmp_path, "local/api-late", b"api-late")

    with pytest.raises(ModelNotPreparedError) as error:
        _ = await service.apply(cast("AsyncSession", object()), "local/api-late")

    assert registry.require("local/api-late").revision == candidate.name
    assert error.value.code == "model_quality_ineligible"


@pytest.mark.anyio
async def test_worker_pending_refreshes_before_target_identity_lookup(tmp_path: Path) -> None:
    registry = load_clip_registry(tmp_path)

    class Prepared:
        def status(self, package: ClipModelPackage) -> PreparedModelStatus:
            del package
            return PreparedModelStatus(prepared=True)

    class DatabaseStub:
        @asynccontextmanager
        async def transaction(self) -> AsyncIterator[object]:
            yield object()

    class EmptyRepository:
        async def active_job(self, _session: object, *, lock: bool = False) -> None:
            _ = lock

    service = ModelSelectionService(
        database=cast("Database", cast("object", DatabaseStub())),
        registry=registry,
        prepared=Prepared(),
        repository=cast("TransitionRepository", cast("object", EmptyRepository())),
        coordinator=cast("TransitionCoordinator", object()),
        imported_assets_root=tmp_path / "imported",
    )
    candidate = _installed_package(tmp_path, "local/worker-late", b"worker-late")

    def clip_factory(_package: ClipModelPackage) -> ClipImageEmbeddingPort:
        return cast("ClipImageEmbeddingPort", object())

    result = await service.run_pending(
        crop_store=cast("CropObjectStore", object()),
        runtime=cast("ClipRuntimePort", object()),
        clip_factory=cast("ClipFactoryPort", clip_factory),
        pipeline=cast("PipelineLifecyclePort", object()),
    )

    assert result is None
    assert registry.require("local/worker-late").revision == candidate.name


def test_duplicate_imported_model_id_is_rejected(tmp_path: Path) -> None:
    _ = _installed_package(tmp_path, "local/shared", b"first")
    other = tmp_path / "other"
    second = _installed_package(other, "local/shared", b"second")
    _ = shutil.copytree(second, tmp_path / "imported" / second.name)
    with pytest.raises(ValueError, match="duplicate_clip_model"):
        _ = load_clip_registry(tmp_path)


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
