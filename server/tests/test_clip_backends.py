import importlib.util
import sys
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import NoReturn, Protocol, cast

import pytest
from PIL import Image as PillowImage

REPOSITORY_ROOT = Path(__file__).parents[2]


class _RuntimeSettings(Protocol):
    model_id: str
    snapshot_path: Path
    revision: str
    dimension: int
    processor: str
    runtime: str


class _BackendModel(Protocol):
    @staticmethod
    def _decode_image(value: bytes) -> PillowImage.Image:
        """Decode one encoded image."""
        ...


class _ImageBackend(Protocol):
    DEFAULT_MODEL_ID: str
    DEFAULT_MODEL_PATH: Path
    DEFAULT_MODEL_REVISION: str
    DEFAULT_EMBEDDING_DIMENSION: int
    DEFAULT_PROCESSOR: str
    DEFAULT_RUNTIME: str
    RuntimeSettings: Callable[..., _RuntimeSettings]
    ImageDecodeError: type[ValueError]
    TritonPythonModel: type[_BackendModel]

    def _validate_snapshot_identity(
        self, settings: _RuntimeSettings, snapshot_config: Mapping[str, object]
    ) -> None: ...


def _load_backend(module_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(  # pyright: ignore[reportAttributeAccessIssue]
        OutOfMemoryError=RuntimeError
    )
    fake_pb_utils = types.ModuleType("triton_python_backend_utils")
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = object  # pyright: ignore[reportAttributeAccessIssue]
    fake_transformers.CLIPModel = object  # pyright: ignore[reportAttributeAccessIssue]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "triton_python_backend_utils", fake_pb_utils)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    module_name = f"backend_{module_path.parent.parent.name}_{module_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _image_backend(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_backend(
        REPOSITORY_ROOT / "inference/models/clip_image/1/model.py", monkeypatch
    )


def test_new_clip_package_without_marker_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = cast("_ImageBackend", cast("object", _image_backend(monkeypatch)))
    snapshot = tmp_path / "clip-vit-base-patch32"
    snapshot.mkdir()
    settings = backend.RuntimeSettings(
        model_id="openai/clip-vit-base-patch32",
        snapshot_path=snapshot,
        revision="revision-b32",
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )

    validate_identity = cast(
        "Callable[[_RuntimeSettings, Mapping[str, object]], None]",
        backend._validate_snapshot_identity,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    with pytest.raises(RuntimeError, match="clip_snapshot_identity_missing"):
        validate_identity(settings, {"_name_or_path": "openai/clip-vit-base-patch32"})


def test_legacy_b16_without_marker_remains_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = cast("_ImageBackend", cast("object", _image_backend(monkeypatch)))
    del tmp_path
    settings = backend.RuntimeSettings(
        model_id=backend.DEFAULT_MODEL_ID,
        snapshot_path=backend.DEFAULT_MODEL_PATH,
        revision=backend.DEFAULT_MODEL_REVISION,
        dimension=backend.DEFAULT_EMBEDDING_DIMENSION,
        processor=backend.DEFAULT_PROCESSOR,
        runtime=backend.DEFAULT_RUNTIME,
    )

    validate_identity = cast(
        "Callable[[_RuntimeSettings, Mapping[str, object]], None]",
        backend._validate_snapshot_identity,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    validate_identity(settings, {"_name_or_path": "openai/clip-vit-base-patch16"})


def test_pillow_decompression_bomb_is_a_decode_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = cast("_ImageBackend", cast("object", _image_backend(monkeypatch)))

    def raise_bomb(_source: object) -> NoReturn:
        message = "pixel limit"
        raise PillowImage.DecompressionBombError(message)

    monkeypatch.setattr(
        PillowImage, "open", cast("Callable[..., object]", raise_bomb)
    )

    decode_image = cast(
        "Callable[[bytes], PillowImage.Image]",
        backend.TritonPythonModel._decode_image,  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
    )
    with pytest.raises(backend.ImageDecodeError):
        _ = decode_image(b"encoded")
