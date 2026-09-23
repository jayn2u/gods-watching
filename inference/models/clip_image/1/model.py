"""Serve pinned CLIP image embeddings through Triton Python backend."""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image, UnidentifiedImageError
from transformers import AutoProcessor, CLIPModel

DEFAULT_MODEL_ID: Final = "openai/clip-vit-base-patch16"
DEFAULT_MODEL_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
DEFAULT_MODEL_PATH: Final = Path("/models/clip")
DEFAULT_EMBEDDING_DIMENSION: Final = 512
DEFAULT_PROCESSOR: Final = "CLIPProcessor"
DEFAULT_RUNTIME: Final = "transformers"
IDENTITY_MARKER_NAME: Final = "gods-watching-model.json"
# Backward-compatible names for diagnostics and offline probes.
MODEL_REVISION: Final = DEFAULT_MODEL_REVISION
MODEL_PATH: Final = DEFAULT_MODEL_PATH
EMBEDDING_DIMENSION: Final = DEFAULT_EMBEDDING_DIMENSION


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Explicit runtime parameters supplied by Triton's model config."""

    model_id: str
    snapshot_path: Path
    revision: str
    dimension: int
    processor: str
    runtime: str


class ImageDecodeError(ValueError):
    """Keep malformed image bytes separate from model/runtime failures."""


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _bad_image(message: str) -> NoReturn:
    raise UnidentifiedImageError(message)


def _string_parameter(
    parameters: Mapping[str, object], name: str, default: str | None = None
) -> str:
    raw = parameters.get(name)
    if raw is None and default is not None:
        return default
    if not isinstance(raw, Mapping):
        _fail(f"clip_parameter_missing: {name}")
    value = raw.get("string_value")
    if not isinstance(value, str) or not value:
        _fail(f"clip_parameter_invalid: {name}")
    return value


def _runtime_settings(model_config: Mapping[str, object]) -> RuntimeSettings:
    """Parse explicit package identity without consulting a network or env var."""
    raw_parameters = model_config.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        _fail("clip_parameters_invalid")
    dimension_text = _string_parameter(
        raw_parameters, "embedding_dimension", str(DEFAULT_EMBEDDING_DIMENSION)
    )
    try:
        dimension = int(dimension_text)
    except ValueError as error:
        _fail(f"clip_dimension_invalid: {dimension_text}")
        raise AssertionError from error
    if dimension < 1:
        _fail("clip_dimension_invalid")
    runtime = _string_parameter(raw_parameters, "runtime", DEFAULT_RUNTIME)
    if runtime != DEFAULT_RUNTIME:
        _fail(f"clip_runtime_unsupported: {runtime}")
    return RuntimeSettings(
        model_id=_string_parameter(raw_parameters, "model_id", DEFAULT_MODEL_ID),
        snapshot_path=Path(
            _string_parameter(raw_parameters, "snapshot_path", str(DEFAULT_MODEL_PATH))
        ),
        revision=_string_parameter(raw_parameters, "model_revision", DEFAULT_MODEL_REVISION),
        dimension=dimension,
        processor=_string_parameter(raw_parameters, "processor", DEFAULT_PROCESSOR),
        runtime=runtime,
    )


class TritonPythonModel:
    """Own one GPU CLIP image model for its Triton instance."""

    def initialize(self, args: dict[str, str]) -> None:
        """Load only the locally prepared snapshot named by explicit config."""
        model_config = json.loads(args["model_config"])
        settings = _runtime_settings(model_config)
        if not settings.snapshot_path.is_dir():
            _fail(f"clip_snapshot_missing: {settings.snapshot_path}")
        metadata_path = settings.snapshot_path / "config.json"
        if not metadata_path.is_file():
            _fail(f"clip_snapshot_metadata_missing: {metadata_path}")
        try:
            snapshot_config = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            _fail(f"clip_snapshot_metadata_invalid: {error}")
        _validate_snapshot_identity(settings, snapshot_config)
        actual_dimension = snapshot_config.get("projection_dim")
        if actual_dimension is not None and actual_dimension != settings.dimension:
            _fail(
                f"clip_dimension_mismatch: expected {settings.dimension}, "
                f"snapshot {actual_dimension}"
            )
        if not torch.cuda.is_available():
            _fail("clip_cuda_unavailable: GPU inference is required")
        self._settings = settings
        self._processor = AutoProcessor.from_pretrained(
            settings.snapshot_path, local_files_only=True
        )
        if self._processor.__class__.__name__ != settings.processor:
            _fail(
                f"clip_processor_mismatch: expected {settings.processor}, "
                f"actual {self._processor.__class__.__name__}"
            )
        self._model = CLIPModel.from_pretrained(settings.snapshot_path, local_files_only=True)
        self._model = self._model.to("cuda").eval()

    def execute(
        self, requests: list[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        """Embed every request batch or return one explicit request error."""
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                images = tuple(self._decode_image(value) for value in tensor.as_numpy().reshape(-1))
                inputs = self._processor(images=list(images), return_tensors="pt")
                inputs = {name: value.to("cuda") for name, value in inputs.items()}
                with torch.inference_mode():
                    features = self._model.get_image_features(**inputs)
                embeddings = self._normalize(features, self._settings.dimension)
                output = pb_utils.Tensor("EMBEDDING", embeddings)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except ImageDecodeError as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"clip_image_decode_failed: {error}")
                    )
                )
            except torch.cuda.OutOfMemoryError as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"clip_gpu_oom: {error}")
                    )
                )
            except (KeyError, OSError, RuntimeError, TypeError, UnicodeError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"clip_image_inference_failed: {error}")
                    )
                )
        return responses

    @staticmethod
    def _decode_image(value: bytes) -> Image.Image:
        try:
            if not value:
                _bad_image("encoded image is empty")
            with Image.open(io.BytesIO(value)) as source:
                source.load()
                return source.convert("RGB")
        except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError) as error:
            raise ImageDecodeError(str(error)) from error

    @staticmethod
    def _normalize(
        features: torch.Tensor, dimension: int = DEFAULT_EMBEDDING_DIMENSION
    ) -> np.ndarray:
        if not torch.isfinite(features).all():
            _fail("CLIP image features contain non-finite values")
        norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if torch.any(norms <= 0):
            _fail("CLIP image features have zero norm")
        normalized = features / norms
        result = normalized.detach().cpu().float().numpy()
        if result.shape[-1] != dimension or not np.isfinite(result).all():
            _fail(f"CLIP image embedding violates the FP32[{dimension}] contract")
        return np.ascontiguousarray(result, dtype=np.float32)


def _validate_snapshot_identity(
    settings: RuntimeSettings, snapshot_config: Mapping[str, object]
) -> None:
    """Validate local checkpoint identity and an optional preparation marker.

    Transformers snapshots do not always retain the Hub commit in
    ``config.json``.  Preparation can therefore emit the sidecar marker below
    to bind model id and immutable revision.  Older prepared B/16 snapshots
    remain usable when their local model id and dimension are still available;
    a malformed marker is always fatal.
    """
    actual_name = snapshot_config.get("_name_or_path")
    expected_name = Path(settings.model_id).name
    if (
        isinstance(actual_name, str)
        and actual_name.strip()
        and Path(actual_name.rstrip("/")).name != expected_name
    ):
        _fail(
            f"clip_model_id_mismatch: expected {settings.model_id}, "
            f"snapshot {actual_name}"
        )
    marker_path = settings.snapshot_path / IDENTITY_MARKER_NAME
    if not marker_path.is_file():
        legacy_default = (
            settings.model_id == DEFAULT_MODEL_ID
            and settings.snapshot_path == DEFAULT_MODEL_PATH
            and settings.revision == DEFAULT_MODEL_REVISION
            and settings.dimension == DEFAULT_EMBEDDING_DIMENSION
            and settings.processor == DEFAULT_PROCESSOR
            and settings.runtime == DEFAULT_RUNTIME
        )
        if not legacy_default:
            _fail(f"clip_snapshot_identity_missing: {marker_path}")
        return
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        _fail(f"clip_snapshot_identity_invalid: {error}")
    if not isinstance(marker, Mapping):
        _fail("clip_snapshot_identity_invalid")
    expected = {
        "model_id": settings.model_id,
        "revision": settings.revision,
        "dimension": settings.dimension,
        "processor": settings.processor,
        "runtime": settings.runtime,
    }
    if any(marker.get(key) != value for key, value in expected.items()):
        _fail("clip_snapshot_identity_mismatch")
