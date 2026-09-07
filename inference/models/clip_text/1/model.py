"""Serve pinned CLIP text embeddings through Triton Python backend."""

from __future__ import annotations

import json
import os
import unicodedata
from pathlib import Path
from typing import Final, NoReturn

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, CLIPModel

MODEL_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
MODEL_PATH: Final = Path("/models/clip")
MAX_TOKENS: Final = 77
MAX_TEXT_CHARACTERS: Final = 300
EMBEDDING_DIMENSION: Final = 512


def _fail(message: str) -> NoReturn:
    raise RuntimeError(message)


def _bad_text(message: str) -> NoReturn:
    raise UnicodeError(message)


class TritonPythonModel:
    """Own one GPU CLIP text model for its Triton instance."""

    def initialize(self, args: dict[str, str]) -> None:
        """Load only the prepared pinned model snapshot on CUDA."""
        model_config = json.loads(args["model_config"])
        configured_revision = model_config["parameters"]["model_revision"]["string_value"]
        requested_revision = os.environ.get("GW_CLIP_REVISION", configured_revision)
        if configured_revision != MODEL_REVISION or requested_revision != MODEL_REVISION:
            message = (
                f"clip_revision_mismatch: expected {MODEL_REVISION}, "
                f"configured {configured_revision}, requested {requested_revision}"
            )
            _fail(message)
        if not MODEL_PATH.is_dir():
            _fail(f"clip_snapshot_missing: {MODEL_PATH}")
        if not torch.cuda.is_available():
            _fail("clip_cuda_unavailable: GPU inference is required")
        self._processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
        self._model = CLIPModel.from_pretrained(MODEL_PATH, local_files_only=True)
        self._model = self._model.to("cuda").eval()

    def execute(
        self, requests: list[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        """Embed every request batch or return one explicit request error."""
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
                texts = tuple(self._parse_text(value) for value in tensor.as_numpy().reshape(-1))
                inputs = self._processor(
                    text=list(texts), return_tensors="pt", padding=True, truncation=False
                )
                token_count = int(inputs["input_ids"].shape[-1])
                if token_count > MAX_TOKENS:
                    _fail(f"clip_text_too_many_tokens: {token_count} exceeds {MAX_TOKENS}")
                inputs = {name: value.to("cuda") for name, value in inputs.items()}
                with torch.inference_mode():
                    features = self._model.get_text_features(**inputs)
                embeddings = self._normalize(features)
                output = pb_utils.Tensor("EMBEDDING", embeddings)
                responses.append(pb_utils.InferenceResponse(output_tensors=[output]))
            except (KeyError, RuntimeError, TypeError, UnicodeError) as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"clip_text_invalid: {error}")
                    )
                )
        return responses

    @staticmethod
    def _parse_text(value: bytes) -> str:
        text = unicodedata.normalize("NFKC", value.decode("utf-8"))
        if any(character < " " or character > "~" for character in text):
            _bad_text("text must contain printable ASCII characters only")
        normalized = " ".join(text.split())
        if not normalized:
            _bad_text("text must not be empty")
        if len(normalized) > MAX_TEXT_CHARACTERS:
            _bad_text("text must not exceed 300 characters")
        return normalized

    @staticmethod
    def _normalize(features: torch.Tensor) -> np.ndarray:
        if not torch.isfinite(features).all():
            _fail("CLIP text features contain non-finite values")
        norms = torch.linalg.vector_norm(features, dim=-1, keepdim=True)
        if torch.any(norms <= 0):
            _fail("CLIP text features have zero norm")
        normalized = features / norms
        result = normalized.detach().cpu().float().numpy()
        if result.shape[-1] != EMBEDDING_DIMENSION or not np.isfinite(result).all():
            _fail("CLIP text embedding violates the FP32[512] contract")
        return np.ascontiguousarray(result, dtype=np.float32)
