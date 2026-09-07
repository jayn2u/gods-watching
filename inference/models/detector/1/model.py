"""Triton Python backend for GPU YOLO11s person detection."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final, override

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

_BOX_CAPACITY: Final = 300
_BOX_COLUMNS: Final = 6
_IMAGE_SIZE: Final = 640
_IOU_THRESHOLD: Final = 0.7
_PERSON_CLASS: Final = 0
_MIN_CONFIDENCE: Final = 0.1
_MAX_CONFIDENCE: Final = 0.95


@dataclass(frozen=True, slots=True)
class DetectorBackendRequestError(ValueError):
    """Report an invalid tensor value at the backend request boundary."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class DetectorBackendInferenceError(RuntimeError):
    """Report an unexpected YOLO result cardinality."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


class TritonPythonModel:
    """Load YOLO once and execute independently thresholded batched requests."""

    def initialize(self, args: dict[str, str]) -> None:
        """Load the configured immutable weights on CUDA device zero."""
        config = json.loads(args["model_config"])
        weights_path = Path(config["parameters"]["weights_path"]["string_value"])
        if not weights_path.is_file():
            message = f"detector weights are missing: {weights_path}"
            raise pb_utils.TritonModelException(message)
        if not torch.cuda.is_available():
            message = "CUDA is required for detector inference"
            raise pb_utils.TritonModelException(message)
        self._model = YOLO(str(weights_path))
        self._model.to("cuda:0")

    @staticmethod
    def _decode(encoded: bytes) -> Image.Image:
        with Image.open(io.BytesIO(encoded)) as source:
            source.load()
            return source.convert("RGB")

    def _detect(self, encoded: bytes, confidence: float) -> np.ndarray:
        if not np.isfinite(confidence) or not _MIN_CONFIDENCE <= confidence <= _MAX_CONFIDENCE:
            raise DetectorBackendRequestError(
                detail=f"CONFIDENCE must be finite and between 0.1 and 0.95: {confidence}"
            )
        image = self._decode(encoded)
        results = self._model.predict(
            source=image,
            imgsz=_IMAGE_SIZE,
            conf=confidence,
            iou=_IOU_THRESHOLD,
            max_det=_BOX_CAPACITY,
            classes=[_PERSON_CLASS],
            device=0,
            half=False,
            verbose=False,
        )
        if len(results) != 1:
            raise DetectorBackendInferenceError(
                detail=f"YOLO returned {len(results)} results for one image"
            )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, _BOX_COLUMNS), dtype=np.float32)
        return boxes.data.detach().to(device="cpu", dtype=torch.float32).numpy()

    def execute(
        self, requests: list[pb_utils.InferenceRequest]
    ) -> list[pb_utils.InferenceResponse]:
        """Decode every request in memory and return fixed-capacity tensors."""
        responses: list[pb_utils.InferenceResponse] = []
        for request in requests:
            try:
                image_tensor = pb_utils.get_input_tensor_by_name(request, "IMAGE")
                confidence_tensor = pb_utils.get_input_tensor_by_name(request, "CONFIDENCE")
                if image_tensor is None or confidence_tensor is None:
                    raise DetectorBackendRequestError(
                        detail="IMAGE and CONFIDENCE inputs are required"
                    )
                encoded_batch = image_tensor.as_numpy().reshape(-1)
                confidence_batch = confidence_tensor.as_numpy().reshape(-1)
                if len(encoded_batch) != len(confidence_batch):
                    raise DetectorBackendRequestError(
                        detail="IMAGE and CONFIDENCE batch sizes differ"
                    )
                output_boxes = np.zeros(
                    (len(encoded_batch), _BOX_CAPACITY, _BOX_COLUMNS),
                    dtype=np.float32,
                )
                output_counts = np.zeros((len(encoded_batch), 1), dtype=np.int32)
                for index, (encoded, confidence) in enumerate(
                    zip(encoded_batch, confidence_batch, strict=True)
                ):
                    if not isinstance(encoded, bytes):
                        raise DetectorBackendRequestError(
                            detail="IMAGE elements must contain encoded bytes"
                        )
                    detected = self._detect(encoded, float(confidence))
                    count = min(len(detected), _BOX_CAPACITY)
                    output_boxes[index, :count] = detected[:count]
                    output_counts[index, 0] = count
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("BOXES", output_boxes),
                            pb_utils.Tensor("COUNT", output_counts),
                        ]
                    )
                )
            except (OSError, RuntimeError, UnidentifiedImageError, ValueError) as error:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"detector request failed: {error}")
                    )
                )
        return responses
