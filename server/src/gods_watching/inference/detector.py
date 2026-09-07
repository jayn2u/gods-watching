"""Typed gRPC adapter for the Triton person detector."""

import math
from array import array
from dataclasses import dataclass
from typing import Final, Protocol, final, override

import numpy as np
import numpy.typing as npt
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.grpc.aio import InferenceServerClient

type FloatTensor = npt.NDArray[np.float32]
type IntTensor = npt.NDArray[np.int32]

_MIN_CONFIDENCE: Final = 0.1
_MAX_CONFIDENCE: Final = 0.95
_MAX_DETECTIONS: Final = 300


@dataclass(frozen=True, slots=True)
class DetectorRequestError(ValueError):
    """Report a camera detector request outside the public contract."""

    confidence: float

    @override
    def __str__(self) -> str:
        return f"confidence must be finite and between 0.1 and 0.95: {self.confidence}"


@dataclass(frozen=True, slots=True)
class DetectorOutputError(RuntimeError):
    """Report a malformed response from the resident detector model."""

    detail: str

    @override
    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class DetectorRequest:
    """Carry one encoded frame and its exact camera confidence threshold."""

    encoded_image: bytes
    confidence: float

    def __post_init__(self) -> None:
        """Reject thresholds outside the camera settings contract."""
        if (
            not math.isfinite(self.confidence)
            or not _MIN_CONFIDENCE <= self.confidence <= _MAX_CONFIDENCE
        ):
            raise DetectorRequestError(confidence=self.confidence)


@dataclass(frozen=True, slots=True)
class DetectorTensorOutput:
    """Carry the two typed tensors declared by the Triton model config."""

    boxes: FloatTensor
    count: IntTensor


@dataclass(frozen=True, slots=True)
class Detection:
    """Represent one original-frame person bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Return coordinates in the order consumed by crop extraction."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Expose only the valid prefix of the fixed-size detector tensor."""

    detections: tuple[Detection, ...]

    @property
    def count(self) -> int:
        """Return the validated number of person detections."""
        return len(self.detections)


class DetectorTransport(Protocol):
    """Describe the narrow async transport needed by the typed adapter."""

    async def infer(
        self, *, encoded_image: bytes, confidence: float, timeout_seconds: float
    ) -> DetectorTensorOutput:
        """Return raw detector tensors or propagate the transport error."""
        ...


@final
class TritonGrpcDetectorTransport:
    """Send detector requests through Triton's asynchronous gRPC client."""

    def __init__(self, *, url: str) -> None:
        """Connect to a private Triton gRPC endpoint."""
        self._client = InferenceServerClient(url=url)

    async def infer(
        self, *, encoded_image: bytes, confidence: float, timeout_seconds: float
    ) -> DetectorTensorOutput:
        """Submit exact tensor names, shapes, and request confidence."""
        image_input = InferInput("IMAGE", [1, 1], "BYTES")
        _ = image_input.set_data_from_numpy(np.array([[encoded_image]], dtype=np.object_))
        confidence_input = InferInput("CONFIDENCE", [1, 1], "FP32")
        _ = confidence_input.set_data_from_numpy(np.array([[confidence]], dtype=np.float32))
        response = await self._client.infer(
            model_name="detector",
            inputs=[image_input, confidence_input],
            outputs=[InferRequestedOutput("BOXES"), InferRequestedOutput("COUNT")],
            client_timeout=timeout_seconds,
        )
        boxes = response.as_numpy("BOXES")
        count = response.as_numpy("COUNT")
        if not isinstance(boxes, np.ndarray) or boxes.dtype != np.float32:
            raise DetectorOutputError(detail="BOXES is missing or is not FP32")
        if not isinstance(count, np.ndarray) or count.dtype != np.int32:
            raise DetectorOutputError(detail="COUNT is missing or is not INT32")
        return DetectorTensorOutput(
            boxes=np.ascontiguousarray(boxes, dtype=np.float32),
            count=np.ascontiguousarray(count, dtype=np.int32),
        )

    async def close(self) -> None:
        """Close the underlying gRPC channel."""
        await self._client.close()


@final
class DetectorClient:
    """Validate detector tensors before they enter tracking and crop logic."""

    def __init__(self, *, transport: DetectorTransport, timeout_seconds: float = 2.0) -> None:
        """Set the transport and per-request deadline."""
        self._transport = transport
        self._timeout_seconds = timeout_seconds

    async def detect(self, request: DetectorRequest) -> DetectorResult:
        """Perform one deadline-bounded detector request and parse its result."""
        output = await self._transport.infer(
            encoded_image=request.encoded_image,
            confidence=request.confidence,
            timeout_seconds=self._timeout_seconds,
        )
        if output.boxes.shape != (1, 300, 6):
            raise DetectorOutputError(detail=f"BOXES has invalid shape {output.boxes.shape}")
        if output.count.shape != (1, 1):
            raise DetectorOutputError(detail=f"COUNT has invalid shape {output.count.shape}")
        counts = array("i")
        counts.frombytes(output.count.tobytes())
        count = counts[0]
        if not 0 <= count <= _MAX_DETECTIONS:
            raise DetectorOutputError(detail=f"COUNT is outside 0..300: {count}")
        values = array("f")
        values.frombytes(output.boxes.tobytes())
        detections: list[Detection] = []
        for index in range(count):
            offset = index * 6
            class_id = int(values[offset + 5])
            if class_id != 0:
                raise DetectorOutputError(detail=f"detector returned class {class_id}, expected 0")
            detections.append(
                Detection(
                    x1=values[offset],
                    y1=values[offset + 1],
                    x2=values[offset + 2],
                    y2=values[offset + 3],
                    confidence=values[offset + 4],
                    class_id=class_id,
                )
            )
        return DetectorResult(detections=tuple(detections))
