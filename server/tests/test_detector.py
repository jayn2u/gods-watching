from typing import final

import anyio
import numpy as np
import pytest

from gods_watching.inference.detector import (
    DetectorClient,
    DetectorOutputError,
    DetectorRequest,
    DetectorTensorOutput,
)


@final
class RecordingTransport:
    def __init__(self, *, output: DetectorTensorOutput) -> None:
        self.output: DetectorTensorOutput = output
        self.observed_confidence: float | None = None

    async def infer(
        self, *, encoded_image: bytes, confidence: float, timeout_seconds: float
    ) -> DetectorTensorOutput:
        assert encoded_image == b"encoded-frame"
        assert timeout_seconds > 0
        self.observed_confidence = confidence
        return self.output


def _valid_output() -> DetectorTensorOutput:
    boxes = np.zeros((1, 300, 6), dtype=np.float32)
    boxes[0, 0] = (10.0, 20.0, 110.0, 220.0, 0.81, 0.0)
    count = np.array([[1]], dtype=np.int32)
    return DetectorTensorOutput(boxes=boxes, count=count)


def test_request_preserves_low_camera_threshold() -> None:
    # Given: the minimum approved detector threshold and a typed transport
    transport = RecordingTransport(output=_valid_output())
    client = DetectorClient(transport=transport, timeout_seconds=1.5)
    request = DetectorRequest(encoded_image=b"encoded-frame", confidence=0.1)
    # When: the detector adapter performs one request
    result = anyio.run(client.detect, request)
    # Then: no implicit library default raises the threshold and one person is parsed
    assert transport.observed_confidence == 0.1
    assert result.count == 1
    assert result.detections[0].class_id == 0
    assert result.detections[0].xyxy == (10.0, 20.0, 110.0, 220.0)


@pytest.mark.parametrize("confidence", [0.099, 0.951, float("nan")])
def test_request_rejects_confidence_outside_camera_contract(confidence: float) -> None:
    # Given: a threshold outside the approved camera boundary
    # When/Then: parsing the adapter request rejects it before network inference
    with pytest.raises(ValueError, match="confidence"):
        _ = DetectorRequest(encoded_image=b"encoded-frame", confidence=confidence)


def test_adapter_rejects_malformed_tensor_shape() -> None:
    # Given: a transport response whose box tensor violates Triton's declared shape
    malformed = DetectorTensorOutput(
        boxes=np.zeros((1, 299, 6), dtype=np.float32),
        count=np.array([[0]], dtype=np.int32),
    )
    client = DetectorClient(transport=RecordingTransport(output=malformed))
    request = DetectorRequest(encoded_image=b"encoded-frame", confidence=0.5)
    # When/Then: the typed adapter rejects the response instead of truncating it
    with pytest.raises(DetectorOutputError, match="BOXES"):
        _ = anyio.run(client.detect, request)


def test_adapter_rejects_non_person_class() -> None:
    # Given: a backend response that claims an out-of-contract class
    output = _valid_output()
    output.boxes[0, 0, 5] = 2.0
    client = DetectorClient(transport=RecordingTransport(output=output))
    request = DetectorRequest(encoded_image=b"encoded-frame", confidence=0.5)
    # When/Then: class filtering cannot silently regress at the service boundary
    with pytest.raises(DetectorOutputError, match="class"):
        _ = anyio.run(client.detect, request)
