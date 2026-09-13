"""Typed boundary for CLIP image and text inference."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Final, Protocol, final, override

import numpy as np
import tritonclient.grpc.aio as grpc_aio
from pydantic import ValidationError
from tritonclient.grpc import InferInput, InferRequestedOutput
from tritonclient.utils import InferenceServerException

from gods_watching.contracts.search import TextSearchRequest

if TYPE_CHECKING:
    from numpy.typing import NDArray

type ClipEmbedding = tuple[float, ...]
_EMBEDDING_DIMENSION: Final = 512
_UNIT_NORM_TOLERANCE: Final = 1e-3


@dataclass(frozen=True, slots=True)
class ClipInputError(ValueError):
    """Reject a request before it reaches inference."""

    code: str

    @override
    def __str__(self) -> str:
        return self.code


@dataclass(frozen=True, slots=True)
class ClipInferenceError(RuntimeError):
    """Report a failed or malformed Triton response."""

    code: str

    @override
    def __str__(self) -> str:
        return self.code


class ClipTransport(Protocol):
    """Provide modality-specific normalized embedding calls."""

    async def embed_image(self, image: bytes) -> Sequence[float]:
        """Return one normalized image embedding."""
        ...

    async def embed_text(self, text: str) -> Sequence[float]:
        """Return one normalized text embedding."""
        ...


@final
class TritonClipTransport:
    """Call the two resident CLIP models through Triton gRPC."""

    def __init__(self, url: str) -> None:
        """Create a client for a private Triton gRPC endpoint."""
        self._client = grpc_aio.InferenceServerClient(url=url)

    async def __aenter__(self) -> "TritonClipTransport":
        """Retain the transport until its context exits."""
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the owned gRPC channel."""
        del exception_type, exception, traceback
        await self._client.close()

    async def embed_image(self, image: bytes) -> ClipEmbedding:
        """Submit one encoded image to the resident image model."""
        return await self._infer(model_name="clip_image", input_name="IMAGE", value=image)

    async def embed_text(self, text: str) -> ClipEmbedding:
        """Submit one normalized query to the resident text model."""
        return await self._infer(
            model_name="clip_text", input_name="TEXT", value=text.encode("utf-8")
        )

    async def _infer(self, *, model_name: str, input_name: str, value: bytes) -> ClipEmbedding:
        infer_input = InferInput(input_name, [1, 1], "BYTES")
        _ = infer_input.set_data_from_numpy(np.asarray([[value]], dtype=np.object_))
        try:
            result = await self._client.infer(
                model_name=model_name,
                inputs=[infer_input],
                outputs=[InferRequestedOutput("EMBEDDING")],
            )
        except InferenceServerException as error:
            raise ClipInferenceError(code=f"clip_inference_failed: {error}") from error
        raw: NDArray[np.float32] | None = result.as_numpy("EMBEDDING")
        if raw is None:
            raise ClipInferenceError(code="clip_embedding_missing")
        return tuple(float(value) for value in raw.reshape(-1))


@final
class ClipAdapter:
    """Validate CLIP requests and enforce the shared vector contract."""

    def __init__(self, transport: ClipTransport) -> None:
        """Wrap one typed modality transport."""
        self._transport = transport

    async def embed_image(self, image: bytes) -> ClipEmbedding:
        """Validate encoded bytes and return a unit image embedding."""
        if not image:
            raise ClipInputError(code="clip_image_empty")
        return self._parse_embedding(await self._transport.embed_image(image))

    async def embed_text(self, text: str) -> ClipEmbedding:
        """Normalize in-scope text and return a unit text embedding."""
        try:
            request = TextSearchRequest(mode="text", query=text)
        except ValidationError as error:
            raise ClipInputError(code="clip_text_invalid") from error
        return self._parse_embedding(await self._transport.embed_text(request.query))

    @staticmethod
    def _parse_embedding(values: Sequence[float]) -> ClipEmbedding:
        embedding = tuple(float(value) for value in values)
        norm = math.sqrt(sum(value * value for value in embedding))
        if (
            len(embedding) != _EMBEDDING_DIMENSION
            or not all(math.isfinite(value) for value in embedding)
            or not math.isfinite(norm)
            or abs(norm - 1.0) > _UNIT_NORM_TOLERANCE
        ):
            raise ClipInferenceError(code="clip_embedding_invalid")
        return embedding
