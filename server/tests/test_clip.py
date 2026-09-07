import math
from dataclasses import dataclass

import pytest

from gods_watching.inference.clip import (
    ClipAdapter,
    ClipInferenceError,
    ClipInputError,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True, slots=True)
class FakeClipTransport:
    embedding: tuple[float, ...]

    async def embed_image(self, image: bytes) -> tuple[float, ...]:
        del image
        return self.embedding

    async def embed_text(self, text: str) -> tuple[float, ...]:
        del text
        return self.embedding


def _unit_embedding() -> tuple[float, ...]:
    return (1.0, *([0.0] * 511))


@pytest.mark.anyio
async def test_text_embedding_when_query_is_valid_and_normalized() -> None:
    # Given: a typed transport that returns a valid unit CLIP vector
    adapter = ClipAdapter(FakeClipTransport(_unit_embedding()))
    # When: printable ASCII text contains collapsible whitespace
    embedding = await adapter.embed_text("  a person   with a red bag  ")
    # Then: the adapter accepts the request and preserves the vector contract
    assert embedding == _unit_embedding()


@pytest.mark.anyio
@pytest.mark.parametrize("query", ["", "   ", "검은 옷", "person 🎒", "person\nwith bag"])
async def test_text_embedding_rejected_when_query_is_out_of_scope(query: str) -> None:
    # Given: a transport that would accept any string
    adapter = ClipAdapter(FakeClipTransport(_unit_embedding()))
    # When / Then: the adapter rejects unsupported text before transport inference
    with pytest.raises(ClipInputError, match="clip_text_invalid"):
        _ = await adapter.embed_text(query)


@pytest.mark.anyio
async def test_image_embedding_rejected_when_bytes_are_empty() -> None:
    # Given: a configured typed adapter
    adapter = ClipAdapter(FakeClipTransport(_unit_embedding()))
    # When / Then: empty encoded image bytes are rejected at the boundary
    with pytest.raises(ClipInputError, match="clip_image_empty"):
        _ = await adapter.embed_image(b"")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "embedding",
    [
        (1.0,),
        (math.nan, *([0.0] * 511)),
        tuple([0.0] * 512),
        (2.0, *([0.0] * 511)),
    ],
)
async def test_embedding_rejected_when_transport_breaks_vector_contract(
    embedding: tuple[float, ...],
) -> None:
    # Given: a transport response with a malformed CLIP vector
    adapter = ClipAdapter(FakeClipTransport(embedding))
    # When / Then: dimensions, finite values, and unit norm are enforced
    with pytest.raises(ClipInferenceError, match="clip_embedding_invalid"):
        _ = await adapter.embed_text("a person")
