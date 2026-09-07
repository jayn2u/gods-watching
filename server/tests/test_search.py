import math
from collections.abc import Sequence
from typing import final

import pytest

from gods_watching.contracts.search import TextSearchRequest
from gods_watching.inference.clip import ClipInferenceError
from gods_watching.search import (
    InvalidCachedEmbeddingError,
    SearchInferenceUnavailableError,
    SearchRepository,
    SearchService,
    SearchTextInvalidError,
    TextEmbeddingCache,
    TextEmbeddingCacheKey,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _unit_embedding(index: int = 0) -> tuple[float, ...]:
    return tuple(1.0 if position == index else 0.0 for position in range(512))


@final
class _CountingTextTransport:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = tuple(values)
        self.calls: list[str] = []

    async def embed_text(self, text: str) -> Sequence[float]:
        self.calls.append(text)
        return self.values


class _FailingTextTransport:
    async def embed_text(self, text: str) -> Sequence[float]:
        del text
        raise ClipInferenceError(code="transport_offline")


class _InvalidTextTransport:
    async def embed_text(self, text: str) -> Sequence[float]:
        del text
        raise ClipInferenceError(code="clip_text_too_many_tokens: 78 exceeds 77")


def test_text_cache_evicts_oldest_successful_revision_key() -> None:
    # Given: a deliberately tiny revision-aware LRU cache
    cache = TextEmbeddingCache(capacity=2)
    first = TextEmbeddingCacheKey("revision-a", "first")
    second = TextEmbeddingCacheKey("revision-a", "second")
    third = TextEmbeddingCacheKey("revision-b", "third")

    # When: a third successful vector is inserted after touching the first
    _ = cache.put(first, _unit_embedding())
    _ = cache.put(second, _unit_embedding(1))
    assert cache.get(first) == _unit_embedding()
    _ = cache.put(third, _unit_embedding(2))

    # Then: the untouched oldest entry is evicted while revisions remain keyed
    assert len(cache) == 2
    assert cache.get(first) is not None
    assert cache.get(second) is None
    assert cache.get(third) == _unit_embedding(2)


@pytest.mark.anyio
async def test_text_search_reuses_normalized_query_cache_without_second_inference() -> None:
    # Given: Triton transport and one parsed normalized text request
    transport = _CountingTextTransport(_unit_embedding())
    service = SearchService(SearchRepository(), transport)
    request = TextSearchRequest(mode="text", query="  person   with bag ")

    # When: the same normalized query is embedded twice
    first = await service.embed_text(request)
    second = await service.embed_text(request)

    # Then: the successful finite vector is served from the bounded cache
    assert first == second == _unit_embedding()
    assert transport.calls == ["person with bag"]


@pytest.mark.anyio
async def test_text_search_does_not_cache_transport_failure() -> None:
    # Given: a text transport that reports Triton unavailability
    service = SearchService(SearchRepository(), _FailingTextTransport())
    request = TextSearchRequest(mode="text", query="person")

    # When / Then: the service exposes a typed unavailable outcome
    with pytest.raises(SearchInferenceUnavailableError):
        _ = await service.embed_text(request)
    assert len(service.text_cache) == 0


@pytest.mark.anyio
async def test_text_search_preserves_invalid_token_boundary_as_typed_error() -> None:
    service = SearchService(SearchRepository(), _InvalidTextTransport())
    request = TextSearchRequest(mode="text", query="person")

    with pytest.raises(SearchTextInvalidError):
        _ = await service.embed_text(request)
    assert len(service.text_cache) == 0


def test_text_cache_rejects_nonfinite_vector_without_insertion() -> None:
    # Given: a cache insertion containing a non-finite coordinate
    cache = TextEmbeddingCache()
    invalid = (math.nan, *([0.0] * 511))

    # When / Then: invalid vectors are rejected before they can be retained
    with pytest.raises(InvalidCachedEmbeddingError):
        _ = cache.put(TextEmbeddingCacheKey("revision", "query"), invalid)
    assert len(cache) == 0
