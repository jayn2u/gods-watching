"""Bounded model-revision-aware text embedding cache."""

import math
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, final

type SearchEmbedding = tuple[float, ...]

_EMBEDDING_DIMENSION: Final = 512
_UNIT_NORM_TOLERANCE: Final = 1e-3
_DEFAULT_CAPACITY: Final = 256


@dataclass(frozen=True, slots=True)
class TextEmbeddingCacheKey:
    """Identify a normalized query within one exact model package revision."""

    model_revision: str
    normalized_query: str
    model_id: str = ""


class InvalidCachedEmbeddingError(ValueError):
    """Reject a cache insertion that would make retrieval nondeterministic."""


@final
class TextEmbeddingCache:
    """Maintain at most 256 successful finite unit text vectors.

    The cache is intentionally mutable because bounded LRU eviction is its
    documented purpose. It is event-loop local and has no awaits in its methods.
    """

    __slots__: tuple[str, ...] = ("_capacity", "_dimension", "_entries")
    _capacity: int
    _dimension: int

    def __init__(
        self, capacity: int = _DEFAULT_CAPACITY, dimension: int = _EMBEDDING_DIMENSION
    ) -> None:
        """Create an empty cache with positive bounded capacity and dimension."""
        if capacity < 1:
            raise ValueError
        if dimension < 1:
            raise ValueError
        self._capacity = capacity
        self._dimension = dimension
        self._entries: OrderedDict[TextEmbeddingCacheKey, SearchEmbedding] = OrderedDict()

    def get(self, key: TextEmbeddingCacheKey) -> SearchEmbedding | None:
        """Return and promote a successful vector, if present."""
        embedding = self._entries.get(key)
        if embedding is None:
            return None
        self._entries.move_to_end(key)
        return embedding

    def put(self, key: TextEmbeddingCacheKey, values: Sequence[float]) -> SearchEmbedding:
        """Validate and retain one successful vector, evicting the oldest entry."""
        embedding = _validated_embedding(values, dimension=self._dimension)
        self._entries[key] = embedding
        self._entries.move_to_end(key)
        while len(self._entries) > self._capacity:
            _ = self._entries.popitem(last=False)
        return embedding

    def clear(self) -> None:
        """Remove every cached revision and query."""
        self._entries.clear()

    def __len__(self) -> int:
        """Return the number of successful cached vectors."""
        return len(self._entries)

    @property
    def capacity(self) -> int:
        """Return the fixed maximum entry count."""
        return self._capacity

    @property
    def dimension(self) -> int:
        """Return the selected vector dimension enforced by this cache."""
        return self._dimension


def _validated_embedding(
    values: Sequence[float], *, dimension: int = _EMBEDDING_DIMENSION
) -> SearchEmbedding:
    embedding = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in embedding))
    if (
        len(embedding) != dimension
        or not all(math.isfinite(value) for value in embedding)
        or not math.isfinite(norm)
        or abs(norm - 1.0) > _UNIT_NORM_TOLERANCE
    ):
        raise InvalidCachedEmbeddingError
    return embedding
