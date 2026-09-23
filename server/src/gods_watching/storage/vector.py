"""SQLAlchemy mappings for model-aware pgvector representations."""

# ruff: noqa: D107, TRY003, EM101

from __future__ import annotations

from typing import override

from sqlalchemy.types import UserDefinedType


class Vector(UserDefinedType[str]):
    """Map an optionally dimensioned PostgreSQL vector column.

    The appearance table deliberately stores an unconstrained ``vector`` so
    512- and 768-dimensional model spaces can coexist during a transition.
    Retrieval adds an expression cast to the selected dimension and uses the
    matching partial HNSW index.
    """

    cache_ok: bool | None = True
    dimension: int | None

    def __init__(self, dimension: int | None = None) -> None:
        if dimension is not None and dimension < 1:
            raise ValueError("vector dimension must be positive")
        self.dimension = dimension

    @override
    def get_col_spec(self, **_kwargs: str) -> str:
        """Declare the pgvector dimension in generated SQL."""
        return "vector" if self.dimension is None else f"vector({self.dimension})"


class Vector512(UserDefinedType[str]):
    """Backward-compatible fixed 512-dimensional vector type."""

    cache_ok: bool | None = True

    @override
    def get_col_spec(self, **_kwargs: str) -> str:
        """Declare the pgvector dimension in generated SQL."""
        return "vector(512)"
