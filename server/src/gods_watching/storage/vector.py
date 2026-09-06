"""SQLAlchemy mapping for the fixed CLIP pgvector representation."""

from typing import override

from sqlalchemy.types import UserDefinedType


class Vector512(UserDefinedType[str]):
    """Map the fixed PostgreSQL vector column without an extra driver package."""

    cache_ok: bool | None = True

    @override
    def get_col_spec(self, **_kwargs: str) -> str:
        """Declare the pgvector dimension in generated SQL."""
        return "vector(512)"
