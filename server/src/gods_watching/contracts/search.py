"""Discriminated person search request and response contracts."""

import unicodedata
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from .appearances import AppearanceResponse
from .base import ContractModel
from .identifiers import AppearanceId, CameraId
from .primitives import UtcDatetime


class SearchFilters(ContractModel):
    """Apply inclusive camera and UTC overlap filters to a search."""

    camera_ids: tuple[CameraId, ...] | None = None
    from_: UtcDatetime | None = Field(default=None, alias="from")
    to: UtcDatetime | None = None
    limit: Annotated[int, Field(ge=1, le=100)] = 30

    @model_validator(mode="after")
    def require_ordered_range(self) -> Self:
        """Reject an interval whose inclusive lower bound follows its upper bound."""
        if self.from_ is not None and self.to is not None and self.from_ > self.to:
            error_code = "reversed_time_range"
            error_message = "from must be before or equal to to"
            raise PydanticCustomError(error_code, error_message)
        return self


class TextSearchRequest(SearchFilters):
    """Search with a normalized printable-ASCII text description."""

    mode: Literal["text"]
    query: Annotated[str, Field(min_length=1, max_length=300)]
    sort: Literal["similarity"] = "similarity"

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        """Normalize supported text before it can become an embedding cache key."""
        normalized = unicodedata.normalize("NFKC", value)
        if any(character < " " or character > "~" for character in normalized):
            error_code = "unsupported_search_text"
            error_message = "query must contain printable ASCII text"
            raise PydanticCustomError(
                error_code,
                error_message,
            )
        collapsed = " ".join(normalized.split())
        if not collapsed:
            error_code = "blank_search_text"
            error_message = "query must not be blank"
            raise PydanticCustomError(error_code, error_message)
        return collapsed


class SimilarSearchRequest(SearchFilters):
    """Search by a committed appearance vector."""

    mode: Literal["similar"]
    appearance_id: AppearanceId
    sort: Literal["similarity"] = "similarity"


class BrowseSearchRequest(SearchFilters):
    """Browse committed appearances in newest-first order."""

    mode: Literal["browse"]
    sort: Literal["newest"] = "newest"


SearchRequest = Annotated[
    TextSearchRequest | SimilarSearchRequest | BrowseSearchRequest,
    Field(discriminator="mode"),
]


class SearchResponse(ContractModel):
    """Return stable ranked results and their applied mode."""

    mode: Literal["text", "similar", "browse"]
    results: tuple[AppearanceResponse, ...]
