"""Typed retrieval services for committed appearance representatives."""

from .cache import InvalidCachedEmbeddingError, TextEmbeddingCache, TextEmbeddingCacheKey
from .crop import AppearanceLookupService, CropPayload
from .errors import (
    CropChangedDuringReadError,
    CropUnavailableError,
    SearchInferenceUnavailableError,
    SearchSeedNotFoundError,
    SearchTextInvalidError,
    UnknownCameraError,
)
from .repository import SearchCandidate, SearchRepository
from .service import (
    DEFAULT_CLIP_MODEL_REVISION,
    SearchService,
    SearchTextEmbeddingPort,
)

__all__ = [
    "DEFAULT_CLIP_MODEL_REVISION",
    "AppearanceLookupService",
    "CropChangedDuringReadError",
    "CropPayload",
    "CropUnavailableError",
    "InvalidCachedEmbeddingError",
    "SearchCandidate",
    "SearchInferenceUnavailableError",
    "SearchRepository",
    "SearchSeedNotFoundError",
    "SearchService",
    "SearchTextEmbeddingPort",
    "SearchTextInvalidError",
    "TextEmbeddingCache",
    "TextEmbeddingCacheKey",
    "UnknownCameraError",
]
