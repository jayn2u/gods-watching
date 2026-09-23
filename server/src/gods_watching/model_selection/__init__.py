"""Immutable metadata and preparation status for selectable CLIP packages."""

from .assets import IDENTITY_MARKER_NAME, PreparedModelCatalog, PreparedModelStatus
from .registry import (
    B16_REVISION,
    B32_REVISION,
    BUILTIN_CLIP_MODELS,
    DEFAULT_CLIP_MODEL,
    DEFAULT_CLIP_MODEL_ID,
    L14_REVISION,
    ClipModelPackage,
    ClipModelRegistry,
    UnknownClipModelError,
    get_clip_model,
)

__all__ = [
    "B16_REVISION",
    "B32_REVISION",
    "BUILTIN_CLIP_MODELS",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_CLIP_MODEL_ID",
    "IDENTITY_MARKER_NAME",
    "L14_REVISION",
    "ClipModelPackage",
    "ClipModelRegistry",
    "PreparedModelCatalog",
    "PreparedModelStatus",
    "UnknownClipModelError",
    "get_clip_model",
]
