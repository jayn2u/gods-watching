"""Typed failures at the retrieval and crop ownership boundaries."""

from typing import final, override
from uuid import UUID


class SearchError(RuntimeError):
    """Base class for failures that the HTTP layer can map to API outcomes."""


@final
class UnknownCameraError(SearchError):
    """Report camera identifiers that are absent from durable history."""

    __slots__ = ("camera_ids",)

    camera_ids: tuple[UUID, ...]

    def __init__(self, camera_ids: tuple[UUID, ...]) -> None:
        """Initialize the typed missing-camera failure."""
        self.camera_ids = camera_ids
        super().__init__(camera_ids)

    @override
    def __str__(self) -> str:
        return "one or more cameras were not found"


@final
class SearchSeedNotFoundError(SearchError):
    """Report an absent, tombstoned, or vectorless similarity seed."""

    __slots__ = ("appearance_id",)

    appearance_id: UUID

    def __init__(self, appearance_id: UUID) -> None:
        """Initialize the typed missing-seed failure."""
        self.appearance_id = appearance_id
        super().__init__(appearance_id)

    @override
    def __str__(self) -> str:
        return "appearance was not found"


@final
class SearchInferenceUnavailableError(SearchError):
    """Report text embedding transport or vector failures."""

    __slots__ = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        """Initialize the typed inference-transport failure."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return "text search inference is unavailable"


@final
class SearchTextInvalidError(SearchError):
    """Report text rejected by the shared CLIP input/token boundary."""

    __slots__ = ("reason",)

    reason: str

    def __init__(self, reason: str) -> None:
        """Initialize the typed invalid-input failure."""
        self.reason = reason
        super().__init__(reason)

    @override
    def __str__(self) -> str:
        return "search text is invalid"


@final
class CropUnavailableError(SearchError):
    """Report a crop that cannot be read from its current object pointer."""

    __slots__ = ("appearance_id",)

    appearance_id: UUID

    def __init__(self, appearance_id: UUID) -> None:
        """Initialize the typed unavailable-crop failure."""
        self.appearance_id = appearance_id
        super().__init__(appearance_id)

    @override
    def __str__(self) -> str:
        return "appearance crop is unavailable"


@final
class CropChangedDuringReadError(SearchError):
    """Report a representative pointer changing during a crop read."""

    __slots__ = ("appearance_id", "representative_version")

    appearance_id: UUID
    representative_version: int

    def __init__(self, appearance_id: UUID, representative_version: int) -> None:
        """Initialize the typed crop-version race failure."""
        self.appearance_id = appearance_id
        self.representative_version = representative_version
        super().__init__(appearance_id, representative_version)

    @override
    def __str__(self) -> str:
        return "appearance representative changed during crop read"
