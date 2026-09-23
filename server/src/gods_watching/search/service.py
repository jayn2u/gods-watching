"""Application service for browse, text, and stored-vector retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.appearances import AppearanceResponse, BoundingBox
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.search import (
    BrowseSearchRequest,
    SearchRequest,
    SearchResponse,
    SimilarSearchRequest,
    TextSearchRequest,
)
from gods_watching.inference.clip import ClipInferenceError, ClipInputError
from gods_watching.model_selection.coordinator import TransitionCoordinator

from .cache import (
    InvalidCachedEmbeddingError,
    SearchEmbedding,
    TextEmbeddingCache,
    TextEmbeddingCacheKey,
)
from .errors import (
    SearchInferenceUnavailableError,
    SearchSeedNotFoundError,
    SearchTextInvalidError,
)
from .repository import SearchCandidate, SearchRepository

DEFAULT_CLIP_MODEL_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"


class SearchTextEmbeddingPort(Protocol):
    """Provide normalized text vectors through the Triton CLIP boundary."""

    async def embed_text(self, text: str) -> Sequence[float]:
        """Embed one already normalized text query."""
        ...


@dataclass(slots=True)
class SearchService:
    """Run typed search modes over a caller-owned database transaction."""

    repository: SearchRepository
    text_embeddings: SearchTextEmbeddingPort | None = None
    model_id: str = "openai/clip-vit-base-patch16"
    model_revision: str = DEFAULT_CLIP_MODEL_REVISION
    dimension: int = 512
    text_cache: TextEmbeddingCache = field(default_factory=TextEmbeddingCache)
    coordinator: TransitionCoordinator | None = None

    async def search(
        self,
        session: AsyncSession,
        request: SearchRequest,
    ) -> SearchResponse:
        """Execute one parsed search request and return only visible representatives."""
        coordinator = self.coordinator
        if coordinator is None:
            identity = await self._active_identity(session)
            candidates = await self._search_unlocked(session, request, identity)
        else:
            async with coordinator.search_lock(session):
                identity = await self._active_identity(session)
                candidates = await self._search_unlocked(session, request, identity)
        return SearchResponse(
            mode=request.mode,
            results=tuple(appearance_response(candidate) for candidate in candidates),
        )

    async def _search_unlocked(
        self,
        session: AsyncSession,
        request: SearchRequest,
        identity: tuple[str | None, str, int],
    ) -> tuple[SearchCandidate, ...]:
        model_id, model_revision, dimension = identity
        await self.repository.validate_camera_ids(session, request.camera_ids or ())
        match request:
            case BrowseSearchRequest():
                return await self.repository.browse(
                    session,
                    request,
                    model_id=model_id,
                    model_revision=model_revision,
                    dimension=dimension,
                )
            case SimilarSearchRequest():
                return await self._similar_from_seed(session, request, identity)
            case TextSearchRequest():
                embedding = await self._embed_text(request, identity)
                return await self.repository.similar(
                    session,
                    request,
                    embedding=embedding,
                    model_id=model_id,
                    model_revision=model_revision,
                    dimension=dimension,
                    exclude_appearance_id=None,
                )

    async def _similar_from_seed(
        self,
        session: AsyncSession,
        request: SimilarSearchRequest,
        identity: tuple[str | None, str, int],
    ) -> tuple[SearchCandidate, ...]:
        seed = await self.repository.get_seed(session, request.appearance_id)
        if seed is None or seed.appearance.embedding is None:
            raise SearchSeedNotFoundError(appearance_id=request.appearance_id)
        model_id, model_revision, dimension = identity
        effective_model_id = model_id
        effective_model_revision = model_revision or seed.appearance.model_revision
        if (
            (model_revision and seed.appearance.model_revision != model_revision)
            or seed.appearance.embedding_dimension != dimension
            or (model_id is not None and seed.appearance.model_id != model_id)
        ):
            raise SearchSeedNotFoundError(appearance_id=request.appearance_id)
        embedding = _parse_stored_embedding(
            seed.appearance.embedding,
            request.appearance_id,
            dimension=dimension,
        )
        if len(embedding) != dimension:
            raise SearchSeedNotFoundError(appearance_id=request.appearance_id)
        return await self.repository.similar(
            session,
            request,
            embedding=embedding,
            model_id=effective_model_id,
            model_revision=effective_model_revision,
            dimension=dimension,
            exclude_appearance_id=request.appearance_id,
        )

    async def _embed_text(
        self,
        request: TextSearchRequest,
        identity: tuple[str | None, str, int] | None = None,
    ) -> SearchEmbedding:
        if self.text_embeddings is None:
            raise SearchInferenceUnavailableError(reason="text transport is not configured")
        model_id, model_revision, dimension = identity or (
            self.model_id,
            self.model_revision,
            self.dimension,
        )
        if self.text_cache.dimension != dimension:
            self.text_cache = TextEmbeddingCache(
                capacity=self.text_cache.capacity,
                dimension=dimension,
            )
        key = TextEmbeddingCacheKey(
            model_revision=model_revision,
            normalized_query=request.query,
            model_id=model_id or self.model_id,
        )
        cached = self.text_cache.get(key)
        if cached is not None:
            return cached
        try:
            values = await self.text_embeddings.embed_text(request.query)
        except ClipInputError as error:
            raise SearchTextInvalidError(reason=error.code) from error
        except ClipInferenceError as error:
            if "clip_text_invalid" in error.code or "clip_text_too_many_tokens" in error.code:
                raise SearchTextInvalidError(reason=error.code) from error
            raise SearchInferenceUnavailableError(reason=type(error).__name__) from error
        except (ConnectionError, OSError, TimeoutError) as error:
            raise SearchInferenceUnavailableError(reason=type(error).__name__) from error
        try:
            return self.text_cache.put(key, values)
        except (InvalidCachedEmbeddingError, TypeError, ValueError) as error:
            raise SearchInferenceUnavailableError(reason="invalid text embedding") from error

    async def embed_text(self, request: TextSearchRequest) -> SearchEmbedding:
        """Return one normalized text vector through the revision-aware cache."""
        return await self._embed_text(request)

    async def _active_identity(self, session: AsyncSession) -> tuple[str | None, str, int]:
        if self.model_revision != DEFAULT_CLIP_MODEL_REVISION:
            return self.model_id, self.model_revision, self.dimension
        durable = await self.repository.active_identity(session)
        if durable is not None:
            return durable
        return self.model_id, self.model_revision, self.dimension


def _parse_stored_embedding(
    serialized: str,
    appearance_id: UUID,
    *,
    dimension: int = 512,
) -> SearchEmbedding:
    normalized = serialized.strip()
    if not (normalized.startswith("[") and normalized.endswith("]")):
        raise SearchSeedNotFoundError(appearance_id=appearance_id)
    try:
        values = tuple(
            float(piece.strip()) for piece in normalized[1:-1].split(",") if piece.strip()
        )
        cache = TextEmbeddingCache(capacity=1, dimension=dimension)
        return cache.put(TextEmbeddingCacheKey("stored", "stored"), values)
    except (InvalidCachedEmbeddingError, TypeError, ValueError) as error:
        raise SearchSeedNotFoundError(appearance_id=appearance_id) from error


def appearance_response(candidate: SearchCandidate) -> AppearanceResponse:
    """Convert one repository candidate without exposing its crop path or vector."""
    appearance = candidate.appearance
    similarity = candidate.similarity
    if similarity is not None:
        similarity = max(-1.0, min(1.0, similarity))
    return AppearanceResponse(
        appearance_id=AppearanceId(appearance.id),
        camera_id=CameraId(appearance.camera_id),
        camera_name=candidate.camera.name,
        session_id=CameraSessionId(appearance.session_id),
        track_id=appearance.track_id,
        first_seen=appearance.first_seen,
        last_seen=appearance.last_seen,
        ended_at=appearance.ended_at,
        representative_version=appearance.representative_version,
        bounding_box=BoundingBox(
            x_min=appearance.x_min,
            y_min=appearance.y_min,
            x_max=appearance.x_max,
            y_max=appearance.y_max,
        ),
        source_width=appearance.source_width,
        source_height=appearance.source_height,
        detector_confidence=appearance.detector_confidence,
        crop_quality=appearance.crop_quality,
        model_id=appearance.model_id,
        model_revision=appearance.model_revision,
        similarity=similarity,
    )
