"""Filtered PostgreSQL retrieval over committed appearance representatives."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from uuid import UUID

from sqlalchemy import ColumnElement, Float, Select, cast, func, literal, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from gods_watching.contracts.search import SearchFilters
from gods_watching.storage import Appearance, Camera
from gods_watching.storage.vector import Vector512

from .cache import SearchEmbedding
from .errors import UnknownCameraError

_VISIBLE: Final = Appearance.tombstoned_at.is_(None)


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """Pair a visible appearance with the durable camera label and score."""

    appearance: Appearance
    camera: Camera
    similarity: float | None


@dataclass(frozen=True, slots=True)
class AppearanceSeed:
    """Carry the committed vector metadata required by a similar search."""

    appearance: Appearance
    camera: Camera


@dataclass(frozen=True, slots=True)
class SearchRepository:
    """Execute retrieval statements in a caller-owned PostgreSQL transaction."""

    async def validate_camera_ids(
        self,
        session: AsyncSession,
        camera_ids: Sequence[UUID],
    ) -> None:
        """Reject identifiers absent from durable camera history, including deleted rows."""
        if not camera_ids:
            return
        requested = tuple(dict.fromkeys(camera_ids))
        statement = select(Camera.id).where(Camera.id.in_(requested))
        found = set((await session.scalars(statement)).all())
        missing = tuple(sorted(set(requested) - found, key=str))
        if missing:
            raise UnknownCameraError(camera_ids=missing)

    async def get_seed(
        self,
        session: AsyncSession,
        appearance_id: UUID,
    ) -> AppearanceSeed | None:
        """Load a seed only when it is a current visible representative with a vector."""
        statement = (
            select(Appearance, Camera)
            .join(Camera, Camera.id == Appearance.camera_id)
            .where(
                Appearance.id == appearance_id,
                _VISIBLE,
                Appearance.embedding.is_not(None),
            )
        )
        row = (await session.execute(statement)).tuples().one_or_none()
        if row is None:
            return None
        appearance, camera = row
        return AppearanceSeed(appearance=appearance, camera=camera)

    async def browse(
        self,
        session: AsyncSession,
        filters: SearchFilters,
    ) -> tuple[SearchCandidate, ...]:
        """Return visible rows newest by first-seen time with deterministic ID ties."""
        statement = self._base_statement(filters).order_by(
            Appearance.first_seen.desc(),
            Appearance.id.asc(),
        )
        rows = (await session.execute(statement.limit(filters.limit))).tuples().all()
        return tuple(SearchCandidate(appearance, camera, None) for appearance, camera in rows)

    async def similar(
        self,
        session: AsyncSession,
        filters: SearchFilters,
        *,
        embedding: SearchEmbedding,
        model_revision: str,
        exclude_appearance_id: UUID | None,
    ) -> tuple[SearchCandidate, ...]:
        """Run strict filtered HNSW retrieval and exact filtered fallback when needed."""
        base = self._similar_statement(
            filters,
            embedding=embedding,
            model_revision=model_revision,
            exclude_appearance_id=exclude_appearance_id,
        )
        eligible = await self._eligible_count(
            session,
            filters,
            model_revision=model_revision,
            exclude_appearance_id=exclude_appearance_id,
        )
        _ = await session.execute(text("SET LOCAL hnsw.iterative_scan = 'strict_order'"))
        rows = await self._run_similar(session, base, filters.limit)
        required = min(filters.limit, eligible)
        if len(rows) < required:
            _ = await session.execute(text("SET LOCAL enable_indexscan = off"))
            _ = await session.execute(text("SET LOCAL enable_bitmapscan = off"))
            _ = await session.execute(text("SET LOCAL hnsw.iterative_scan = 'off'"))
            rows = await self._run_similar(session, base, filters.limit)
        return tuple(
            SearchCandidate(appearance, camera, float(score)) for appearance, camera, score in rows
        )

    async def get_visible(
        self,
        session: AsyncSession,
        appearance_id: UUID,
    ) -> tuple[Appearance, Camera] | None:
        """Read the current appearance pointer and camera label for detail/crop use."""
        statement = (
            select(Appearance, Camera)
            .join(Camera, Camera.id == Appearance.camera_id)
            .where(Appearance.id == appearance_id, _VISIBLE, Appearance.embedding.is_not(None))
        )
        fresh_statement = statement.execution_options(populate_existing=True)
        return (await session.execute(fresh_statement)).tuples().one_or_none()

    def _base_statement(self, filters: SearchFilters) -> Select[tuple[Appearance, Camera]]:
        predicates = self._filter_predicates(filters)
        return (
            select(Appearance, Camera)
            .join(Camera, Camera.id == Appearance.camera_id)
            .where(
                _VISIBLE,
                Appearance.embedding.is_not(None),
                *predicates,
            )
        )

    def _similar_statement(
        self,
        filters: SearchFilters,
        *,
        embedding: SearchEmbedding,
        model_revision: str,
        exclude_appearance_id: UUID | None,
    ) -> Select[tuple[Appearance, Camera, float]]:
        vector = "[" + ",".join(format(value, ".9g") for value in embedding) + "]"
        query_vector = cast(literal(vector), Vector512())
        distance: ColumnElement[float] = Appearance.embedding.op("<=>")(query_vector).cast(Float)
        similarity = (literal(1.0) - distance).label("similarity")
        predicates = [
            *self._filter_predicates(filters),
            Appearance.model_revision == model_revision,
            Appearance.embedding.is_not(None),
        ]
        if exclude_appearance_id is not None:
            predicates.append(Appearance.id != exclude_appearance_id)
        return (
            select(Appearance, Camera, similarity)
            .join(Camera, Camera.id == Appearance.camera_id)
            .where(_VISIBLE, *predicates)
            .order_by(distance.asc(), Appearance.id.asc())
        )

    async def _eligible_count(
        self,
        session: AsyncSession,
        filters: SearchFilters,
        *,
        model_revision: str,
        exclude_appearance_id: UUID | None,
    ) -> int:
        predicates = [
            *self._filter_predicates(filters),
            Appearance.model_revision == model_revision,
            Appearance.embedding.is_not(None),
        ]
        if exclude_appearance_id is not None:
            predicates.append(Appearance.id != exclude_appearance_id)
        statement = (
            select(func.count(Appearance.id))
            .join(Camera, Camera.id == Appearance.camera_id)
            .where(_VISIBLE, *predicates)
        )
        count = await session.scalar(statement)
        if count is None:
            raise AssertionError
        return int(count)

    @staticmethod
    async def _run_similar(
        session: AsyncSession,
        statement: Select[tuple[Appearance, Camera, float]],
        limit: int,
    ) -> list[tuple[Appearance, Camera, float]]:
        rows = (await session.execute(statement.limit(limit))).tuples().all()
        return list(rows)

    @staticmethod
    def _filter_predicates(filters: SearchFilters) -> list[ColumnElement[bool]]:
        predicates: list[ColumnElement[bool]] = []
        if filters.camera_ids:
            predicates.append(Appearance.camera_id.in_(filters.camera_ids))
        if filters.from_ is not None:
            predicates.append(Appearance.last_seen >= filters.from_)
        if filters.to is not None:
            predicates.append(Appearance.first_seen <= filters.to)
        return predicates
