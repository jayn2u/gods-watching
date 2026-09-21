"""Transactional publication of bounded representative crops and vectors."""

from __future__ import annotations

import shutil
import time
from asyncio import CancelledError
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from math import isfinite
from typing import TYPE_CHECKING, Final
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.storage import (
    Appearance,
    ApplicationSettings,
    Camera,
    CameraSession,
    CropGarbage,
    CropObjectStore,
    Database,
    StaleAppearanceVersionError,
    StorageRepository,
)
from gods_watching.storage.managed_files import ManagedFileKind, safe_scan, safe_unlink
from gods_watching.tracking.models import LifecycleKind

from .budget import (
    BudgetLease,
    BudgetSnapshot,
    WriterBudget,
    WriterGate,
    WriterGateProvider,
)
from .encoding import encode_rgb_crop, laplacian_variance
from .policy import (
    CandidateRank,
    CropRejectedError,
    appearance_id_for_track,
    rank_candidate,
    should_upgrade,
    validate_crop_geometry,
)
from .queue import PendingEmbedding, PendingEmbeddingQueue, QueueDecision, QueueKey

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.contracts.identifiers import AppearanceId, CameraId
    from gods_watching.contracts.pipeline import GenerationBinding, PipelineHandoff
    from gods_watching.inference.clip import ClipAdapter

_DEFAULT_QUOTA_BYTES: Final = 100_000_000_000


class PublicationOutcome(StrEnum):
    """Describe the durable result of one lifecycle handoff."""

    PUBLISHED = "published"
    METADATA_UPDATED = "metadata_updated"
    ENDED = "ended"
    NOOP = "noop"
    QUEUED = "queued"
    DROPPED = "dropped"
    REJECTED = "rejected"
    SUPPRESSED = "suppressed"
    PAUSED = "paused"
    STALE_GENERATION = "stale_generation"
    STALE_VERSION = "stale_version"
    EMBEDDING_FAILED = "embedding_failed"
    COMMIT_FAILED = "commit_failed"


PublicationStatus = PublicationOutcome


class PublicationError(RuntimeError):
    """Describe an impossible publication queue payload or clock value."""


class PublicationInputError(ValueError):
    """Describe a malformed monotonic clock supplied by a caller."""


@dataclass(frozen=True, slots=True)
class PublicationAck:
    """Report ordering clocks and visibility for one publication decision."""

    outcome: PublicationOutcome
    appearance_id: AppearanceId
    sequence: int
    t_detect_monotonic: float
    t_searchable_monotonic: float | None = None
    representative_version: int | None = None
    queue_decision: QueueDecision | None = None
    detail: str | None = None

    @property
    def status(self) -> PublicationOutcome:
        """Expose the outcome under the service status vocabulary."""
        return self.outcome


@dataclass(frozen=True, slots=True)
class PublicationWork:
    """Carry normalized metadata and JPEG bytes without retaining an RGB frame."""

    key: QueueKey
    appearance_id: AppearanceId
    generation: GenerationBinding
    representative_version: int
    rank: CandidateRank
    jpeg_bytes: bytes
    bounding_box: tuple[int, int, int, int]
    source_width: int
    source_height: int
    detector_confidence: float
    first_seen: datetime
    last_seen: datetime
    t_detect_monotonic: float
    sequence: int
    ended_at: datetime | None = None


@dataclass(slots=True)
class _TrackState:
    """Retain clocks and committed representative policy for one queue key."""

    first_seen: datetime
    t_detect_monotonic: float
    last_seen: datetime
    committed_version: int = 0
    rank: CandidateRank | None = None
    last_upgrade_monotonic: float | None = None
    ended: bool = False


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Count durable crop files removed during orphan reconciliation."""

    removed_jpegs: int
    removed_temps: int
    referenced_jpegs: int


class AppearancePublisher:
    """Fence, rank, embed, and atomically publish appearance representatives."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        database: Database,
        storage: StorageRepository,
        crop_store: CropObjectStore,
        clip: ClipAdapter,
        model_id: str,
        model_revision: str,
        writer_budget: WriterBudget | None = None,
        queue: PendingEmbeddingQueue[PublicationWork] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
        active_binding: Callable[[CameraId], Awaitable[GenerationBinding | None]] | None = None,
    ) -> None:
        """Create a publisher whose writes require an explicit budget reservation."""
        self.database: Database = database
        self.storage: StorageRepository = storage
        self.crop_store: CropObjectStore = crop_store
        self.clip: ClipAdapter = clip
        self.model_id: str = model_id
        self.model_revision: str = model_revision
        self.writer_budget: WriterBudget | None = writer_budget
        self.queue: PendingEmbeddingQueue[PublicationWork] = queue or PendingEmbeddingQueue()
        self._monotonic_clock: Callable[[], float] = monotonic_clock
        self._wall_clock: Callable[[], datetime] = wall_clock or (lambda: datetime.now(UTC))
        self._states: dict[QueueKey, _TrackState] = {}
        self._suppressed: set[AppearanceId] = set()
        self._pending_appearance_ids: dict[QueueKey, AppearanceId] = {}
        self._active_binding: Callable[[CameraId], Awaitable[GenerationBinding | None]] | None = (
            active_binding
        )
        self.writer_gate: WriterGate = (
            writer_budget.writer_gate
            if isinstance(writer_budget, WriterGateProvider)
            else WriterGate()
        )

    @property
    def pending_embeddings(self) -> int:
        """Return the current bounded embedding backlog."""
        return len(self.queue)

    async def accept_handoff(  # noqa: C901, PLR0911, PLR0912
        self,
        handoff: PipelineHandoff,
        *,
        now_monotonic: float | None = None,
    ) -> PublicationAck:
        """Accept one canonical lifecycle handoff and drain eligible work."""
        now = self._now(now_monotonic)
        lifecycle = handoff.lifecycle
        key = QueueKey(
            camera_id=lifecycle.track_key.camera_id,
            session_id=lifecycle.track_key.session_id,
            source_generation_id=lifecycle.track_key.source_generation_id,
            track_id=int(lifecycle.track_key.local_track_id),
        )
        appearance_id = appearance_id_for_track(
            lifecycle.track_key.camera_id,
            lifecycle.track_key.session_id,
            int(lifecycle.track_key.local_track_id),
        )
        state = self._states.setdefault(
            key,
            _TrackState(
                first_seen=lifecycle.first_seen,
                t_detect_monotonic=lifecycle.t_detect_monotonic,
                last_seen=lifecycle.last_seen,
            ),
        )
        state.last_seen = max(state.last_seen, lifecycle.last_seen)
        if lifecycle.kind is LifecycleKind.END:
            return await self._accept_end(handoff, appearance_id, state, now)
        if appearance_id in self._suppressed:
            return self._ack(
                PublicationOutcome.SUPPRESSED,
                appearance_id,
                lifecycle.sequence,
                state,
                detail="quota eviction suppresses this active track",
            )
        if lifecycle.kind is LifecycleKind.START:
            state.first_seen = lifecycle.first_seen
            state.t_detect_monotonic = lifecycle.t_detect_monotonic
        if not await self._is_current_generation(handoff):
            return self._ack(
                PublicationOutcome.STALE_GENERATION,
                appearance_id,
                lifecycle.sequence,
                state,
            )
        if state.committed_version == 0 and not await self._hydrate_state(
            handoff, appearance_id, state
        ):
            return self._ack(
                PublicationOutcome.STALE_GENERATION,
                appearance_id,
                lifecycle.sequence,
                state,
            )
        if handoff.candidate is None:
            return await self._touch_metadata(handoff, key, appearance_id, state, now)
        try:
            work = self._work_from_candidate(handoff, key, appearance_id, state)
        except CropRejectedError as error:
            return self._ack(
                PublicationOutcome.REJECTED,
                appearance_id,
                lifecycle.sequence,
                state,
                detail=str(error),
            )
        if state.rank is not None and state.committed_version > 0:
            elapsed = now - (state.last_upgrade_monotonic or 0.0)
            if not should_upgrade(state.rank, work.rank, elapsed):
                return await self._touch_metadata(handoff, key, appearance_id, state, now)
        decision = self.queue.enqueue(
            PendingEmbedding(
                key=key,
                is_initial=work.representative_version == 1,
                representative_version=work.representative_version,
                jpeg_bytes=work.jpeg_bytes,
                rank=work.rank,
                payload=work,
            )
        )
        if decision in {
            QueueDecision.DROPPED_DUPLICATE,
            QueueDecision.DROPPED_OVERLOAD,
        }:
            return self._ack(
                PublicationOutcome.DROPPED,
                appearance_id,
                lifecycle.sequence,
                state,
                queue_decision=decision,
            )
        self._pending_appearance_ids[key] = appearance_id
        pending = self.queue.pop()
        if pending is None:
            return self._ack(
                PublicationOutcome.QUEUED,
                appearance_id,
                lifecycle.sequence,
                state,
                queue_decision=decision,
            )
        pending_appearance_id = self._pending_appearance_ids.pop(pending.key, None)
        if pending.key != key:
            _ = self.queue.enqueue(pending)
            if pending_appearance_id is not None:
                self._pending_appearance_ids[pending.key] = pending_appearance_id
            return self._ack(
                PublicationOutcome.QUEUED,
                appearance_id,
                lifecycle.sequence,
                state,
                queue_decision=decision,
            )
        return await self._publish(pending)

    async def process_next(self) -> PublicationAck | None:
        """Drain the highest-priority pending representative, if one exists."""
        pending = self.queue.pop()
        if pending is None:
            return None
        _ = self._pending_appearance_ids.pop(pending.key, None)
        return await self._publish(pending)

    def suppress_track(self, appearance_id: AppearanceId) -> None:
        """Suppress new writes for a quota-evicted track until its END event."""
        self._suppressed.add(appearance_id)
        for key, pending_appearance_id in tuple(self._pending_appearance_ids.items()):
            if pending_appearance_id == appearance_id:
                _ = self.queue.discard(key)
                _ = self._pending_appearance_ids.pop(key, None)

    def mark_quota_evicted(self, appearance_id: AppearanceId) -> None:
        """Alias the quota eviction hook used by retention workers."""
        self.suppress_track(appearance_id)

    async def reconcile_orphans(self) -> ReconciliationReport:
        """Delete unreferenced JPEGs and all interrupted temporary crop files."""
        async with self.writer_gate.hold():
            async with self.database.transaction() as session:
                appearance_keys = set(await session.scalars(select(Appearance.crop_object_key)))
                garbage_keys = set(await session.scalars(select(CropGarbage.object_key)))
            referenced = appearance_keys | garbage_keys
            removed_jpegs = 0
            removed_temps = 0
            # Only generated objects are eligible: symlinks and operator files under the
            # crop root are never followed or removed.
            for managed in safe_scan(self.crop_store.root):
                if managed.kind is ManagedFileKind.TEMPORARY:
                    if safe_unlink(self.crop_store.root, managed.object_key):
                        removed_temps += 1
                elif managed.object_key not in referenced and safe_unlink(
                    self.crop_store.root, managed.object_key
                ):
                    removed_jpegs += 1
            return ReconciliationReport(
                removed_jpegs=removed_jpegs,
                removed_temps=removed_temps,
                referenced_jpegs=len(referenced),
            )

    async def _publish(  # noqa: PLR0911
        self, pending: PendingEmbedding[PublicationWork]
    ) -> PublicationAck:
        work = pending.payload
        if work is None:
            raise PublicationError
        if not await self._runtime_generation_matches(work.generation):
            return self._ack(
                PublicationOutcome.STALE_GENERATION,
                work.appearance_id,
                work.sequence,
                self._states[work.key],
            )
        lease = await self._reserve(len(work.jpeg_bytes))
        if lease is None:
            _ = self.queue.enqueue(pending)
            self._pending_appearance_ids[work.key] = work.appearance_id
            state = self._states[work.key]
            return self._ack(
                PublicationOutcome.PAUSED,
                work.appearance_id,
                work.sequence,
                state,
                queue_decision=QueueDecision.ENQUEUED,
                detail="writer budget unavailable",
            )
        stored_key: str | None = None
        published = False
        try:
            try:
                embedding = await self.clip.embed_image(work.jpeg_bytes)
            except (RuntimeError, ValueError, OSError) as error:
                return self._ack(
                    PublicationOutcome.EMBEDDING_FAILED,
                    work.appearance_id,
                    work.sequence,
                    self._states[work.key],
                    detail=str(error),
                )
            stored = self.crop_store.write(work.jpeg_bytes)
            stored_key = stored.object_key
            publication = AppearancePublication(
                appearance_id=work.appearance_id,
                camera_id=work.key.camera_id,
                session_id=work.key.session_id,
                track_id=work.key.track_id,
                first_seen=work.first_seen,
                last_seen=work.last_seen,
                ended_at=work.ended_at,
                representative_version=work.representative_version,
                crop_object_key=stored.object_key,
                bounding_box=BoundingBox(
                    x_min=work.bounding_box[0],
                    y_min=work.bounding_box[1],
                    x_max=work.bounding_box[2],
                    y_max=work.bounding_box[3],
                ),
                source_width=work.source_width,
                source_height=work.source_height,
                detector_confidence=work.detector_confidence,
                crop_quality=work.rank.score,
                byte_size=stored.byte_size,
                embedded_at=self._utc(self._wall_clock()),
                model_id=self.model_id,
                model_revision=self.model_revision,
                embedding=tuple(embedding),
            )
            current = True
            async with self.database.transaction() as session:
                current = await self._verify_generation(session, work.generation)
                if current:
                    _ = await self.storage.publish_appearance(session, publication)
                    published = True
            if not current:
                return self._ack(
                    PublicationOutcome.STALE_GENERATION,
                    work.appearance_id,
                    work.sequence,
                    self._states[work.key],
                )
        except StaleAppearanceVersionError as error:
            return self._ack(
                PublicationOutcome.STALE_VERSION,
                work.appearance_id,
                work.sequence,
                self._states[work.key],
                detail=str(error),
            )
        except (RuntimeError, ValueError, OSError, SQLAlchemyError) as error:
            return self._ack(
                PublicationOutcome.COMMIT_FAILED,
                work.appearance_id,
                work.sequence,
                self._states[work.key],
                detail=str(error),
            )
        finally:
            if stored_key is not None and not published:
                with suppress(OSError):
                    self.crop_store.delete(stored_key)
            await lease.release()
        state = self._states[work.key]
        state.committed_version = work.representative_version
        state.rank = work.rank
        state.last_upgrade_monotonic = self._monotonic_clock()
        state.last_seen = max(state.last_seen, work.last_seen)
        return self._ack(
            PublicationOutcome.PUBLISHED,
            work.appearance_id,
            work.sequence,
            state,
            t_searchable_monotonic=self._monotonic_clock(),
            representative_version=work.representative_version,
        )

    async def _reserve(self, new_crop_bytes: int) -> BudgetLease | None:
        if self.writer_budget is None:
            return None
        gate_acquired = False
        try:
            await self.writer_gate.acquire()
            gate_acquired = True
            async with self.database.transaction() as session:
                relation_sizes = await self.storage.application_relation_sizes(session)
                pending_gc = await session.scalar(
                    select(func.coalesce(func.sum(CropGarbage.byte_size), 0))
                )
                settings = await session.scalar(
                    select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
                )
                snapshot = BudgetSnapshot(
                    new_crop_bytes=new_crop_bytes,
                    physical_crop_bytes=self._physical_crop_bytes(),
                    pending_gc_bytes=int(pending_gc or 0),
                    relation_bytes=sum(relation_sizes.values()),
                    filesystem_free_bytes=shutil.disk_usage(self.crop_store.root).free,
                    quota_bytes=(
                        _DEFAULT_QUOTA_BYTES if settings is None else settings.quota_bytes
                    ),
                    in_flight_reserved_bytes=self.writer_gate.reserved_bytes,
                )
            lease = await self.writer_budget.reserve(snapshot)
            if lease is None:
                self.writer_gate.release()
                gate_acquired = False
                return None
            release_bytes = self.writer_gate.reserve(lease.reserved_bytes)

            async def release() -> None:
                try:
                    await lease.release()
                finally:
                    release_bytes()
                    self.writer_gate.release()

            return BudgetLease(reserved_bytes=lease.reserved_bytes, _release_callback=release)
        except CancelledError:
            if gate_acquired and self.writer_gate.held_by_current_task:
                self.writer_gate.release()
            raise
        except (RuntimeError, ValueError, OSError, SQLAlchemyError):
            if gate_acquired and self.writer_gate.held_by_current_task:
                self.writer_gate.release()
            return None
        except BaseException:
            if gate_acquired and self.writer_gate.held_by_current_task:
                self.writer_gate.release()
            raise

    async def _is_current_generation(self, handoff: PipelineHandoff) -> bool:
        if not await self._runtime_generation_matches(handoff.generation):
            return False
        async with self.database.transaction() as session:
            return await self._verify_generation(session, handoff.generation)

    async def _runtime_generation_matches(self, generation: GenerationBinding) -> bool:
        if self._active_binding is None:
            return True
        active = await self._active_binding(generation.camera_id)
        return active == generation

    async def _hydrate_state(
        self,
        handoff: PipelineHandoff,
        appearance_id: AppearanceId,
        state: _TrackState,
    ) -> bool:
        async with self.database.transaction() as session:
            if not await self._verify_generation(session, handoff.generation):
                return False
            appearance = await session.get(Appearance, UUID(str(appearance_id)))
            if appearance is None:
                return True
            state.committed_version = appearance.representative_version
            state.rank = CandidateRank(
                fully_inside=(
                    appearance.x_min > 0
                    and appearance.y_min > 0
                    and appearance.x_max < appearance.source_width
                    and appearance.y_max < appearance.source_height
                ),
                score=appearance.crop_quality,
            )
            state.first_seen = appearance.first_seen
            state.last_seen = max(state.last_seen, appearance.last_seen)
            return True

    @staticmethod
    async def _verify_generation(session: AsyncSession, generation: GenerationBinding) -> bool:
        camera = await session.scalar(
            select(Camera).where(Camera.id == UUID(str(generation.camera_id))).with_for_update()
        )
        if camera is None:
            return False
        camera_session = await session.scalar(
            select(CameraSession)
            .where(
                CameraSession.camera_id == camera.id,
                CameraSession.ended_at.is_(None),
            )
            .order_by(CameraSession.started_at.desc(), CameraSession.id.desc())
            .with_for_update()
        )
        return bool(
            camera_session is not None
            and camera_session.id == UUID(str(generation.camera_session_id))
            and camera_session.camera_id == UUID(str(generation.camera_id))
            and camera_session.generation_id == UUID(str(generation.db_generation_id))
            and camera.version == generation.camera_version
        )

    @staticmethod
    async def _verify_end_generation(session: AsyncSession, generation: GenerationBinding) -> bool:
        camera = await session.scalar(
            select(Camera).where(Camera.id == UUID(str(generation.camera_id))).with_for_update()
        )
        camera_session = await session.scalar(
            select(CameraSession)
            .where(CameraSession.id == UUID(str(generation.camera_session_id)))
            .with_for_update()
        )
        return bool(
            camera is not None
            and camera_session is not None
            and camera_session.camera_id == UUID(str(generation.camera_id))
            and camera_session.generation_id == UUID(str(generation.db_generation_id))
        )

    def _work_from_candidate(
        self,
        handoff: PipelineHandoff,
        key: QueueKey,
        appearance_id: AppearanceId,
        state: _TrackState,
    ) -> PublicationWork:
        candidate = handoff.candidate
        if candidate is None:
            raise PublicationError
        box = validate_crop_geometry(
            bounding_box=candidate.observation.bounding_box,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
        )
        quality = laplacian_variance(candidate.crop)
        rank = rank_candidate(
            bounding_box=candidate.observation.bounding_box,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            confidence=candidate.observation.confidence,
            laplacian_variance=quality,
        )
        return PublicationWork(
            key=key,
            appearance_id=appearance_id,
            generation=handoff.generation,
            representative_version=max(1, state.committed_version + 1),
            rank=rank,
            jpeg_bytes=encode_rgb_crop(candidate.crop),
            bounding_box=box,
            source_width=candidate.source_width,
            source_height=candidate.source_height,
            detector_confidence=candidate.observation.confidence,
            first_seen=state.first_seen,
            last_seen=max(state.last_seen, handoff.lifecycle.last_seen),
            t_detect_monotonic=state.t_detect_monotonic,
            sequence=handoff.lifecycle.sequence,
        )

    async def _touch_metadata(
        self,
        handoff: PipelineHandoff,
        key: QueueKey,
        appearance_id: AppearanceId,
        state: _TrackState,
        now: float,
    ) -> PublicationAck:
        del key, now
        changed = False
        current = True
        async with self.database.transaction() as session:
            current = await self._verify_generation(session, handoff.generation)
            if current:
                appearance = await session.get(Appearance, UUID(str(appearance_id)))
                if appearance is not None and appearance.tombstoned_at is None:
                    if handoff.lifecycle.last_seen > appearance.last_seen:
                        appearance.last_seen = handoff.lifecycle.last_seen
                        changed = True
                    await session.flush()
        if not current:
            return self._ack(
                PublicationOutcome.STALE_GENERATION,
                appearance_id,
                handoff.lifecycle.sequence,
                state,
            )
        return self._ack(
            PublicationOutcome.METADATA_UPDATED if changed else PublicationOutcome.NOOP,
            appearance_id,
            handoff.lifecycle.sequence,
            state,
            t_searchable_monotonic=self._monotonic_clock() if changed else None,
            representative_version=state.committed_version or None,
        )

    async def _accept_end(
        self,
        handoff: PipelineHandoff,
        appearance_id: AppearanceId,
        state: _TrackState,
        now: float,
    ) -> PublicationAck:
        del now
        current = True
        changed = False
        async with self.database.transaction() as session:
            current = await self._verify_end_generation(session, handoff.generation)
            if current:
                appearance = await session.get(Appearance, UUID(str(appearance_id)))
                if appearance is not None and appearance.tombstoned_at is None:
                    appearance.last_seen = max(appearance.last_seen, handoff.lifecycle.last_seen)
                    ended_at = handoff.lifecycle.ended_at or handoff.lifecycle.last_seen
                    appearance.ended_at = max(ended_at, appearance.last_seen)
                    await session.flush()
                    changed = True
        if current:
            state.ended = True
            self._suppressed.discard(appearance_id)
        if not current:
            return self._ack(
                PublicationOutcome.STALE_GENERATION,
                appearance_id,
                handoff.lifecycle.sequence,
                state,
            )
        return self._ack(
            PublicationOutcome.ENDED if changed else PublicationOutcome.NOOP,
            appearance_id,
            handoff.lifecycle.sequence,
            state,
            t_searchable_monotonic=self._monotonic_clock() if changed else None,
            representative_version=state.committed_version or None,
        )

    def _ack(  # noqa: PLR0913
        self,
        outcome: PublicationOutcome,
        appearance_id: AppearanceId,
        sequence: int,
        state: _TrackState,
        *,
        t_searchable_monotonic: float | None = None,
        representative_version: int | None = None,
        queue_decision: QueueDecision | None = None,
        detail: str | None = None,
    ) -> PublicationAck:
        return PublicationAck(
            outcome=outcome,
            appearance_id=appearance_id,
            sequence=sequence,
            t_detect_monotonic=state.t_detect_monotonic,
            t_searchable_monotonic=t_searchable_monotonic,
            representative_version=representative_version,
            queue_decision=queue_decision,
            detail=detail,
        )

    def _now(self, value: float | None) -> float:
        now = self._monotonic_clock() if value is None else value
        if not isfinite(now) or now < 0:
            raise PublicationInputError
        return now

    def _physical_crop_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.crop_store.root.rglob("*")
            if path.is_file() and path.suffix in {".jpg", ".tmp"}
        )

    @staticmethod
    def _utc(value: datetime) -> datetime:
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


__all__ = [
    "AppearancePublicationService",
    "AppearancePublisher",
    "PublicationAck",
    "PublicationOutcome",
    "PublicationResult",
    "PublicationStatus",
    "PublicationWork",
    "ReconciliationReport",
]


AppearancePublicationService = AppearancePublisher
PublicationResult = PublicationAck
