"""Age/quota retention, transactional tombstones, and retryable crop GC."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Final, final

import anyio
from anyio.to_thread import run_sync
from sqlalchemy import func, select

from gods_watching.appearances.budget import WriterGate, WriterGateProvider
from gods_watching.contracts.identifiers import AppearanceId
from gods_watching.storage import (
    Appearance,
    AppearanceNotFoundError,
    ApplicationSettings,
    CameraSession,
    CropGarbage,
    CropObjectStore,
    Database,
    StorageRepository,
)
from gods_watching.storage.physical_usage import PhysicalUsageError, managed_crop_bytes

from .filesystem import ManagedFileKind, safe_scan, safe_unlink
from .models import (
    FailureInjector,
    FailurePoint,
    QuotaSuppression,
    ReconciliationReport,
    RetentionClock,
    RetentionSettings,
    StorageAccounting,
    SweepReport,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.appearances.budget import WriterBudget

_MAX_GC_ITEMS: Final = 1024
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _SystemClock:
    """Read the current UTC wall clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """Carry one oldest-first appearance and its existing suppression key."""

    appearance_id: UUID
    crop_object_key: str
    first_seen: datetime
    active: bool


@dataclass(frozen=True, slots=True)
class _GcResult:
    """Report one outbox attempt without swallowing a filesystem failure."""

    unlinked: bool
    error: str | None = None


@final
class RetentionService:
    """Run recoverable age/quota eviction against PostgreSQL and crop objects."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        database: Database,
        storage: StorageRepository,
        crop_store: CropObjectStore,
        publisher: QuotaSuppression | None = None,
        writer_budget: WriterBudget | None = None,
        clock: RetentionClock | None = None,
        failure_injector: FailureInjector | None = None,
    ) -> None:
        """Create a service using the shared publication/storage boundaries."""
        self.database = database
        self.storage = storage
        self.crop_store = crop_store
        self.publisher = publisher
        self.writer_budget = writer_budget
        self.writer_gate: WriterGate | None = (
            writer_budget.writer_gate if isinstance(writer_budget, WriterGateProvider) else None
        )
        self.clock = clock or _SystemClock()
        self.failure_injector = failure_injector

    async def reconcile_startup(self) -> ReconciliationReport:
        """Remove unreferenced generated objects and interrupted temp files."""
        async with self._writer_hold():
            async with self.database.transaction() as session:
                referenced = set(await session.scalars(select(Appearance.crop_object_key)))
                referenced.update(await session.scalars(select(CropGarbage.object_key)))
            await self._restore_active_suppression()
            removed_jpegs = 0
            removed_temps = 0
            skipped_files = 0
            for managed in safe_scan(self.crop_store.root):
                if managed.kind is ManagedFileKind.TEMPORARY:
                    if safe_unlink(self.crop_store.root, managed.object_key):
                        removed_temps += 1
                elif managed.object_key not in referenced:
                    if safe_unlink(self.crop_store.root, managed.object_key):
                        removed_jpegs += 1
                else:
                    skipped_files += 1
            return ReconciliationReport(
                removed_jpegs=removed_jpegs,
                removed_temps=removed_temps,
                referenced_files=len(referenced),
                skipped_files=skipped_files,
            )

    async def accounting(self) -> StorageAccounting:
        """Read physical crop, pending outbox, relation, and free-space totals."""
        async with self._writer_hold():
            async with self.database.transaction() as session:
                settings = await self._settings(session)
                relations = await self.storage.application_relation_sizes(session)
                pending = await session.scalar(
                    select(func.coalesce(func.sum(CropGarbage.byte_size), 0))
                )
            physical = await run_sync(
                managed_crop_bytes,
                self.crop_store.root,
                abandon_on_cancel=False,
            )
            return StorageAccounting(
                physical_crop_bytes=physical,
                pending_gc_bytes=int(pending or 0),
                relation_bytes=sum(relations.values()),
                filesystem_free_bytes=shutil.disk_usage(self.crop_store.root).free,
                quota_bytes=settings.quota_bytes,
                cleanup_threshold=settings.cleanup_threshold,
                minimum_free_bytes=settings.minimum_free_bytes,
            )

    async def sweep(self, now: datetime | None = None) -> SweepReport:
        """Evict expired/oldest representatives and drain retryable GC work."""
        observed = _utc(now or self.clock.now())
        async with self._writer_hold():
            return await self._sweep(observed)

    async def _sweep(self, observed: datetime) -> SweepReport:
        await self._restore_active_suppression()
        before = await self.accounting()
        current = before
        errors: list[str] = []
        unlinked, gc_failures = await self._drain_gc(errors)
        if unlinked or gc_failures:
            current = await self.accounting()
        _, age_candidates = await self._age_candidates(observed)
        tombstoned = 0
        suppressed = 0
        for candidate in age_candidates:
            if await self._tombstone(candidate):
                tombstoned += 1
        more_unlinked, more_failures = await self._drain_gc(errors)
        unlinked += more_unlinked
        gc_failures += more_failures
        if tombstoned or more_unlinked or more_failures:
            current = await self.accounting()
        quota_candidates = 0
        if current.cleanup_required or current.filesystem_guard_active:
            for candidate in await self._all_candidates():
                if (
                    current.managed_bytes < current.threshold_bytes
                    and not current.filesystem_guard_active
                ):
                    break
                if not await self._tombstone(candidate):
                    continue
                quota_candidates += 1
                tombstoned += 1
                if candidate.active and self.publisher is not None:
                    suppressed += 1
                drained, failures = await self._drain_gc(errors)
                unlinked += drained
                gc_failures += failures
                current = await self.accounting()
        storage_full = (
            current.managed_bytes >= current.quota_bytes or current.filesystem_guard_active
        )
        return SweepReport(
            age_candidates=len(age_candidates),
            quota_candidates=quota_candidates,
            tombstoned=tombstoned,
            unlinked=unlinked,
            gc_failures=gc_failures,
            suppressed_tracks=suppressed,
            managed_bytes_before=before.managed_bytes,
            managed_bytes_after=current.managed_bytes,
            relation_bytes_after=current.relation_bytes,
            physical_crop_bytes_after=current.physical_crop_bytes,
            pending_gc_bytes_after=current.pending_gc_bytes,
            filesystem_free_bytes_after=current.filesystem_free_bytes,
            storage_full=storage_full,
            errors=tuple(errors),
        )

    async def run_forever(self, stop: anyio.Event) -> None:
        """Reconcile once and run the prescribed ten-second scheduler."""
        _ = await self.reconcile_startup()
        while True:
            settings = await self._settings_snapshot()
            try:
                _ = await self.sweep()
            except PhysicalUsageError:
                _LOGGER.warning(
                    "retention sweep skipped because physical crop accounting is unavailable"
                )
            with anyio.move_on_after(settings.sweep_interval_seconds) as scope:
                await stop.wait()
            if not scope.cancelled_caught:
                return

    async def _settings_snapshot(self) -> RetentionSettings:
        async with self.database.transaction() as session:
            return await self._settings(session)

    async def _settings(self, session: AsyncSession) -> RetentionSettings:
        row = await session.scalar(
            select(ApplicationSettings).where(ApplicationSettings.singleton.is_(True))
        )
        return RetentionSettings.from_values(
            retention_days=None if row is None else row.retention_days,
            quota_bytes=None if row is None else row.quota_bytes,
        )

    @asynccontextmanager
    async def _writer_hold(self) -> AsyncIterator[None]:
        if self.writer_gate is None:
            yield
            return
        async with self.writer_gate.hold():
            yield

    async def _age_candidates(
        self, now: datetime
    ) -> tuple[RetentionSettings, tuple[_Candidate, ...]]:
        async with self.database.transaction() as session:
            settings = await self._settings(session)
            cutoff = now - timedelta(days=settings.retention_days)
            rows = await self._candidate_rows(session, first_seen_on_or_before=cutoff)
        return settings, rows

    async def _restore_active_suppression(self) -> None:
        if self.publisher is None:
            return
        async with self.database.transaction() as session:
            statement = (
                select(Appearance, CameraSession)
                .join(CameraSession, CameraSession.id == Appearance.session_id)
                .where(
                    Appearance.tombstoned_at.is_not(None),
                    Appearance.ended_at.is_(None),
                )
            )
            rows = (await session.execute(statement)).tuples().all()
        for appearance, _camera_session in rows:
            self.publisher.mark_quota_evicted(AppearanceId(appearance.id))

    async def _all_candidates(self) -> tuple[_Candidate, ...]:
        async with self.database.transaction() as session:
            return await self._candidate_rows(session)

    async def _candidate_rows(
        self,
        session: AsyncSession,
        *,
        first_seen_on_or_before: datetime | None = None,
    ) -> tuple[_Candidate, ...]:
        statement = select(
            Appearance.id,
            Appearance.crop_object_key,
            Appearance.first_seen,
            Appearance.ended_at,
        ).where(Appearance.tombstoned_at.is_(None))
        if first_seen_on_or_before is not None:
            statement = statement.where(Appearance.first_seen <= first_seen_on_or_before)
        statement = statement.order_by(Appearance.first_seen.asc(), Appearance.id.asc())
        candidates: list[_Candidate] = []
        for appearance_id, crop_object_key, first_seen, ended_at in (
            (await session.execute(statement)).tuples().all()
        ):
            candidates.append(
                _Candidate(
                    appearance_id=appearance_id,
                    crop_object_key=crop_object_key,
                    first_seen=first_seen,
                    active=ended_at is None,
                )
            )
        return tuple(candidates)

    async def _tombstone(self, candidate: _Candidate) -> bool:
        if self.failure_injector is not None:
            self.failure_injector.check(FailurePoint.BEFORE_TOMBSTONE, candidate.crop_object_key)
        try:
            async with self.database.transaction() as session:
                await self.storage.tombstone_appearance(session, candidate.appearance_id)
        except AppearanceNotFoundError:
            return False
        if self.failure_injector is not None:
            self.failure_injector.check(FailurePoint.AFTER_TOMBSTONE, candidate.crop_object_key)
        if candidate.active and self.publisher is not None:
            self.publisher.mark_quota_evicted(AppearanceId(candidate.appearance_id))
        return True

    async def _drain_gc(self, errors: list[str]) -> tuple[int, int]:
        unlinked = 0
        failures = 0
        for _ in range(_MAX_GC_ITEMS):
            result = await self.collect_one_gc()
            if result is None:
                break
            if result.error is not None:
                failures += 1
                errors.append(result.error)
                break
            if result.unlinked:
                unlinked += 1
        return unlinked, failures

    async def collect_one_gc(self) -> _GcResult | None:
        """Collect one retryable crop-garbage item and finalize safe metadata."""
        async with self.database.transaction() as session:
            row = await session.scalar(
                select(CropGarbage)
                .order_by(CropGarbage.enqueued_at.asc(), CropGarbage.id.asc())
                .limit(1)
                .with_for_update()
            )
            if row is None:
                return None
            try:
                if self.failure_injector is not None:
                    self.failure_injector.check(FailurePoint.BEFORE_UNLINK, row.object_key)
                _ = safe_unlink(self.crop_store.root, row.object_key)
                if self.failure_injector is not None:
                    self.failure_injector.check(FailurePoint.AFTER_UNLINK, row.object_key)
            except (OSError, ValueError) as error:
                row.attempts += 1
                row.last_error = str(error)
                await session.flush()
                return _GcResult(unlinked=False, error=str(error))
            await session.delete(row)
            await session.flush()
            if row.appearance_id is not None:
                _ = await self.storage.finalize_tombstoned_appearance(
                    session,
                    appearance_id=row.appearance_id,
                )
            return _GcResult(unlinked=True)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = ["RetentionService"]
