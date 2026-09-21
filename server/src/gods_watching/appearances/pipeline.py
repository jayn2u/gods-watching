"""Concrete ingest handoff consumer for appearance publication."""

from dataclasses import dataclass
from typing import final

import anyio

from gods_watching.contracts.pipeline import PipelineHandoff

from .publication import AppearancePublisher, PublicationAck, ReconciliationReport


@dataclass(frozen=True, slots=True)
class PublicationStatsSnapshot:
    """Expose the bounded indexing backlog and latest searchable latency."""

    pending_embeddings: int
    last_searchable_latency_seconds: float | None


@final
class AppearanceHandoffConsumer:
    """Serialize ingest handoffs and reconcile crop storage before first use."""

    def __init__(self, publisher: AppearancePublisher) -> None:
        """Bind one publisher to the callable ingest-consumer surface."""
        self._publisher = publisher
        self._lock = anyio.Lock()
        self._reconciliation: ReconciliationReport | None = None
        self._last_ack: PublicationAck | None = None
        self._last_searchable_latency_seconds: float | None = None

    @property
    def last_ack(self) -> PublicationAck | None:
        """Return the most recent durable publication decision."""
        return self._last_ack

    @property
    def stats(self) -> PublicationStatsSnapshot:
        """Return a current publication telemetry snapshot."""
        return PublicationStatsSnapshot(
            pending_embeddings=self._publisher.pending_embeddings,
            last_searchable_latency_seconds=self._last_searchable_latency_seconds,
        )

    async def start(self) -> ReconciliationReport:
        """Reconcile interrupted crop writes exactly once before consumption."""
        async with self._lock:
            if self._reconciliation is None:
                self._reconciliation = await self._publisher.reconcile_orphans()
            return self._reconciliation

    async def consume(self, handoff: PipelineHandoff) -> PublicationAck:
        """Publish one ordered lifecycle handoff and retain its acknowledgement."""
        _ = await self.start()
        async with self._lock:
            acknowledgement = await self._publisher.accept_handoff(handoff)
            self._record(acknowledgement)
            return acknowledgement

    async def drain_one(self) -> PublicationAck | None:
        """Retry the highest-priority queued publication once."""
        async with self._lock:
            acknowledgement = await self._publisher.process_next()
            if acknowledgement is not None:
                self._record(acknowledgement)
            return acknowledgement

    def _record(self, acknowledgement: PublicationAck) -> None:
        self._last_ack = acknowledgement
        if acknowledgement.t_searchable_monotonic is not None:
            self._last_searchable_latency_seconds = max(
                0.0,
                acknowledgement.t_searchable_monotonic - acknowledgement.t_detect_monotonic,
            )

    async def __call__(self, handoff: PipelineHandoff) -> None:
        """Adapt publication to the Task 10 PipelineHandoffConsumer signature."""
        _ = await self.consume(handoff)


__all__ = ["AppearanceHandoffConsumer", "PublicationStatsSnapshot"]
