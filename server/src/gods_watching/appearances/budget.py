"""Writer reservation contracts for bounded appearance persistence."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


class BudgetConfigurationError(ValueError):
    """Describe an invalid writer headroom configuration."""


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    """Describe the complete managed-storage cost of one prospective crop."""

    new_crop_bytes: int
    physical_crop_bytes: int
    pending_gc_bytes: int
    relation_bytes: int
    filesystem_free_bytes: int
    quota_bytes: int
    in_flight_reserved_bytes: int = 0

    @property
    def projected_bytes(self) -> int:
        """Return managed bytes after adding the prospective crop.

        ``physical_crop_bytes`` is an authoritative filesystem measurement and
        already includes crops referenced by ``crop_gc``. ``pending_gc_bytes``
        is retained as an audit field and must not be added a second time.
        """
        return (
            self.physical_crop_bytes
            + self.relation_bytes
            + self.new_crop_bytes
            + self.in_flight_reserved_bytes
        )


class WriterGate:
    """Serialize accounting, object mutations, and retention cleanup.

    The gate is re-entrant for one task so a concrete budget can be used by a
    caller that already acquired it before taking its fresh storage snapshot.
    ``reserved_bytes`` tracks accepted leases which have not yet materialized
    in the filesystem or database.
    """

    def __init__(self) -> None:
        """Create an unlocked gate with no active reservations."""
        self._lock: asyncio.Lock = asyncio.Lock()
        self._owner: asyncio.Task[object] | None = None
        self._depth: int = 0
        self._reserved_bytes: int = 0

    @property
    def reserved_bytes(self) -> int:
        """Return bytes reserved by accepted but unfinished leases."""
        return self._reserved_bytes

    @property
    def held_by_current_task(self) -> bool:
        """Report whether the current task owns this gate."""
        return self._owner is asyncio.current_task()

    async def acquire(self) -> None:
        """Acquire the gate, waiting for another writer or retention pass."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError
        if self._owner is task:
            self._depth += 1
            return
        _ = await self._lock.acquire()
        self._owner = task
        self._depth = 1

    def release(self) -> None:
        """Release one acquisition held by the current task."""
        if self._owner is not asyncio.current_task() or self._depth == 0:
            raise RuntimeError
        self._release_unchecked()

    def release_unchecked(self) -> None:
        """Release a lease-owned acquisition from its eventual caller."""
        if self._depth == 0:
            raise RuntimeError
        self._release_unchecked()

    def _release_unchecked(self) -> None:
        """Release the underlying lock after ownership has been validated."""
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()

    @asynccontextmanager
    async def hold(self) -> AsyncIterator[None]:
        """Hold the gate across retention or reconciliation work."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def reserve(self, amount: int) -> Callable[[], None]:
        """Account one accepted lease and return its idempotent release hook."""
        if amount < 0:
            raise ValueError
        self._reserved_bytes += amount
        released = False

        def release() -> None:
            nonlocal released
            if released:
                return
            released = True
            self._reserved_bytes -= amount

        return release


@dataclass(slots=True)
class BudgetLease:
    """Represent a writer reservation that is released after commit or cleanup."""

    reserved_bytes: int
    _release_callback: Callable[[], Awaitable[None]] | None = field(
        default=None, repr=False, compare=False
    )
    _released: bool = field(default=False, repr=False, compare=False)

    async def release(self) -> None:
        """Release a reservation exactly once, including its writer gate."""
        if self._released:
            return
        self._released = True
        if self._release_callback is not None:
            await self._release_callback()


class WriterBudget(Protocol):
    """Reserve a complete crop write before the filesystem mutation begins."""

    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease | None:
        """Return a lease when the snapshot fits, otherwise pause persistence."""
        ...


@runtime_checkable
class WriterGateProvider(Protocol):
    """Expose the shared gate used by publication and retention workers."""

    @property
    def writer_gate(self) -> WriterGate:
        """Expose the shared writer gate."""
        ...


@final
class ConservativeWriterBudget:
    """Apply a 95% quota and five-GiB free-space guard to one writer."""

    def __init__(self, *, minimum_free_bytes: int = 5 * 1024**3) -> None:
        """Configure the minimum free filesystem headroom."""
        if minimum_free_bytes < 0:
            raise BudgetConfigurationError
        self._minimum_free_bytes: int = minimum_free_bytes
        self._writer_gate = WriterGate()

    @property
    def writer_gate(self) -> WriterGate:
        """Return the gate shared with publication and retention cleanup."""
        return self._writer_gate

    async def reserve(self, snapshot: BudgetSnapshot) -> BudgetLease | None:
        """Reserve only when quota and filesystem headroom remain available.

        Direct callers are also serialized. The publisher acquires the gate
        before its fresh accounting read, in which case this method reuses
        that ownership and its returned lease only releases the byte counter;
        the publisher releases the outer gate after object/DB cleanup.
        """
        acquired_here = not self._writer_gate.held_by_current_task
        gate_acquired = False
        if acquired_here:
            await self._writer_gate.acquire()
            gate_acquired = True
        try:
            quota_limit = snapshot.quota_bytes * 95 // 100
            if snapshot.new_crop_bytes < 0 or snapshot.projected_bytes > quota_limit:
                if gate_acquired:
                    self._writer_gate.release()
                    gate_acquired = False
                return None
            if snapshot.filesystem_free_bytes - snapshot.new_crop_bytes < self._minimum_free_bytes:
                if gate_acquired:
                    self._writer_gate.release()
                    gate_acquired = False
                return None

            async def release() -> None:
                if gate_acquired:
                    self._writer_gate.release_unchecked()

            return BudgetLease(
                reserved_bytes=snapshot.new_crop_bytes,
                _release_callback=release,
            )
        except BaseException:
            if gate_acquired:
                self._writer_gate.release()
            raise


__all__ = [
    "BudgetConfigurationError",
    "BudgetLease",
    "BudgetSnapshot",
    "ConservativeWriterBudget",
    "WriterBudget",
    "WriterGate",
    "WriterGateProvider",
]
