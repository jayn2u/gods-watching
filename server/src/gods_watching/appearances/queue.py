"""Bounded pending CLIP embedding work with first-representative priority."""

from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.media.models import SourceGenerationId

from .policy import CandidateRank

_DEFAULT_MAX_PENDING: Final = 256


@dataclass(frozen=True, slots=True)
class QueueKey:
    """Identify one continuous camera/session/local-track appearance."""

    camera_id: CameraId
    session_id: CameraSessionId
    source_generation_id: SourceGenerationId
    track_id: int


@dataclass(frozen=True, slots=True)
class PendingEmbedding[T]:
    """Carry encoded JPEG work without retaining a source frame."""

    key: QueueKey
    is_initial: bool
    representative_version: int
    jpeg_bytes: bytes = b""
    rank: CandidateRank | None = None
    payload: T | None = None


class QueueDecision(StrEnum):
    """Describe how a queue submission changed bounded pending work."""

    ENQUEUED = "enqueued"
    EVICTED_UPGRADE = "evicted_upgrade"
    REPLACED_BY_FIRST = "replaced_by_first"
    REPLACED_UPGRADE = "replaced_upgrade"
    DROPPED_DUPLICATE = "dropped_duplicate"
    DROPPED_OVERLOAD = "dropped_overload"


class QueueErrorCode(StrEnum):
    """Name invalid queue configuration or work."""

    MAX_PENDING = "max_pending_invalid"
    VERSION = "representative_version_invalid"


@dataclass(frozen=True, slots=True)
class QueueError(ValueError):
    """Describe a queue boundary violation."""

    code: QueueErrorCode


class PendingEmbeddingQueue[T]:
    """Keep at most one item per key and never let upgrades displace firsts."""

    def __init__(self, *, max_pending: int = _DEFAULT_MAX_PENDING) -> None:
        """Create a bounded queue with separate first and upgrade lanes."""
        if max_pending <= 0:
            raise QueueError(QueueErrorCode.MAX_PENDING)
        self._max_pending: int = max_pending
        self._firsts: deque[PendingEmbedding[T]] = deque()
        self._upgrades: deque[PendingEmbedding[T]] = deque()
        self._by_key: dict[QueueKey, PendingEmbedding[T]] = {}

    def __len__(self) -> int:
        """Return the count of unique pending track keys."""
        return len(self._by_key)

    @property
    def max_pending(self) -> int:
        """Return the configured global bound."""
        return self._max_pending

    def enqueue(self, item: PendingEmbedding[T]) -> QueueDecision:
        """Add one item while preserving first-representative priority."""
        if item.representative_version <= 0:
            raise QueueError(QueueErrorCode.VERSION)
        current = self._by_key.get(item.key)
        if current is not None:
            return self._replace_existing(current, item)
        if len(self) >= self._max_pending:
            return self._enqueue_at_capacity(item)
        self._insert(item)
        return QueueDecision.ENQUEUED

    def _replace_existing(
        self, current: PendingEmbedding[T], item: PendingEmbedding[T]
    ) -> QueueDecision:
        if current.is_initial:
            return QueueDecision.REPLACED_BY_FIRST
        if item.is_initial:
            self._remove(current)
            self._insert(item)
            return QueueDecision.REPLACED_BY_FIRST
        if item.rank is not None and current.rank is not None and item.rank > current.rank:
            self._remove(current)
            self._insert(item)
            return QueueDecision.REPLACED_UPGRADE
        return QueueDecision.DROPPED_DUPLICATE

    def _enqueue_at_capacity(self, item: PendingEmbedding[T]) -> QueueDecision:
        if item.is_initial and self._upgrades:
            self._remove(self._upgrades[0])
            self._insert(item)
            return QueueDecision.EVICTED_UPGRADE
        return QueueDecision.DROPPED_OVERLOAD

    def pop(self) -> PendingEmbedding[T] | None:
        """Pop a first representative before any upgrade work."""
        if self._firsts:
            item = self._firsts.popleft()
        elif self._upgrades:
            item = self._upgrades.popleft()
        else:
            return None
        _ = self._by_key.pop(item.key)
        return item

    def discard(self, key: QueueKey) -> PendingEmbedding[T] | None:
        """Remove pending work for a suppressed track without disturbing other lanes."""
        item = self._by_key.get(key)
        if item is None:
            return None
        self._remove(item)
        return item

    def _insert(self, item: PendingEmbedding[T]) -> None:
        self._by_key[item.key] = item
        (self._firsts if item.is_initial else self._upgrades).append(item)

    def _remove(self, item: PendingEmbedding[T]) -> None:
        lane = self._firsts if item.is_initial else self._upgrades
        lane.remove(item)
        _ = self._by_key.pop(item.key)
