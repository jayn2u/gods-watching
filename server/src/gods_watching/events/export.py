"""Bounded CSV export interfaces for camera events."""

from __future__ import annotations

import csv
import io
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, cast
from uuid import UUID

import anyio
from sqlalchemy import text

from gods_watching.contracts.events import EventExportFilters, EventExportRow
from gods_watching.events.repository import EventRepository

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from sqlalchemy.engine import Row
    from sqlalchemy.ext.asyncio import AsyncResult, AsyncSession

    from gods_watching.storage import Database

CSV_HEADER = b"id,occurred_at,event_type,camera_id,camera_name\r\n"
_BATCH_SIZE = 1_000
_FORMULA_PREFIXES = frozenset("=+-@")
_CONTROL_PREFIXES = frozenset(("\t", "\r", "\n"))
_ExportValues = tuple[int, datetime, str, UUID, str]


def _export_row(row: Row[_ExportValues]) -> EventExportRow:
    return EventExportRow(
        id=cast("int", row[0]),
        occurred_at=cast("datetime", row[1]),
        event_type=cast("str", row[2]),
        camera_id=cast("UUID", row[3]),
        camera_name=cast("str", row[4]),
    )


def _safe_string(value: str) -> str:
    leading_trimmed = value.lstrip()
    if value and (
        value[0] in _CONTROL_PREFIXES
        or (leading_trimmed and leading_trimmed[0] in _FORMULA_PREFIXES)
    ):
        return "'" + value
    return value


def encode_csv_row(row: EventExportRow) -> bytes:
    """Encode one UTF-8 CSV row with UTC time and formula-prefix protection."""
    if row.occurred_at.tzinfo is None or row.occurred_at.utcoffset() is None:
        field_name = "occurred_at"
        raise InvalidEventExportRowError(field_name)
    occurred_at = row.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\r\n")
    writer.writerow(
        (
            row.id,
            occurred_at,
            _safe_string(row.event_type),
            str(row.camera_id),
            _safe_string(row.camera_name),
        )
    )
    return output.getvalue().encode("utf-8")


class InvalidEventExportRowError(ValueError):
    """Identify invalid internal values that cannot be encoded unambiguously."""

    def __init__(self, field_name: str) -> None:
        """Store the invalid field name."""
        message = f"{field_name} must include a timezone offset"
        super().__init__(message)


class InvalidExportTimeoutError(ValueError):
    """Reject non-positive internal timeout configuration."""

    def __init__(self) -> None:
        """Format the fixed positive-timeout requirement."""
        message = "export timeouts must be positive"
        super().__init__(message)


class PreparedCsvExport:
    """Own one prepared export stream and its cleanup callback."""

    def __init__(
        self,
        chunks: AsyncIterator[bytes],
        close: Callable[[], Awaitable[None]],
        *,
        deadline_at: float | None = None,
    ) -> None:
        """Store the body iterator, cleanup callback and optional absolute deadline."""
        self.chunks: AsyncIterator[bytes] = chunks
        self._close: Callable[[], Awaitable[None]] = close
        self.deadline_at: float | None = deadline_at
        self._closed: bool = False

    async def aclose(self) -> None:
        """Release the resources owned by this export."""
        if self._closed:
            return
        self._closed = True
        with anyio.CancelScope(shield=True):
            try:
                close_iterator = getattr(self.chunks, "aclose", None)
                if close_iterator is not None:
                    await close_iterator()
            finally:
                await self._close()


class EventExporter(Protocol):
    """Prepare a streaming or benchmark-only buffered export."""

    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        """Initialize an export and return its bounded body iterator."""
        ...


class EventExportService:
    """Production, bounded database-backed event exporter."""

    def __init__(
        self,
        database: Database,
        *,
        statement_timeout_ms: int = 60_000,
        deadline_seconds: float = 300.0,
    ) -> None:
        """Configure fixed production limits, with injectable test deadlines."""
        if statement_timeout_ms <= 0 or deadline_seconds <= 0:
            raise InvalidExportTimeoutError
        self.database: Database = database
        self._statement_timeout_ms: int = statement_timeout_ms
        self._deadline_seconds: float = deadline_seconds

    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        """Initialize a read-only cursor before a response starts."""
        started_at = time.monotonic()
        deadline_at = started_at + self._deadline_seconds
        session: AsyncSession = self.database.session_factory()
        transaction = None
        result: AsyncResult[_ExportValues] | None = None
        closed = False

        async def close_resources() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                if result is not None:
                    await result.close()
            finally:
                try:
                    if transaction is not None and transaction.is_active:
                        await transaction.rollback()
                finally:
                    await session.close()

        try:
            with anyio.fail_after(self._deadline_seconds):
                transaction = await session.begin()
                _ = await session.execute(
                    text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY")
                )
                _ = await session.execute(
                    text(f"SET LOCAL statement_timeout = '{self._statement_timeout_ms}ms'")
                )
                statement = EventRepository.statement(filters).execution_options(
                    yield_per=_BATCH_SIZE
                )
                active_result = await session.stream(statement)
                result = active_result
                partitions = active_result.partitions(_BATCH_SIZE)
                first_partition = await anext(partitions, ())

            async def chunks() -> AsyncIterator[bytes]:
                try:
                    yield CSV_HEADER
                    if first_partition:
                        yield b"".join(encode_csv_row(_export_row(row)) for row in first_partition)
                    async for partition in partitions:
                        yield b"".join(encode_csv_row(_export_row(row)) for row in partition)
                finally:
                    with anyio.CancelScope(shield=True):
                        await close_resources()

            return PreparedCsvExport(chunks(), close_resources, deadline_at=deadline_at)
        except BaseException:
            with anyio.CancelScope(shield=True):
                await close_resources()
            raise
