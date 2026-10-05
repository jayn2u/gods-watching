"""Benchmark-only full-buffer implementation of the event CSV exporter."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, override
from uuid import UUID

import anyio
from sqlalchemy import text

from gods_watching.contracts.events import EventExportFilters, EventExportRow
from gods_watching.events.export import (
    CSV_HEADER,
    EventExporter,
    PreparedCsvExport,
    encode_csv_row,
)
from gods_watching.events.repository import EventRepository

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.ext.asyncio import AsyncSession

    from gods_watching.storage import Database

_ExportValues = tuple[int, datetime, str, UUID, str]


def _as_export_row(values: _ExportValues) -> EventExportRow:
    return EventExportRow(
        id=values[0],
        occurred_at=values[1],
        event_type=values[2],
        camera_id=values[3],
        camera_name=values[4],
    )


class BufferedEventExporter(EventExporter):
    """Read, encode, and retain the complete CSV body before HTTP sends bytes."""

    database: Database
    statement_timeout_ms: int
    deadline_seconds: float

    def __init__(
        self,
        database: Database,
        *,
        statement_timeout_ms: int = 60_000,
        deadline_seconds: float = 300.0,
    ) -> None:
        """Configure the buffered benchmark exporter and its request limits."""
        if statement_timeout_ms <= 0 or deadline_seconds <= 0:
            error_message = "export timeouts must be positive"
            raise ValueError(error_message)
        self.database = database
        self.statement_timeout_ms = statement_timeout_ms
        self.deadline_seconds = deadline_seconds

    @override
    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        """Read all matching events and return a one-chunk CSV response."""
        started_at = time.monotonic()
        deadline_at = started_at + self.deadline_seconds
        session: AsyncSession = self.database.session_factory()
        body = bytearray(CSV_HEADER)
        try:
            with anyio.fail_after(self.deadline_seconds):
                transaction = await session.begin()
                try:
                    _ = await session.execute(
                        text("SET TRANSACTION ISOLATION LEVEL READ COMMITTED READ ONLY")
                    )
                    _ = await session.execute(
                        text(f"SET LOCAL statement_timeout = '{self.statement_timeout_ms}ms'")
                    )
                    result = (await session.execute(EventRepository.statement(filters))).tuples()
                    for row in result:
                        body.extend(encode_csv_row(_as_export_row(row)))
                finally:
                    if transaction.is_active:
                        await transaction.rollback()
        except BaseException:
            with anyio.CancelScope(shield=True):
                await session.close()
            raise
        await session.close()

        async def chunks() -> AsyncIterator[bytes]:
            yield bytes(body)

        async def close() -> None:
            return None

        return PreparedCsvExport(chunks(), close, deadline_at=deadline_at)
