"""Benchmark-only full-buffer implementation of the event CSV exporter."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, cast
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

    from sqlalchemy.engine import Row
    from sqlalchemy.ext.asyncio import AsyncResult, AsyncSession

    from gods_watching.storage import Database

_ExportValues = tuple[int, datetime, str, UUID, str]


def _as_export_row(row: Row[_ExportValues]) -> EventExportRow:
    occurred_at = row[1]
    return EventExportRow(
        id=cast("int", row[0]),
        occurred_at=cast("datetime", occurred_at),
        event_type=cast("str", row[2]),
        camera_id=cast("UUID", row[3]),
        camera_name=cast("str", row[4]),
    )


class BufferedEventExporter(EventExporter):
    """Read, encode, and retain the complete CSV body before HTTP sends bytes."""

    def __init__(
        self,
        database: Database,
        *,
        statement_timeout_ms: int = 60_000,
        deadline_seconds: float = 300.0,
    ) -> None:
        if statement_timeout_ms <= 0 or deadline_seconds <= 0:
            error_message = "export timeouts must be positive"
            raise ValueError(error_message)
        self.database = database
        self.statement_timeout_ms = statement_timeout_ms
        self.deadline_seconds = deadline_seconds

    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
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
                    result: AsyncResult[_ExportValues] = await session.execute(
                        EventRepository.statement(filters)
                    )
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
