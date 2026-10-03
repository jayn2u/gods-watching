"""Authenticated camera-event CSV route."""

import time
from datetime import datetime
from typing import Annotated, override
from uuid import UUID

import anyio
from fastapi import APIRouter, Depends, HTTPException, Query
from starlette.background import BackgroundTask
from starlette.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from gods_watching.api.camera_routes import SessionDependencyFactory
from gods_watching.contracts.events import EventExportFilters
from gods_watching.events.export import EventExporter, EventExportService, PreparedCsvExport
from gods_watching.storage import Database


class _PreparedStreamingResponse(StreamingResponse):
    """Keep prepared DB resources owned by the complete HTTP response call."""

    def __init__(self, prepared: PreparedCsvExport) -> None:
        super().__init__(
            prepared.chunks,
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="camera-events.csv"',
                "Cache-Control": "no-store",
            },
            background=BackgroundTask(prepared.aclose),
        )
        self._prepared: PreparedCsvExport = prepared

    @override
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            remaining = (
                None
                if self._prepared.deadline_at is None
                else max(0.0, self._prepared.deadline_at - time.monotonic())
            )
            if remaining is None:
                await super().__call__(scope, receive, send)
            else:
                with anyio.fail_after(remaining):
                    await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):
                await self._prepared.aclose()


def build_event_router(
    *,
    database: Database,
    require_session: SessionDependencyFactory,
    exporter: EventExporter | None = None,
) -> APIRouter:
    """Build the authenticated event export route."""
    router = APIRouter()
    exporter_service = exporter if exporter is not None else EventExportService(database)
    passive_session = require_session(user_action=False)

    async def export_events(
        camera_id: Annotated[UUID | None, Query()] = None,
        since: Annotated[datetime | None, Query()] = None,
        until: Annotated[datetime | None, Query()] = None,
    ) -> Response:
        try:
            filters = EventExportFilters(camera_id=camera_id, since=since, until=until)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        prepared = await exporter_service.prepare(filters)
        return _PreparedStreamingResponse(prepared)

    router.add_api_route(
        "/api/events/export.csv",
        export_events,
        methods=["GET"],
        dependencies=[Depends(passive_session)],
    )
    return router
