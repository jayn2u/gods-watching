from __future__ import annotations

import time
from contextlib import suppress
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID

import anyio
import pytest
from fastapi import FastAPI, HTTPException

from gods_watching.api.camera_routes import AuthenticatedRequest
from gods_watching.api.event_routes import build_event_router
from gods_watching.contracts.events import EventExportFilters
from gods_watching.events.export import PreparedCsvExport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from starlette.types import Message, Scope

    from gods_watching.api.camera_routes import SessionDependencyFactory
    from gods_watching.events.export import EventExporter
    from gods_watching.storage import Database


class _SyntheticPostHeaderFailureError(RuntimeError):
    """Signal a body iteration failure after the response has started."""


class _SyntheticDisconnectError(RuntimeError):
    """Signal a client disconnect before body iteration begins."""


class _Exporter:
    def __init__(
        self,
        chunks: AsyncIterator[bytes] | None = None,
        *,
        deadline_at: float | None = None,
    ) -> None:
        self.filters: list[EventExportFilters] = []
        self.close_count: int = 0
        self._chunks: AsyncIterator[bytes] | None = chunks
        self._deadline_at: float | None = deadline_at

    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        self.filters.append(filters)

        async def content() -> AsyncIterator[bytes]:
            if self._chunks is None:
                yield b"id,occurred_at,event_type,camera_id,camera_name\r\n"
                yield (
                    b"1,2026-10-03T12:00:00Z,camera.created,"
                    b"00000000-0000-0000-0000-000000000001,Camera\r\n"
                )
                return
            async for chunk in self._chunks:
                yield chunk

        async def close() -> None:
            self.close_count += 1

        return PreparedCsvExport(content(), close, deadline_at=self._deadline_at)


def _guard_factory(*, authenticated: bool) -> SessionDependencyFactory:
    def require_session(*, user_action: bool) -> Callable[..., Awaitable[AuthenticatedRequest]]:
        assert user_action is False

        async def guard() -> AuthenticatedRequest:
            if not authenticated:
                raise HTTPException(status_code=401, detail="authentication required")
            return AuthenticatedRequest(session_id="synthetic-session")

        return guard

    return require_session


async def _request(
    app: FastAPI,
    path: str,
    query: bytes = b"",
) -> tuple[list[Message], Exception | None]:
    messages: list[Message] = []
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await anyio.sleep_forever()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 80),
    }
    failure: Exception | None = None
    try:
        await app(scope, receive, send)
    except (RuntimeError, TimeoutError) as error:
        failure = error
    return messages, failure


async def _disconnect_before_body(app: FastAPI) -> list[Message]:
    messages: list[Message] = []
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)
        if message["type"] == "http.response.start":
            raise _SyntheticDisconnectError

    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/api/events/export.csv",
        "raw_path": b"/api/events/export.csv",
        "query_string": b"",
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 80),
    }
    with suppress(Exception):
        await app(scope, receive, send)
    return messages


def _app(exporter: EventExporter, *, authenticated: bool = True) -> FastAPI:
    app = FastAPI()
    app.include_router(
        build_event_router(
            database=cast("Database", object()),
            require_session=_guard_factory(authenticated=authenticated),
            exporter=exporter,
        )
    )
    return app


def _response(messages: list[Message]) -> tuple[int, dict[bytes, bytes], bytes]:
    start = next(message for message in messages if message["type"] == "http.response.start")
    raw_headers = cast("list[tuple[bytes, bytes]]", start["headers"])
    headers: dict[bytes, bytes] = dict(raw_headers)
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return start["status"], headers, body


def test_export_requires_the_existing_session_guard() -> None:
    exporter = _Exporter()
    messages, failure = anyio.run(
        _request, _app(exporter, authenticated=False), "/api/events/export.csv"
    )
    status, _, _ = _response(messages)

    assert failure is None
    assert status == 401
    assert exporter.filters == []


def test_authenticated_export_is_an_attachment_and_passes_filters() -> None:
    exporter = _Exporter()
    camera_id = UUID("00000000-0000-0000-0000-000000000001")
    messages, failure = anyio.run(
        _request,
        _app(exporter),
        "/api/events/export.csv",
        (
            f"camera_id={camera_id}&since=2026-10-01T00%3A00%3A00%2B00%3A00"
            "&until=2026-10-04T00%3A00%3A00%2B00%3A00"
        ).encode(),
    )
    status, headers, body = _response(messages)

    assert failure is None
    assert status == 200
    assert headers[b"content-type"] == b"text/csv; charset=utf-8"
    assert headers[b"content-disposition"] == b'attachment; filename="camera-events.csv"'
    assert headers[b"cache-control"] == b"no-store"
    assert body.startswith(b"id,occurred_at,event_type,camera_id,camera_name\r\n")
    assert exporter.filters == [
        EventExportFilters(
            camera_id=camera_id,
            since=datetime(2026, 10, 1, tzinfo=UTC),
            until=datetime(2026, 10, 4, tzinfo=UTC),
        )
    ]
    assert exporter.close_count == 1


@pytest.mark.parametrize(
    "query",
    [
        b"camera_id=bad-uuid",
        b"since=2026-10-03T00%3A00%3A00",
        b"since=2026-10-04T00%3A00%3A00%2B00%3A00&until=2026-10-03T00%3A00%3A00%2B00%3A00",
    ],
)
def test_invalid_filters_are_rejected_before_exporter_use(query: bytes) -> None:
    exporter = _Exporter()
    messages, failure = anyio.run(
        _request,
        _app(exporter),
        "/api/events/export.csv",
        query,
    )
    status, _, _ = _response(messages)

    assert failure is None
    assert status == 422
    assert exporter.filters == []


def test_response_closes_export_when_stream_fails_after_first_chunk() -> None:
    async def broken_chunks() -> AsyncIterator[bytes]:
        yield b"id,occurred_at,event_type,camera_id,camera_name\r\n"
        raise _SyntheticPostHeaderFailureError

    exporter = _Exporter(broken_chunks())
    messages, failure = anyio.run(
        _request,
        _app(exporter),
        "/api/events/export.csv",
    )
    status, _, body = _response(messages)

    assert isinstance(failure, _SyntheticPostHeaderFailureError)
    assert status == 200
    assert body == b"id,occurred_at,event_type,camera_id,camera_name\r\n"
    assert exporter.close_count == 1


def test_response_closes_export_if_disconnect_prevents_iteration() -> None:
    exporter = _Exporter()
    messages = anyio.run(_disconnect_before_body, _app(exporter))

    assert any(message["type"] == "http.response.start" for message in messages)
    assert not any(message["type"] == "http.response.body" for message in messages)
    assert exporter.close_count == 1


def test_response_deadline_closes_a_slow_export() -> None:
    async def slow_chunks() -> AsyncIterator[bytes]:
        yield b"id,occurred_at,event_type,camera_id,camera_name\r\n"
        await anyio.sleep_forever()

    exporter = _Exporter(slow_chunks(), deadline_at=time.monotonic() + 0.03)
    messages, failure = anyio.run(
        _request,
        _app(exporter),
        "/api/events/export.csv",
    )
    status, _, body = _response(messages)

    assert isinstance(failure, TimeoutError)
    assert status == 200
    assert body == b"id,occurred_at,event_type,camera_id,camera_name\r\n"
    assert exporter.close_count == 1
