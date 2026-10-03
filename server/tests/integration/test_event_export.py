from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import anyio
import pytest
from fastapi import FastAPI
from sqlalchemy import event, func, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from gods_watching.api.camera_routes import AuthenticatedRequest
from gods_watching.api.event_routes import build_event_router
from gods_watching.contracts.events import EventExportFilters
from gods_watching.events import EventRepository
from gods_watching.events.export import CSV_HEADER, EventExportService
from gods_watching.storage import CameraEvent, Database

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine
    from sqlalchemy.pool import QueuePool
    from starlette.types import Message, Scope

    from gods_watching.api.camera_routes import SessionDependency


class _StatementConstructionFailureError(RuntimeError):
    """Represent a failure while building the export query."""


async def _seed_events(
    engine: AsyncEngine,
    camera_id: UUID,
    count: int,
) -> tuple[int, ...]:
    async with engine.begin() as connection:
        # The returned cursor isn't needed for fixture seeding.
        _ = await connection.execute(
            insert(CameraEvent),
            [
                {
                    "event_type": "camera.updated",
                    "camera_id": camera_id,
                    "camera_name": f"Snapshot {index:04d}",
                }
                for index in range(count)
            ],
        )
        ids = await connection.scalars(
            select(CameraEvent.id)
            .where(CameraEvent.camera_id == camera_id)
            .order_by(CameraEvent.id)
        )
        return tuple(ids.all())


def _database(engine: AsyncEngine) -> Database:
    return Database(engine, async_sessionmaker(engine, expire_on_commit=False))


def _checked_out(database: Database) -> int:
    return cast("QueuePool", database.engine.pool).checkedout()


@pytest.mark.anyio
async def test_single_select_snapshot_excludes_later_commits(
    engine: AsyncEngine,
) -> None:
    camera_id = uuid4()
    expected_ids = await _seed_events(engine, camera_id, 1_105)
    statements: list[str] = []

    def record_sql(  # noqa: PLR0913
        connection: object,
        cursor: object,
        statement: str,
        parameters: object,
        context: object,
        executemany: bool,
    ) -> None:
        del connection, cursor, parameters, context, executemany
        if statement.lstrip().upper().startswith("SELECT") and "camera_events" in statement:
            statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record_sql)
    prepared = await EventExportService(_database(engine)).prepare(
        EventExportFilters(camera_id=camera_id)
    )
    chunks = [await anext(prepared.chunks), await anext(prepared.chunks)]

    async with engine.begin() as connection:
        inserted = await connection.execute(
            insert(CameraEvent)
            .values(
                event_type="camera.updated",
                camera_id=camera_id,
                camera_name="Committed after snapshot",
            )
            .returning(CameraEvent.id)
        )
        later_id = inserted.scalar_one()

    chunks.extend([chunk async for chunk in prepared.chunks])
    await prepared.aclose()
    event.remove(engine.sync_engine, "before_cursor_execute", record_sql)

    rows = list(csv.reader(io.StringIO(b"".join(chunks).decode("utf-8"))))
    observed_ids = [int(row[0]) for row in rows[1:]]
    assert rows[0] == ["id", "occurred_at", "event_type", "camera_id", "camera_name"]
    assert observed_ids == list(expected_ids)
    assert len(observed_ids) == len(set(observed_ids))
    assert later_id not in observed_ids
    assert len(statements) == 1


@pytest.mark.anyio
async def test_empty_export_emits_only_header_and_closes_resources(
    database_url: str,
) -> None:
    database = Database.connect(database_url)
    try:
        prepared = await EventExportService(database).prepare(EventExportFilters(camera_id=uuid4()))
        payload = b"".join([chunk async for chunk in prepared.chunks])

        assert payload == CSV_HEADER
        assert _checked_out(database) == 0
    finally:
        await database.close()


@pytest.mark.anyio
async def test_statement_failure_during_prepare_closes_the_connection(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.connect(database_url)

    def fail_statement(filters: EventExportFilters) -> object:
        del filters
        raise _StatementConstructionFailureError

    monkeypatch.setattr(EventRepository, "statement", staticmethod(fail_statement))
    app = FastAPI()
    app.include_router(build_event_router(database=database, require_session=_guard_factory))
    try:
        messages, failure = await _request_prepare_failure(app)

        assert isinstance(failure, _StatementConstructionFailureError)
        status = cast(
            "int",
            next(
                message["status"]
                for message in messages
                if message["type"] == "http.response.start"
            ),
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        assert status == 500
        assert not body.startswith(CSV_HEADER)
        assert _checked_out(database) == 0
    finally:
        await database.close()


def _guard_factory(*, user_action: bool) -> SessionDependency:
    assert user_action is False

    async def guard() -> AuthenticatedRequest:
        return AuthenticatedRequest(session_id="synthetic-session")

    return guard


async def _request_with_disconnect(
    app: FastAPI,
) -> list[Message]:
    messages: list[Message] = []
    request_sent = False
    disconnected = anyio.Event()

    async def receive() -> Message:
        nonlocal request_sent
        if not request_sent:
            request_sent = True
            return {"type": "http.request", "body": b"", "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        messages.append(message)
        if message["type"] == "http.response.body":
            disconnected.set()
            await anyio.sleep(0)  # noqa: ASYNC115

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
    await app(scope, receive, send)
    return messages


async def _request_fully(app: FastAPI, query: bytes = b"") -> list[Message]:
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
        "path": "/api/events/export.csv",
        "raw_path": b"/api/events/export.csv",
        "query_string": query,
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 80),
    }
    await app(scope, receive, send)
    return messages


async def _request_prepare_failure(
    app: FastAPI,
) -> tuple[list[Message], _StatementConstructionFailureError | None]:
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
        "path": "/api/events/export.csv",
        "raw_path": b"/api/events/export.csv",
        "query_string": b"",
        "headers": [(b"host", b"localhost")],
        "client": ("127.0.0.1", 50000),
        "server": ("localhost", 80),
    }
    try:
        await app(scope, receive, send)
    except _StatementConstructionFailureError as error:
        return messages, error
    return messages, None


@pytest.mark.anyio
async def test_real_export_disconnect_returns_connection_to_pool(
    database_url: str,
) -> None:
    camera_id = uuid4()
    writer = Database.connect(database_url)
    _ = await _seed_events(writer.engine, camera_id, 1_005)
    app = FastAPI()
    app.include_router(build_event_router(database=writer, require_session=_guard_factory))

    try:
        messages = await _request_with_disconnect(app)

        assert any(message["type"] == "http.response.start" for message in messages)
        assert _checked_out(writer) == 0
    finally:
        await writer.close()


@pytest.mark.anyio
async def test_authenticated_download_does_not_add_events(database_url: str) -> None:
    camera_id = uuid4()
    database = Database.connect(database_url)
    _ = await _seed_events(database.engine, camera_id, 1)
    app = FastAPI()
    app.include_router(build_event_router(database=database, require_session=_guard_factory))

    try:
        async with database.engine.connect() as connection:
            before = await connection.scalar(
                select(func.count())
                .select_from(CameraEvent)
                .where(CameraEvent.camera_id == camera_id)
            )
        messages = await _request_fully(app, f"camera_id={camera_id}".encode())
        status = cast(
            "int",
            next(
                message["status"]
                for message in messages
                if message["type"] == "http.response.start"
            ),
        )
        body = b"".join(
            message.get("body", b"")
            for message in messages
            if message["type"] == "http.response.body"
        )
        async with database.engine.connect() as connection:
            after = await connection.scalar(
                select(func.count())
                .select_from(CameraEvent)
                .where(CameraEvent.camera_id == camera_id)
            )

        assert status == 200
        assert len(list(csv.reader(io.StringIO(body.decode("utf-8"))))) == 2
        assert before == after == 1
        assert _checked_out(database) == 0
    finally:
        await database.close()
