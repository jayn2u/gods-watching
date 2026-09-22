from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request, Response

from gods_watching.api.camera_routes import (
    AuthenticatedRequest,
    SessionDependency,
    SessionDependencyFactory,
)
from gods_watching.api.live_routes import build_live_detection_router
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.live import DetectionBox, LiveDetectionResponse

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.types import Message, Scope


@dataclass
class _Transaction:
    session: AsyncSession

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        yield self.session


@dataclass
class _Reader:
    value: LiveDetectionResponse

    async def read(
        self,
        session: AsyncSession,
        camera_id: UUID,
        *,
        now: datetime | None = None,
    ) -> LiveDetectionResponse:
        del session, camera_id, now
        return self.value


def _guard_factory() -> SessionDependencyFactory:
    def factory(*, user_action: bool) -> SessionDependency:
        assert user_action is False

        async def guard(request: Request, response: Response) -> AuthenticatedRequest:
            del response
            if request.headers.get("x-test-session") != "valid":
                raise HTTPException(status_code=401, detail="authentication required")
            return AuthenticatedRequest(session_id="test-session")

        return guard

    return factory


async def _request(
    app: FastAPI,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    messages: list[Message] = []
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": request_headers,
        "client": ("127.0.0.1", 50000),
        "server": ("gw.test", 443),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    status = cast("int", start["status"])
    response_headers = {
        key.decode(): value.decode()
        for key, value in cast("list[tuple[bytes, bytes]]", start.get("headers", []))
    }
    body = b"".join(
        body
        for message in messages
        if message.get("type") == "http.response.body"
        and isinstance(body := message.get("body"), bytes)
    )
    return status, response_headers, body


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_live_detection_route_requires_passive_auth_and_disables_caching() -> None:
    camera_id = CameraId(uuid4())
    session_id = CameraSessionId(uuid4())
    reader = _Reader(
        LiveDetectionResponse(
            camera_id=camera_id,
            camera_session_id=session_id,
            frame_at=datetime(2026, 1, 1, tzinfo=UTC),
            frame_age_seconds=0.2,
            width=640,
            height=480,
            boxes=(DetectionBox(x1=1, y1=2, x2=30, y2=40, confidence=0.8),),
        )
    )
    app = FastAPI()
    app.include_router(
        build_live_detection_router(
            database=_Transaction(session=cast("AsyncSession", object())),
            require_session=_guard_factory(),
            reader=reader,
        )
    )

    denied, _, _ = await _request(app, f"/api/live/{camera_id}/detections")
    accepted, headers, body = await _request(
        app,
        f"/api/live/{camera_id}/detections",
        headers={"x-test-session": "valid"},
    )

    assert denied == 401
    assert accepted == 200
    assert headers["cache-control"] == "no-store"
    assert json.loads(body)["camera_session_id"] == str(session_id)


@pytest.mark.anyio
async def test_live_detection_route_returns_empty_detection_snapshot() -> None:
    camera_id = CameraId(uuid4())
    reader = _Reader(
        LiveDetectionResponse(
            camera_id=camera_id,
            camera_session_id=None,
            frame_at=None,
            frame_age_seconds=None,
            width=None,
            height=None,
            boxes=(),
        )
    )
    app = FastAPI()
    app.include_router(
        build_live_detection_router(
            database=_Transaction(session=cast("AsyncSession", object())),
            require_session=_guard_factory(),
            reader=reader,
        )
    )

    status, _, body = await _request(
        app,
        f"/api/live/{camera_id}/detections",
        headers={"x-test-session": "valid"},
    )

    assert status == 200
    assert json.loads(body) == {
        "camera_id": str(camera_id),
        "camera_session_id": None,
        "frame_at": None,
        "frame_age_seconds": None,
        "width": None,
        "height": None,
        "boxes": [],
    }
