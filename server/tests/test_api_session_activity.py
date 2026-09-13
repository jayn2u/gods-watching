from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from fastapi import FastAPI
from pydantic import TypeAdapter

from gods_watching.api.app_settings import ApiSettings
from gods_watching.api.session_routes import build_session_router
from gods_watching.auth import (
    Authenticated,
    AuthenticationResult,
    LoginAttempt,
    LoginResult,
    SessionRevocation,
    SessionToken,
    SessionView,
)
from gods_watching.contracts.identifiers import LoginSessionId

if TYPE_CHECKING:
    from starlette.types import Message, Scope

_STATUS = TypeAdapter(int)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _Auth:
    def __init__(self) -> None:
        now = datetime.now(UTC)
        self.session: SessionView = SessionView(
            session_id=LoginSessionId(UUID(int=1)),
            created_at=now,
            last_activity_at=now,
            idle_expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
        )
        self.actions: list[bool] = []

    async def authenticate(
        self,
        token: SessionToken,
        *,
        user_action: bool = False,
    ) -> AuthenticationResult:
        del token
        self.actions.append(user_action)
        return Authenticated(self.session)

    async def login(self, attempt: LoginAttempt) -> LoginResult:
        del attempt
        raise AssertionError

    async def logout(self, token: SessionToken) -> tuple[SessionRevocation, ...]:
        del token
        raise AssertionError


async def _request_status(app: FastAPI, origin: str) -> int:
    delivered = False
    messages: list[Message] = []

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
        "method": "POST",
        "scheme": "https",
        "path": "/api/session/activity",
        "raw_path": b"/api/session/activity",
        "query_string": b"",
        "headers": [
            (b"cookie", b"gw_session=installed-session"),
            (b"origin", origin.encode()),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("gw.test", 443),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    return _STATUS.validate_python(start.get("status"))


@pytest.mark.anyio
async def test_activity_refresh_accepts_same_origin_and_rejects_cross_origin() -> None:
    auth = _Auth()
    app = FastAPI()
    app.include_router(build_session_router(auth, ApiSettings(public_origin="https://gw.test")))

    accepted = await _request_status(app, "https://gw.test")
    denied = await _request_status(app, "https://evil.test")

    assert accepted == 200
    assert denied == 403
    assert auth.actions == [False, True]
