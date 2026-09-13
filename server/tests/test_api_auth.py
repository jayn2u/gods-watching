from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

import anyio
from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import Response

from gods_watching.api.app_settings import ApiSettings
from gods_watching.api.session_routes import build_session_router
from gods_watching.api.sessionguard import AuthenticatedRequest
from gods_watching.auth import (
    AuthenticationFailure,
    AuthenticationFailureReason,
    SessionToken,
    SessionView,
)
from gods_watching.contracts.identifiers import LoginSessionId
from gods_watching.media import AuthorizedMediaSession, MediaSessionId

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.types import Scope

    from gods_watching.auth import (
        AuthenticationResult,
        LoginAttempt,
        LoginResult,
        SessionRevocation,
    )
    from gods_watching.contracts.session import SessionResponse


def test_authenticated_request_keeps_session_token_out_of_repr() -> None:
    now = datetime.now(UTC)
    context = AuthenticatedRequest(
        token=SessionToken.from_raw("secret-session-token"),
        session=SessionView(
            session_id=LoginSessionId(UUID("00000000-0000-0000-0000-000000000001")),
            created_at=now,
            last_activity_at=now,
            idle_expires_at=now + timedelta(minutes=30),
            absolute_expires_at=now + timedelta(hours=8),
        ),
        media_session=AuthorizedMediaSession(
            session_id=MediaSessionId("00000000-0000-0000-0000-000000000001")
        ),
    )

    assert "secret-session-token" not in repr(context)


def test_session_get_without_cookie_is_anonymous() -> None:
    class Auth:
        async def authenticate(
            self,
            token: SessionToken,
            *,
            user_action: bool = False,
        ) -> AuthenticationResult:
            del token, user_action
            return AuthenticationFailure(AuthenticationFailureReason.UNKNOWN)

        async def login(self, attempt: LoginAttempt) -> LoginResult:
            del attempt
            raise AssertionError

        async def logout(self, token: SessionToken) -> tuple[SessionRevocation, ...]:
            del token
            return ()

    router = build_session_router(Auth(), ApiSettings(public_origin="https://gw.test"))
    route = next(route for route in router.routes if isinstance(route, APIRoute))
    assert isinstance(route, APIRoute)
    endpoint: Callable[[Request, Response], Awaitable[SessionResponse]] = route.endpoint
    headers: list[tuple[bytes, bytes]] = []
    scope: Scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/session",
        "raw_path": b"/api/session",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("gw.test", 443),
        "scheme": "https",
    }

    result = anyio.run(endpoint, Request(scope), Response())

    assert result.authenticated is False


def test_session_delete_returns_no_content_response() -> None:
    class Auth:
        async def authenticate(
            self,
            token: SessionToken,
            *,
            user_action: bool = False,
        ) -> AuthenticationResult:
            del token, user_action
            return AuthenticationFailure(AuthenticationFailureReason.UNKNOWN)

        async def login(self, attempt: LoginAttempt) -> LoginResult:
            del attempt
            raise AssertionError

        async def logout(self, token: SessionToken) -> tuple[SessionRevocation, ...]:
            del token
            return ()

    router = build_session_router(Auth(), ApiSettings(public_origin="https://gw.test"))
    route = next(
        route
        for route in router.routes
        if isinstance(route, APIRoute) and "DELETE" in route.methods
    )
    assert isinstance(route, APIRoute)
    endpoint: Callable[[Request, Response], Awaitable[Response]] = route.endpoint
    scope: Scope = {
        "type": "http",
        "method": "DELETE",
        "path": "/api/session",
        "raw_path": b"/api/session",
        "query_string": b"",
        "headers": [
            (b"origin", b"https://gw.test"),
            (b"cookie", b"gw_session=secret-session-token"),
        ],
        "client": ("127.0.0.1", 50000),
        "server": ("gw.test", 443),
        "scheme": "https",
    }

    result = anyio.run(endpoint, Request(scope), Response())

    assert result.status_code == 204
