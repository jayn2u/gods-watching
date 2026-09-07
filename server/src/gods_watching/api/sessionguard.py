"""Cookie session authentication and same-origin mutation dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, override

from fastapi import HTTPException, Request, Response

from gods_watching.auth import (
    Authenticated,
    AuthenticationFailure,
    AuthenticationFailureReason,
    AuthenticationResult,
    LoginAttempt,
    LoginResult,
    SessionRevocation,
    SessionToken,
    SessionView,
    require_same_origin,
)
from gods_watching.auth.policy import MutationOriginError
from gods_watching.media import AuthorizedMediaSession, MediaSessionId

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .app_settings import ApiSettings

_MAX_COOKIE_LENGTH = 512
_CONTROL_CHARACTER_LIMIT = 0x20


class AuthCapability(Protocol):
    """Subset of AuthService required by the request guard."""

    async def authenticate(
        self,
        token: SessionToken,
        *,
        user_action: bool = False,
    ) -> AuthenticationResult:
        """Authenticate an opaque cookie token."""
        ...

    async def login(self, attempt: LoginAttempt) -> LoginResult:
        """Authenticate a password and create a durable session."""
        ...

    async def logout(self, token: SessionToken) -> tuple[SessionRevocation, ...]:
        """Revoke the session represented by one opaque token."""
        ...


@dataclass(frozen=True, slots=True, repr=False)
class AuthenticatedRequest:
    """Carry authenticated request state while redacting the opaque token."""

    token: SessionToken = field(repr=False)
    session: SessionView
    media_session: AuthorizedMediaSession

    @override
    def __repr__(self) -> str:
        """Keep cookie material out of request diagnostics."""
        return f"AuthenticatedRequest(session_id={self.session.session_id!r})"


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def parse_session_cookie(request: Request, cookie_name: str) -> SessionToken | None:
    """Parse a bounded opaque cookie without copying it into error messages."""
    raw = request.cookies.get(cookie_name)
    if raw is None or not 1 <= len(raw) <= _MAX_COOKIE_LENGTH or not raw.isascii():
        return None
    if any(character.isspace() or ord(character) < _CONTROL_CHARACTER_LIMIT for character in raw):
        return None
    return SessionToken.from_raw(raw)


def request_peer_ip(request: Request) -> str:
    """Return the direct ASGI peer address required by canonical-IP policy."""
    client = request.client
    return client.host if client is not None else ""


def clear_session_cookie(response: Response, config: ApiSettings) -> None:
    """Expire the configured session cookie with matching security attributes."""
    response.delete_cookie(
        key=config.session_cookie_name,
        path=config.session_cookie_path,
        secure=config.secure_cookie,
        httponly=True,
        samesite="strict",
    )


@dataclass(frozen=True, slots=True)
class SessionGuard:
    """Resolve one real database-backed cookie session for a request."""

    auth: AuthCapability
    config: ApiSettings

    async def resolve(
        self,
        request: Request,
        response: Response,
        *,
        user_action: bool,
        mutation: bool,
    ) -> AuthenticatedRequest:
        """Authenticate a cookie and optionally enforce same-origin policy."""
        if mutation:
            try:
                require_same_origin(request.headers.get("origin"), self.config.public_origin)
            except MutationOriginError as error:
                raise _error(403, "origin_not_allowed", "mutation origin is not allowed") from error
        token = parse_session_cookie(request, self.config.session_cookie_name)
        if token is None:
            raise _error(401, "authentication_required", "authentication required")
        result = await self.auth.authenticate(token, user_action=user_action)
        match result:
            case Authenticated(session=session):
                return AuthenticatedRequest(
                    token=token,
                    session=session,
                    media_session=AuthorizedMediaSession(
                        session_id=MediaSessionId(str(session.session_id))
                    ),
                )
            case AuthenticationFailure(reason=reason):
                clear_session_cookie(response, self.config)
                match reason:
                    case AuthenticationFailureReason.EXPIRED:
                        code = "session_expired"
                    case AuthenticationFailureReason.UNKNOWN | AuthenticationFailureReason.REVOKED:
                        code = "authentication_required"
                raise _error(401, code, "authentication required")

    def dependency(
        self,
        *,
        user_action: bool,
        mutation: bool = False,
    ) -> Callable[..., Awaitable[AuthenticatedRequest]]:
        """Build a FastAPI dependency with explicit activity/origin semantics."""

        async def _dependency(
            request: Request,
            response: Response,
        ) -> AuthenticatedRequest:
            return await self.resolve(
                request,
                response,
                user_action=user_action,
                mutation=mutation,
            )

        return _dependency


def require_session(
    auth: AuthCapability,
    config: ApiSettings,
    *,
    user_action: bool,
    mutation: bool = False,
) -> Callable[..., Awaitable[AuthenticatedRequest]]:
    """Build the shared FastAPI dependency used by installed route modules."""
    return SessionGuard(auth=auth, config=config).dependency(
        user_action=user_action,
        mutation=mutation,
    )
