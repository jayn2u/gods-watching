"""HTTP session login, status, activity, and logout routes."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Annotated

import anyio
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status

from gods_watching.auth import (
    ABSOLUTE_TIMEOUT,
    Authenticated,
    AuthenticationFailure,
    AuthenticationFailureReason,
    AuthenticationResult,
    InvalidCredentials,
    LoginAccepted,
    LoginAttempt,
    LoginResult,
    LoginThrottled,
    require_same_origin,
)
from gods_watching.auth.policy import MutationOriginError
from gods_watching.contracts.session import LoginRequest, SessionResponse

from .app_settings import ApiSettings
from .sessionguard import (
    AuthCapability,
    AuthenticatedRequest,
    clear_session_cookie,
    parse_session_cookie,
    request_peer_ip,
    require_session,
)


@dataclass(slots=True)
class SessionActivityLimiter:
    """Bound refreshes to one write per session per configured minute."""

    refresh_interval: float = 60.0
    _last_refresh: dict[str, float] = field(default_factory=dict)
    _lock: anyio.Lock = field(default_factory=anyio.Lock)

    async def allow(self, session_id: str) -> bool:
        """Return whether this activity event may refresh the idle timeout."""
        now = anyio.current_time()
        async with self._lock:
            previous = self._last_refresh.get(session_id)
            if previous is not None and now - previous < self.refresh_interval:
                return False
            self._last_refresh[session_id] = now
            return True


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


def _session_response(result: Authenticated | None) -> SessionResponse:
    if result is None:
        return SessionResponse(
            authenticated=False,
            idle_expires_at=None,
            absolute_expires_at=None,
        )
    return SessionResponse(
        authenticated=True,
        idle_expires_at=result.session.idle_expires_at,
        absolute_expires_at=result.session.absolute_expires_at,
    )


def _failure_response(result: AuthenticationFailure) -> SessionResponse:
    match result.reason:
        case AuthenticationFailureReason.UNKNOWN | AuthenticationFailureReason.REVOKED:
            return _session_response(None)
        case AuthenticationFailureReason.EXPIRED:
            return _session_response(None)


def _set_cookie(response: Response, config: ApiSettings, accepted: LoginAccepted) -> None:
    response.set_cookie(
        key=config.session_cookie_name,
        value=accepted.token.raw,
        max_age=int(ABSOLUTE_TIMEOUT.total_seconds()),
        httponly=True,
        secure=config.secure_cookie,
        samesite="strict",
        path=config.session_cookie_path,
    )


def build_session_router(
    auth: AuthCapability,
    config: ApiSettings,
    *,
    activity_limiter: SessionActivityLimiter | None = None,
) -> APIRouter:
    """Build session routes around the supplied database-backed auth service."""
    router = APIRouter(prefix="/api/session", tags=["session"])
    limiter = activity_limiter or SessionActivityLimiter(config.activity_refresh_seconds)
    guard = require_session(auth, config, user_action=False, mutation=True)
    _register_session_status(router, auth, config)
    _register_login(router, auth, config)
    _register_activity(router, auth, config, limiter, guard)
    _register_logout(router, auth, config)
    return router


def _register_session_status(
    router: APIRouter,
    auth: AuthCapability,
    config: ApiSettings,
) -> None:
    async def get_session(request: Request, response: Response) -> SessionResponse:
        token = parse_session_cookie(request, config.session_cookie_name)
        if token is None:
            return _session_response(None)
        result = await auth.authenticate(token, user_action=False)
        match result:
            case Authenticated() as accepted:
                return _session_response(accepted)
            case AuthenticationFailure() as failure:
                clear_session_cookie(response, config)
                return _failure_response(failure)

    router.add_api_route("", get_session, methods=["GET"], response_model=SessionResponse)


def _register_login(
    router: APIRouter,
    auth: AuthCapability,
    config: ApiSettings,
) -> None:
    async def login(
        payload: LoginRequest,
        request: Request,
        response: Response,
    ) -> SessionResponse:
        await _require_origin(request, config)
        result: LoginResult = await auth.login(
            LoginAttempt(
                password=payload.password.get_secret_value(),
                peer_ip=request_peer_ip(request),
                forwarded_for=request.headers.get("x-forwarded-for"),
                trusted_gateway=config.trusted_gateway,
            )
        )
        match result:
            case LoginAccepted() as accepted:
                _set_cookie(response, config, accepted)
                return _session_response(Authenticated(accepted.session))
            case InvalidCredentials():
                raise _error(401, "invalid_credentials", "invalid credentials")
            case LoginThrottled(retry_after_seconds=retry_after):
                error = _error(429, "login_throttled", "too many login attempts")
                error.headers = {"Retry-After": str(retry_after)}
                raise error

    router.add_api_route("", login, methods=["POST"], response_model=SessionResponse)


def _register_activity(
    router: APIRouter,
    auth: AuthCapability,
    config: ApiSettings,
    limiter: SessionActivityLimiter,
    guard: Callable[..., Awaitable[AuthenticatedRequest]],
) -> None:
    async def activity(
        response: Response,
        context: Annotated[AuthenticatedRequest, Depends(guard)],
    ) -> SessionResponse:
        session_key = str(context.session.session_id)
        if not await limiter.allow(session_key):
            return _session_response(Authenticated(context.session))
        result: AuthenticationResult = await auth.authenticate(context.token, user_action=True)
        match result:
            case Authenticated() as accepted:
                return _session_response(accepted)
            case AuthenticationFailure():
                clear_session_cookie(response, config)
                raise _error(401, "authentication_required", "authentication required")

    router.add_api_route(
        "/activity",
        activity,
        methods=["POST"],
        response_model=SessionResponse,
    )


def _register_logout(
    router: APIRouter,
    auth: AuthCapability,
    config: ApiSettings,
) -> None:
    async def logout(request: Request, response: Response) -> Response:
        await _require_origin(request, config)
        token = parse_session_cookie(request, config.session_cookie_name)
        if token is not None:
            _ = await auth.logout(token)
        clear_session_cookie(response, config)
        response.status_code = status.HTTP_204_NO_CONTENT
        return response

    router.add_api_route("", logout, methods=["DELETE"], status_code=status.HTTP_204_NO_CONTENT)


async def _require_origin(request: Request, config: ApiSettings) -> None:
    """Apply the same-origin rule to session mutations."""
    try:
        require_same_origin(request.headers.get("origin"), config.public_origin)
    except MutationOriginError as error:
        raise _error(403, "origin_not_allowed", "mutation origin is not allowed") from error
