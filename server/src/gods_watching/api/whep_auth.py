"""Authenticated WHEP authorizer bound to the operator session cookie."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, override

from gods_watching.auth import (
    Authenticated,
    AuthenticationFailure,
    AuthenticationResult,
    is_same_origin,
)
from gods_watching.media import AuthorizedMediaSession, MediaSessionAuthorizer, MediaSessionId

from .sessionguard import AuthCapability, parse_session_cookie

if TYPE_CHECKING:
    from fastapi import Request

    from .app_settings import ApiSettings


@dataclass(frozen=True, slots=True)
class SessionWhepAuthorizer(MediaSessionAuthorizer):
    """Authorize WHEP without refreshing passive session activity."""

    auth: AuthCapability
    config: ApiSettings

    @override
    async def authorize(self, request: Request) -> AuthorizedMediaSession | None:
        """Return the media ownership projection only for a same-origin session."""
        if not is_same_origin(request.headers.get("origin"), self.config.public_origin):
            return None
        token = parse_session_cookie(request, self.config.session_cookie_name)
        if token is None:
            return None
        result: AuthenticationResult = await self.auth.authenticate(token, user_action=False)
        match result:
            case Authenticated(session=session):
                return AuthorizedMediaSession(session_id=MediaSessionId(str(session.session_id)))
            case AuthenticationFailure():
                return None
