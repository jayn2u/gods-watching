"""Typed dependency bundle shared by all installed HTTP surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from gods_watching.media import MediaSessionId

if TYPE_CHECKING:
    from gods_watching.auth import AuthService, SessionRevocation, SessionRevocationHook
    from gods_watching.cameras.service import CameraService
    from gods_watching.media import WhepProxyService
    from gods_watching.settings.service import SettingsService
    from gods_watching.storage import Database

    from .app_settings import ApiSettings
    from .camera_runtime import CameraRuntimePort


class _RevocationChain:
    """Await the supplied hook and close resources owned by its session."""

    def __init__(self, first: SessionRevocationHook, whep: WhepProxyService) -> None:
        self._first: SessionRevocationHook = first
        self._whep: WhepProxyService = whep

    async def __call__(self, event: SessionRevocation) -> None:
        await self._first(event)
        _ = await self._whep.close_session(MediaSessionId(str(event.session_id)))


@dataclass(frozen=True, slots=True)
class ApiDependencies:
    """Own prepared production services without constructing implicit fallbacks."""

    database: Database
    auth: AuthService
    cameras: CameraService
    settings: SettingsService
    whep: WhepProxyService
    camera_runtime: CameraRuntimePort
    config: ApiSettings

    def with_whep_revocation(self) -> ApiDependencies:
        """Return dependencies whose auth events close this app's WHEP resources."""
        hook = _RevocationChain(self.auth.revocation_hook, self.whep)
        return replace(self, auth=replace(self.auth, revocation_hook=hook))
