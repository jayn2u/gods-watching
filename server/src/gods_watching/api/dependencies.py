"""Typed dependency bundle shared by all installed HTTP surfaces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol

from gods_watching.media import MediaSessionId

if TYPE_CHECKING:
    from types import TracebackType

    from gods_watching.api.appearance_routes import AppearanceLookupProvider
    from gods_watching.api.search_routes import SearchServiceProvider
    from gods_watching.auth import AuthService, SessionRevocation, SessionRevocationHook
    from gods_watching.cameras.service import CameraService
    from gods_watching.inference.clip import ClipTransport
    from gods_watching.media import WhepProxyService
    from gods_watching.settings.service import SettingsService
    from gods_watching.storage import Database

    from .app_settings import ApiSettings
    from .camera_runtime import CameraRuntimePort
    from .model_routes import ModelSelectionProvider
    from .training_routes import TrainingServiceProvider


class ClipTransportLifecycle(Protocol):
    """Own one prepared CLIP transport for the application lifetime."""

    async def __aenter__(self) -> ClipTransport:
        """Open the transport before request handling begins."""
        ...

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the transport after all application resources are released."""
        ...


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
    search: SearchServiceProvider
    appearance: AppearanceLookupProvider
    clip_lifecycle: ClipTransportLifecycle
    model_selection: ModelSelectionProvider | None = None
    training: TrainingServiceProvider | None = None

    def with_whep_revocation(self) -> ApiDependencies:
        """Return dependencies whose auth events close this app's WHEP resources."""
        hook = _RevocationChain(self.auth.revocation_hook, self.whep)
        return replace(self, auth=replace(self.auth, revocation_hook=hook))
