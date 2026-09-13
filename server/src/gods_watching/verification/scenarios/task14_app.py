"""Compose the authenticated search API used by Task 14 verification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from gods_watching.api import ApiDependencies, ApiSettings, create_app
from gods_watching.api.camera_runtime import CameraRuntime, CommittedStateDetector
from gods_watching.auth import AuthService
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.media import (
    HttpWhepGateway,
    MediaGatewayConnection,
    MediaPath,
    WhepProxyService,
)
from gods_watching.pipeline_worker.settings import locked_clip_model
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.settings import SettingsService
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
# Search verification never opens live media, so the WHEP gateway targets a closed port.
_UNUSED_MEDIA_PORT = 9


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        message = f"Task 14 app configuration is missing: {name}"
        raise RuntimeError(message)
    return value


async def build() -> FastAPI:
    """Build a real DB-backed app with CLIP text search and appearance lookups."""
    database = Database.connect(_required("GW_DATABASE_URL"))
    clip_transport = TritonClipTransport(_required("GW_TRITON_GRPC_URL"))
    _model_id, model_revision = locked_clip_model(_REPOSITORY_ROOT / "assets/models.lock.json")
    search_repository = SearchRepository()
    storage = StorageRepository(CredentialCipher(_required("GW_CAMERA_CIPHER_KEY").encode()))
    auth = AuthService(database)
    _ = await auth.initialize_password(_required("GW_OPERATOR_PASSWORD"))
    whep = WhepProxyService(
        gateway=HttpWhepGateway(
            MediaGatewayConnection(
                host="127.0.0.1",
                port=_UNUSED_MEDIA_PORT,
                username="unused",
                password="unused",  # noqa: S106 - never sent; no media path is opened
            )
        ),
        resolve_path=lambda camera_id: MediaPath(f"camera/{camera_id}"),
    )
    dependencies = ApiDependencies(
        database=database,
        auth=auth,
        cameras=CameraService(CameraRepository(storage)),
        settings=SettingsService(),
        whep=whep,
        camera_runtime=CameraRuntime(detector=CommittedStateDetector(), media=None),
        config=ApiSettings(public_origin=_required("GW_TASK14_PUBLIC_ORIGIN"), secure_cookie=False),
        search=SearchService(
            search_repository, ClipAdapter(clip_transport), model_revision=model_revision
        ),
        appearance=AppearanceLookupService(
            search_repository, CropObjectStore(root=Path(_required("GW_CROPS_ROOT")))
        ),
        clip_lifecycle=clip_transport,
    )
    return create_app(dependencies)


class _LazyApp:
    def __init__(self) -> None:
        self._application: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._application is None:
            self._application = await build()
        await self._application(scope, receive, send)


app = _LazyApp()
