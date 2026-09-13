"""Compose the isolated FastAPI application used by Task 12 verification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from gods_watching.api import ApiDependencies, ApiSettings, create_app
from gods_watching.api.camera_runtime import CameraRuntime
from gods_watching.auth import AuthService
from gods_watching.cameras import CameraRepository, CameraService, RtspSourceProbe
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.media import (
    GenerationEvent,
    HttpMediaControlGateway,
    HttpWhepGateway,
    MediaControlAdapter,
    MediaGatewayConnection,
    MediaPath,
    SourceGenerationCoordinator,
    WhepProxyService,
)
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.settings import SettingsService
from gods_watching.storage import (
    Camera,
    CredentialCipher,
    CropObjectStore,
    Database,
    StorageRepository,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

    from gods_watching.contracts.identifiers import CameraId


def _required(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        message = f"Task 12 app configuration is missing: {name}"
        raise RuntimeError(message)
    return value


class _GenerationSink:
    async def publish(self, event: GenerationEvent) -> None:
        del event


async def _seed_camera(
    database: Database,
    storage: StorageRepository,
    source_url: str,
) -> Camera:
    async with database.transaction() as session:
        existing = await session.scalar(select(Camera).order_by(Camera.created_at, Camera.id))
        if existing is not None:
            return existing
        request = CameraCreateRequest.model_validate(
            {
                "name": "Task 12 fixture camera",
                "source_url": source_url,
                "detection_enabled": False,
                "detection_threshold": 0.5,
            }
        )
        return await storage.add_camera(
            session,
            name=request.name,
            source_url=request.source_url,
            detection_enabled=request.detection_enabled,
            detection_threshold=request.detection_threshold,
        )


async def build() -> FastAPI:
    """Build a real DB-backed app with private MediaMTX adapters."""
    database = Database.connect(_required("GW_DATABASE_URL"))
    clip_transport = TritonClipTransport(_required("GW_TRITON_GRPC_URL"))
    search_repository = SearchRepository()
    clip = ClipAdapter(clip_transport)
    search = SearchService(search_repository, clip)
    appearance = AppearanceLookupService(
        search_repository,
        CropObjectStore(root=Path(_required("GW_CROPS_ROOT"))),
    )
    cipher = CredentialCipher(_required("GW_CAMERA_CIPHER_KEY").encode())
    storage = StorageRepository(cipher)
    source_url = _required("GW_TASK12_SOURCE_URL")
    camera = await _seed_camera(database, storage, source_url)
    auth = AuthService(database)
    _ = await auth.initialize_password(_required("GW_OPERATOR_PASSWORD"))

    control = HttpMediaControlGateway(
        MediaGatewayConnection(
            host="127.0.0.1",
            port=int(_required("GW_MEDIA_CONTROL_PORT")),
            username=_required("GW_MEDIA_CONTROL_USER"),
            password=_required("GW_MEDIA_CONTROL_PASSWORD"),
        )
    )
    await control.upsert_path(MediaPath(f"camera/{camera.id}"), source_url)
    media = MediaControlAdapter(
        gateway=control,
        generations=SourceGenerationCoordinator(sink=_GenerationSink()),
    )
    whep = WhepProxyService(
        gateway=HttpWhepGateway(
            MediaGatewayConnection(
                host="127.0.0.1",
                port=int(_required("GW_MEDIA_WHEP_PORT")),
                username=_required("GW_MEDIA_READER_USER"),
                password=_required("GW_MEDIA_READER_PASSWORD"),
            )
        ),
        resolve_path=lambda camera_id: MediaPath(f"camera/{camera_id}"),
    )
    cameras = CameraService(CameraRepository(storage), source_probe=RtspSourceProbe())

    async def close_camera(camera_id: CameraId) -> None:
        _ = await whep.close_camera(camera_id)

    dependencies = ApiDependencies(
        database=database,
        auth=auth,
        cameras=cameras,
        settings=SettingsService(),
        whep=whep,
        camera_runtime=CameraRuntime(
            detector=None,
            media=media,
            close_resources=close_camera,
        ),
        config=ApiSettings(
            public_origin=_required("GW_TASK12_PUBLIC_ORIGIN"),
            secure_cookie=_required("GW_TASK12_SECURE_COOKIE").lower() == "true",
        ),
        search=search,
        appearance=appearance,
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
