from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Final

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.staticfiles import StaticFiles

from gods_watching.auth import AuthService
from gods_watching.cameras import CameraRepository, CameraService, RtspSourceProbe
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
from gods_watching.pipeline_worker.settings import locked_clip_model
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.settings import SettingsService
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

from .app_settings import ApiSettings
from .application import create_app
from .camera_runtime import CameraRuntime, CommittedStateDetector
from .dependencies import ApiDependencies

_MISSING_RUNTIME_PATH_CODE: Final = "missing_runtime_path"

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

    from gods_watching.contracts.identifiers import CameraId


class _ProductionSettings(BaseSettings):
    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="GW_",
        extra="ignore",
        frozen=True,
        validate_default=True,
    )

    database_url: str = Field(default="", repr=False, min_length=1)
    triton_grpc_url: str = Field(default="", min_length=1)
    crops_root: Path = Path()
    camera_cipher_key: str = Field(default="", repr=False, min_length=1)
    operator_username: str = Field(default="admin", min_length=1, max_length=64)
    operator_password: str = Field(default="", repr=False, min_length=4, max_length=128)
    public_origin: str = Field(default="", min_length=1)
    secure_cookie: bool = False
    trusted_gateway: str | None = None
    media_control_host: str = Field(default="", min_length=1)
    media_control_port: int = Field(default=9997, ge=1, le=65535)
    media_control_user: str = Field(default="", min_length=1)
    media_control_password: str = Field(default="", repr=False, min_length=1)
    media_whep_host: str = Field(default="", min_length=1)
    media_whep_port: int = Field(default=8889, ge=1, le=65535)
    media_reader_user: str = Field(default="", min_length=1)
    media_reader_password: str = Field(default="", repr=False, min_length=1)
    web_root: Path = Path()
    model_lock_path: Path = Path("/opt/gods-watching/assets/models.lock.json")

    @model_validator(mode="after")
    def _require_paths(self) -> _ProductionSettings:
        if self.crops_root == Path() or self.web_root == Path():
            raise PydanticCustomError(
                _MISSING_RUNTIME_PATH_CODE,
                "GW_CROPS_ROOT and GW_WEB_ROOT are required",
            )
        return self


class _GenerationSink:
    async def publish(self, event: GenerationEvent) -> None:
        del event


async def _build_production_app(settings: _ProductionSettings | None = None) -> FastAPI:
    configured = settings or _ProductionSettings()
    if not (configured.web_root / "index.html").is_file():
        message = f"web build is missing: {configured.web_root}"
        raise RuntimeError(message)

    database = Database.connect(configured.database_url)
    clip_transport = TritonClipTransport(configured.triton_grpc_url)
    _model_id, model_revision = locked_clip_model(configured.model_lock_path)
    repository = SearchRepository()
    storage = StorageRepository(CredentialCipher(configured.camera_cipher_key.encode()))
    auth = AuthService(database, operator_username=configured.operator_username)
    _ = await auth.sync_password(configured.operator_password)

    control = HttpMediaControlGateway(
        MediaGatewayConnection(
            host=configured.media_control_host,
            port=configured.media_control_port,
            username=configured.media_control_user,
            password=configured.media_control_password,
        )
    )
    media = MediaControlAdapter(
        gateway=control,
        generations=SourceGenerationCoordinator(sink=_GenerationSink()),
    )
    whep = WhepProxyService(
        gateway=HttpWhepGateway(
            MediaGatewayConnection(
                host=configured.media_whep_host,
                port=configured.media_whep_port,
                username=configured.media_reader_user,
                password=configured.media_reader_password,
            )
        ),
        resolve_path=lambda camera_id: MediaPath(f"camera/{camera_id}"),
    )

    async def close_camera(camera_id: CameraId) -> None:
        _ = await whep.close_camera(camera_id)

    dependencies = ApiDependencies(
        database=database,
        auth=auth,
        cameras=CameraService(CameraRepository(storage), source_probe=RtspSourceProbe()),
        settings=SettingsService(),
        whep=whep,
        camera_runtime=CameraRuntime(
            detector=CommittedStateDetector(),
            media=media,
            close_resources=close_camera,
        ),
        config=ApiSettings(
            public_origin=configured.public_origin,
            trusted_gateway=configured.trusted_gateway,
            secure_cookie=configured.secure_cookie,
        ),
        search=SearchService(
            repository,
            ClipAdapter(clip_transport),
            model_revision=model_revision,
        ),
        appearance=AppearanceLookupService(
            repository,
            CropObjectStore(root=configured.crops_root),
        ),
        clip_lifecycle=clip_transport,
    )
    application = create_app(dependencies)
    application.mount("/", StaticFiles(directory=configured.web_root, html=True), name="web")
    return application


class _LazyApplication:
    def __init__(self) -> None:
        self._application: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._application is None:
            self._application = await _build_production_app()
        await self._application(scope, receive, send)


ProductionSettings = _ProductionSettings
build_production_app = _build_production_app
app = _LazyApplication()

__all__ = ["ProductionSettings", "app", "build_production_app"]
