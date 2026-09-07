"""Importable uvicorn application for the task 14b live HTTP QA run."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response

from gods_watching.api.app_settings import ApiSettings
from gods_watching.api.appearance_routes import build_appearance_router
from gods_watching.api.camera_routes import (
    AuthenticatedRequest as CameraAuthenticatedRequest,
)
from gods_watching.api.camera_routes import (
    SessionDependency,
)
from gods_watching.api.search_routes import build_search_router
from gods_watching.api.session_routes import build_session_router
from gods_watching.api.sessionguard import require_session
from gods_watching.auth import AuthService
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.storage import CropObjectStore, Database

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        message = f"{name} is required for the HTTP QA app"
        raise RuntimeError(message)
    return value


_database = Database.connect(_required("GW_DATABASE_URL"))
_auth = AuthService(_database)
_transport = TritonClipTransport(_required("GW_TRITON_GRPC_URL"))
_clip = ClipAdapter(_transport)
_search_repository = SearchRepository()
_search = SearchService(_search_repository, _clip)
_lookup = AppearanceLookupService(
    _search_repository,
    CropObjectStore(root=Path(_required("GW_CROPS_ROOT"))),
)
_config = ApiSettings(
    public_origin=os.environ.get("GW_PUBLIC_ORIGIN", "http://127.0.0.1:18181"),
    secure_cookie=os.environ.get("GW_SECURE_COOKIE", "0") == "1",
)


def _session_factory(*, user_action: bool) -> SessionDependency:
    guard = require_session(
        _auth,
        _config,
        user_action=user_action,
        mutation=user_action,
    )

    async def _dependency(
        request: Request,
        response: Response,
    ) -> CameraAuthenticatedRequest:
        context = await guard(request, response)
        return CameraAuthenticatedRequest(session_id=str(context.session.session_id))

    return _dependency


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    async with _transport:
        yield
    await _database.close()


app = FastAPI(
    title="Gods Watching Task 14b HTTP QA",
    lifespan=_lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)
app.include_router(build_session_router(_auth, _config))
app.include_router(
    build_search_router(
        database=_database,
        search=_search,
        require_session=_session_factory,
    )
)
app.include_router(
    build_appearance_router(
        database=_database,
        lookup=_lookup,
        require_session=_session_factory,
    )
)
