"""Serve the built web app and the authenticated search API from one origin for Task 16."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from starlette.staticfiles import StaticFiles

from .task14_app import build as build_search_api

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.types import ASGIApp, Receive, Scope, Send

_WEB_DIST = Path(__file__).resolve().parents[5] / "web/dist"


async def build() -> FastAPI:
    """Mount the production web build behind the real API routes, as one origin."""
    if not (_WEB_DIST / "index.html").is_file():
        message = f"web build is missing: {_WEB_DIST}"
        raise RuntimeError(message)
    application = await build_search_api()
    # API routers are registered first, so /api paths never reach the static mount.
    application.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
    return application


class _LazyApp:
    def __init__(self) -> None:
        self._application: ASGIApp | None = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if self._application is None:
            self._application = await build()
        await self._application(scope, receive, send)


app = _LazyApp()
