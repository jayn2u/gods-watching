"""Isolated private-server entrypoint used only by task-9 verification."""

import hmac
import os
from pathlib import Path
from typing import ClassVar, final
from uuid import UUID

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, SecretStr

from gods_watching.media import (
    AuthorizedMediaSession,
    HttpWhepGateway,
    MediaGatewayConnection,
    MediaPath,
    MediaSessionId,
    WhepProxyService,
    build_whep_router,
)


class MediaQaSettings(BaseModel):
    """Parse ephemeral credentials and loopback endpoints from a cleanup-owned file."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    gateway_host: str
    gateway_port: int
    gateway_username: str
    gateway_password: SecretStr
    camera_id: UUID
    media_path: str
    qa_session: SecretStr
    origin: str


@final
class _QaSessionAuthorizer:
    def __init__(self, expected: SecretStr) -> None:
        self._expected = expected

    async def authorize(self, request: Request) -> AuthorizedMediaSession | None:
        supplied = request.headers.get("X-GW-QA-Session", "")
        expected = self._expected.get_secret_value()
        if not hmac.compare_digest(supplied, expected):
            return None
        return AuthorizedMediaSession(session_id=MediaSessionId("task-9-qa-session"))


def create_qa_app() -> FastAPI:
    """Create the real proxy with an explicit verification-only authorization seam."""
    settings_path = Path(os.environ["GW_MEDIA_QA_SETTINGS_PATH"])
    settings = MediaQaSettings.model_validate_json(settings_path.read_text(encoding="utf-8"))
    connection = MediaGatewayConnection(
        host=settings.gateway_host,
        port=settings.gateway_port,
        username=settings.gateway_username,
        password=settings.gateway_password.get_secret_value(),
    )
    service = WhepProxyService(
        gateway=HttpWhepGateway(connection),
        resolve_path=lambda _camera_id: MediaPath(settings.media_path),
    )
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_whep_router(service, _QaSessionAuthorizer(settings.qa_session)))
    app.add_api_route("/", _live_page, methods=["GET"], response_class=HTMLResponse)
    return app


async def _live_page() -> HTMLResponse:
    video = '<video autoplay muted playsinline style="width:100vw;height:100vh;object-fit:contain">'
    return HTMLResponse(
        "".join(
            (
                "<!doctype html><html><head><title>Task 9 live decode</title></head>",
                f'<body style="margin:0;background:#03070c">{video}</video></body></html>',
            )
        )
    )
