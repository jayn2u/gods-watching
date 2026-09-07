"""FastAPI WHEP routes awaiting task-12 operator-session integration."""

from http import HTTPStatus
from typing import Annotated, Protocol, final
from uuid import UUID

from fastapi import APIRouter, Header, Request, Response

from gods_watching.contracts.identifiers import CameraId

from .errors import InvalidGatewayLocationError
from .models import AuthorizedMediaSession, ProxyResponse, WhepResourceId
from .proxy import WhepProxyService

_MAX_WHEP_BODY_BYTES = 1_048_576


class MediaSessionAuthorizer(Protocol):
    """Resolve an already-authenticated operator request into a media session."""

    async def authorize(self, request: Request) -> AuthorizedMediaSession | None:
        """Return no context unless the application session is authorized."""
        ...


@final
class DenyAllMediaSessionAuthorizer:
    """Keep the proxy private until task 12 injects real session authorization."""

    async def authorize(self, request: Request) -> AuthorizedMediaSession | None:
        """Deny every request regardless of untrusted browser input."""
        del request
        return None


@final
class _WhepHandlers:
    def __init__(
        self,
        service: WhepProxyService,
        authorizer: MediaSessionAuthorizer,
    ) -> None:
        self._service = service
        self._authorizer = authorizer

    async def create(
        self,
        camera_id: UUID,
        request: Request,
        content_type: Annotated[str | None, Header()] = None,
    ) -> Response:
        session = await self._authorizer.authorize(request)
        if session is None:
            return Response(status_code=HTTPStatus.UNAUTHORIZED)
        if content_type != "application/sdp":
            return Response(status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        body = await request.body()
        if len(body) > _MAX_WHEP_BODY_BYTES:
            return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        try:
            result = await self._service.create(CameraId(camera_id), session, body)
        except InvalidGatewayLocationError:
            return Response(status_code=HTTPStatus.BAD_GATEWAY)
        return _response(result)

    async def patch(self, camera_id: UUID, resource_id: UUID, request: Request) -> Response:
        session = await self._authorizer.authorize(request)
        if session is None:
            return Response(status_code=HTTPStatus.UNAUTHORIZED)
        body = await request.body()
        if len(body) > _MAX_WHEP_BODY_BYTES:
            return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        result = await self._service.patch(
            CameraId(camera_id), session, WhepResourceId(resource_id), body
        )
        return _response(result)

    async def delete(self, camera_id: UUID, resource_id: UUID, request: Request) -> Response:
        session = await self._authorizer.authorize(request)
        if session is None:
            return Response(status_code=HTTPStatus.UNAUTHORIZED)
        result = await self._service.delete(
            CameraId(camera_id), session, WhepResourceId(resource_id)
        )
        return _response(result)


def build_whep_router(
    service: WhepProxyService,
    authorizer: MediaSessionAuthorizer | None = None,
) -> APIRouter:
    """Build deny-by-default browser WHEP routes around the private proxy service."""
    handlers = _WhepHandlers(service, authorizer or DenyAllMediaSessionAuthorizer())
    router = APIRouter(prefix="/api/live", tags=["live"])
    router.add_api_route("/{camera_id}/whep", handlers.create, methods=["POST"])
    router.add_api_route(
        "/{camera_id}/whep/{resource_id}",
        handlers.patch,
        methods=["PATCH"],
    )
    router.add_api_route(
        "/{camera_id}/whep/{resource_id}",
        handlers.delete,
        methods=["DELETE"],
    )
    return router


def _response(result: ProxyResponse) -> Response:
    headers: dict[str, str] = {}
    if result.location is not None:
        headers["Location"] = result.location
    if result.etag is not None:
        headers["ETag"] = result.etag
    return Response(
        content=result.body,
        status_code=result.status_code,
        media_type=result.content_type,
        headers=headers,
    )
