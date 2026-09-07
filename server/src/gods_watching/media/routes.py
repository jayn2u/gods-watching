"""FastAPI WHEP routes awaiting task-12 operator-session integration."""

from collections.abc import Awaitable
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


class CameraAccessChecker(Protocol):
    """Check that a requested camera is currently live and not deleted."""

    def __call__(self, camera_id: CameraId, /) -> Awaitable[bool]:
        """Return whether WHEP may resolve this camera path."""
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
        camera_access: CameraAccessChecker | None,
    ) -> None:
        self._service = service
        self._authorizer = authorizer
        self._camera_access = camera_access

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
        body = await _read_bounded(request)
        if body is None:
            return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if self._camera_access is not None and not await self._camera_access(CameraId(camera_id)):
            return Response(status_code=HTTPStatus.NOT_FOUND)
        try:
            result = await self._service.create(CameraId(camera_id), session, body)
        except InvalidGatewayLocationError:
            return Response(status_code=HTTPStatus.BAD_GATEWAY)
        return _response(result)

    async def patch(self, camera_id: UUID, resource_id: UUID, request: Request) -> Response:
        session = await self._authorizer.authorize(request)
        if session is None:
            return Response(status_code=HTTPStatus.UNAUTHORIZED)
        if request.headers.get("content-type") != "application/trickle-ice-sdpfrag":
            return Response(status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE)
        body = await _read_bounded(request)
        if body is None:
            return Response(status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
        if self._camera_access is not None and not await self._camera_access(CameraId(camera_id)):
            return Response(status_code=HTTPStatus.NOT_FOUND)
        result = await self._service.patch(
            CameraId(camera_id), session, WhepResourceId(resource_id), body
        )
        return _response(result)

    async def delete(self, camera_id: UUID, resource_id: UUID, request: Request) -> Response:
        session = await self._authorizer.authorize(request)
        if session is None:
            return Response(status_code=HTTPStatus.UNAUTHORIZED)
        if self._camera_access is not None and not await self._camera_access(CameraId(camera_id)):
            return Response(status_code=HTTPStatus.NOT_FOUND)
        result = await self._service.delete(
            CameraId(camera_id), session, WhepResourceId(resource_id)
        )
        return _response(result)


def build_whep_router(
    service: WhepProxyService,
    authorizer: MediaSessionAuthorizer | None = None,
    camera_access: CameraAccessChecker | None = None,
) -> APIRouter:
    """Build deny-by-default browser WHEP routes around the private proxy service."""
    handlers = _WhepHandlers(
        service,
        authorizer or DenyAllMediaSessionAuthorizer(),
        camera_access,
    )
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


async def _read_bounded(request: Request) -> bytes | None:
    """Read a WHEP body while enforcing the cap for both framed and streamed input."""
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > _MAX_WHEP_BODY_BYTES:
                return None
        except ValueError:
            return None
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > _MAX_WHEP_BODY_BYTES:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


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
