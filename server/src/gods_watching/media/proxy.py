"""Application-owned WHEP resource proxy contract."""

from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from threading import Lock
from typing import Protocol, final
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from gods_watching.contracts.identifiers import CameraId

from .errors import InvalidGatewayLocationError
from .models import (
    AuthorizedMediaSession,
    GatewayResponse,
    MediaPath,
    MediaSessionId,
    ProxyResponse,
    WhepResourceId,
)


@dataclass(frozen=True, slots=True)
class _OwnedResource:
    camera_id: CameraId
    owner_id: MediaSessionId
    upstream_location: str


class WhepGateway(Protocol):
    """Send WHEP lifecycle requests to the private media listener."""

    async def create(self, path: MediaPath, offer: bytes) -> GatewayResponse:
        """Create a receive-only WHEP resource."""
        ...

    async def patch(self, location: str, fragment: bytes) -> GatewayResponse:
        """Forward one trickle-ICE fragment."""
        ...

    async def delete(self, location: str) -> GatewayResponse:
        """Delete one upstream WHEP resource."""
        ...


@final
class WhepProxyService:
    """Own and rewrite browser-visible WHEP resources."""

    def __init__(
        self,
        *,
        gateway: WhepGateway,
        resolve_path: Callable[[CameraId], MediaPath],
    ) -> None:
        """Create an empty browser-resource ownership registry."""
        self._gateway = gateway
        self._resolve_path = resolve_path
        self._resources: dict[WhepResourceId, _OwnedResource] = {}
        self._lock = Lock()

    async def create(
        self,
        camera_id: CameraId,
        session: AuthorizedMediaSession,
        offer: bytes,
    ) -> ProxyResponse:
        """Create an application-owned resource from a browser offer."""
        path = self._resolve_path(camera_id)
        response = await self._gateway.create(path, offer)
        if response.status_code != HTTPStatus.CREATED or response.location is None:
            return _sanitize(response)
        upstream_location = _parse_upstream_location(response.location, path)
        resource_id = WhepResourceId(uuid4())
        with self._lock:
            self._resources[resource_id] = _OwnedResource(
                camera_id=camera_id,
                owner_id=session.session_id,
                upstream_location=upstream_location,
            )
        return ProxyResponse(
            status_code=response.status_code,
            body=response.body,
            content_type=response.content_type,
            location=f"/api/live/{camera_id}/whep/{resource_id}",
            etag=response.etag,
        )

    async def patch(
        self,
        camera_id: CameraId,
        session: AuthorizedMediaSession,
        resource_id: WhepResourceId,
        fragment: bytes,
    ) -> ProxyResponse:
        """Forward trickle ICE only for the owning application session."""
        resource = self._owned_resource(camera_id, session.session_id, resource_id)
        if resource is None:
            return ProxyResponse(status_code=404, body=b"")
        response = await self._gateway.patch(resource.upstream_location, fragment)
        return _sanitize(response)

    async def delete(
        self,
        camera_id: CameraId,
        session: AuthorizedMediaSession,
        resource_id: WhepResourceId,
    ) -> ProxyResponse:
        """Close one resource owned by the requesting application session."""
        resource = self._take_owned_resource(camera_id, session.session_id, resource_id)
        if resource is None:
            return ProxyResponse(status_code=404, body=b"")
        response = await self._gateway.delete(resource.upstream_location)
        if not _is_success(response.status_code):
            with self._lock:
                self._resources[resource_id] = resource
        return _sanitize(response)

    async def close_session(self, session_id: MediaSessionId) -> int:
        """Close every abandoned resource owned by an ending session."""
        with self._lock:
            selected = tuple(
                (resource_id, resource)
                for resource_id, resource in self._resources.items()
                if resource.owner_id == session_id
            )
            for resource_id, _resource in selected:
                del self._resources[resource_id]
        closed = 0
        for resource_id, resource in selected:
            response = await self._gateway.delete(resource.upstream_location)
            if _is_success(response.status_code):
                closed += 1
            else:
                with self._lock:
                    self._resources[resource_id] = resource
        return closed

    def _owned_resource(
        self,
        camera_id: CameraId,
        session_id: MediaSessionId,
        resource_id: WhepResourceId,
    ) -> _OwnedResource | None:
        with self._lock:
            resource = self._resources.get(resource_id)
        if resource is None:
            return None
        if resource.camera_id != camera_id or resource.owner_id != session_id:
            return None
        return resource

    def _take_owned_resource(
        self,
        camera_id: CameraId,
        session_id: MediaSessionId,
        resource_id: WhepResourceId,
    ) -> _OwnedResource | None:
        with self._lock:
            resource = self._resources.get(resource_id)
            if resource is None:
                return None
            if resource.camera_id != camera_id or resource.owner_id != session_id:
                return None
            del self._resources[resource_id]
        return resource


def _parse_upstream_location(location: str, path: MediaPath) -> str:
    parsed = urlsplit(location)
    prefix = f"/{path}/whep/"
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise InvalidGatewayLocationError(location=location)
    if not parsed.path.startswith(prefix):
        raise InvalidGatewayLocationError(location=location)
    secret = parsed.path.removeprefix(prefix)
    try:
        parsed_secret = UUID(secret)
    except ValueError as error:
        raise InvalidGatewayLocationError(location=location) from error
    if str(parsed_secret) != secret:
        raise InvalidGatewayLocationError(location=location)
    return parsed.path


def _sanitize(response: GatewayResponse) -> ProxyResponse:
    successful = _is_success(response.status_code)
    return ProxyResponse(
        status_code=response.status_code,
        body=response.body if successful else b"",
        content_type=response.content_type if successful else None,
        etag=response.etag if successful else None,
    )


def _is_success(status_code: int) -> bool:
    return HTTPStatus.OK <= status_code < HTTPStatus.MULTIPLE_CHOICES
