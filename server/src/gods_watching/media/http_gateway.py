"""Private HTTP adapters for MediaMTX control and WHEP listeners."""

import base64
import http.client
import json
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from typing import final, override
from urllib.parse import quote

from anyio import to_thread

from .models import GatewayResponse, MediaPath

_MAX_RESPONSE_BYTES = 1_048_576


@dataclass(frozen=True, slots=True)
class MediaGatewayConnection:
    """Configure one private MediaMTX listener without printable credentials."""

    host: str
    port: int
    username: str = field(repr=False)
    password: str = field(repr=False)
    timeout_seconds: float = 10.0


@final
class HttpWhepGateway:
    """Forward WHEP requests with a server-side MediaMTX credential."""

    def __init__(self, connection: MediaGatewayConnection) -> None:
        """Bind the adapter to one private listener."""
        self._connection = connection

    async def create(self, path: MediaPath, offer: bytes) -> GatewayResponse:
        """Create one receive-only WHEP resource."""
        return await to_thread.run_sync(
            partial(
                self._request,
                method="POST",
                target=f"/{quote(path, safe='/')}/whep",
                body=offer,
                content_type="application/sdp",
            )
        )

    async def patch(self, location: str, fragment: bytes) -> GatewayResponse:
        """Forward one application-owned ICE fragment."""
        return await to_thread.run_sync(
            partial(
                self._request,
                method="PATCH",
                target=location,
                body=fragment,
                content_type="application/trickle-ice-sdpfrag",
            )
        )

    async def delete(self, location: str) -> GatewayResponse:
        """Close one application-owned WHEP resource."""
        return await to_thread.run_sync(
            partial(self._request, method="DELETE", target=location, body=b"", content_type=None)
        )

    def _request(
        self,
        *,
        method: str,
        target: str,
        body: bytes,
        content_type: str | None,
    ) -> GatewayResponse:
        headers = {"Authorization": _basic_authorization(self._connection)}
        if content_type is not None:
            headers["Content-Type"] = content_type
        connection = http.client.HTTPConnection(
            self._connection.host,
            self._connection.port,
            timeout=self._connection.timeout_seconds,
        )
        try:
            connection.request(method, target, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read(_MAX_RESPONSE_BYTES + 1)
            if len(response_body) > _MAX_RESPONSE_BYTES:
                return GatewayResponse(status_code=HTTPStatus.BAD_GATEWAY, body=b"")
            return GatewayResponse(
                status_code=response.status,
                body=response_body,
                content_type=response.getheader("Content-Type"),
                location=response.getheader("Location"),
                etag=response.getheader("ETag"),
            )
        finally:
            connection.close()


@final
class HttpMediaControlGateway:
    """Mutate private MediaMTX paths through its unexposed control listener."""

    def __init__(self, connection: MediaGatewayConnection) -> None:
        """Bind the adapter to one private control listener."""
        self._connection = connection

    async def upsert_path(self, path: MediaPath, source_url: str) -> None:
        """Replace one camera path with an RTSP/TCP source."""
        payload = json.dumps(
            {"source": source_url, "rtspTransport": "tcp", "record": False},
            separators=(",", ":"),
        ).encode()
        response = await to_thread.run_sync(
            partial(
                self._request,
                "POST",
                f"/v3/config/paths/replace/{quote(path, safe='/')}",
                payload,
            )
        )
        if not HTTPStatus.OK <= response < HTTPStatus.MULTIPLE_CHOICES:
            raise MediaControlRequestError(status_code=response)

    async def delete_path(self, path: MediaPath) -> None:
        """Delete one camera path from the private gateway."""
        response = await to_thread.run_sync(
            partial(
                self._request,
                "DELETE",
                f"/v3/config/paths/delete/{quote(path, safe='/')}",
                b"",
            )
        )
        if not HTTPStatus.OK <= response < HTTPStatus.MULTIPLE_CHOICES:
            raise MediaControlRequestError(status_code=response)

    def _request(self, method: str, target: str, body: bytes) -> int:
        headers = {
            "Authorization": _basic_authorization(self._connection),
            "Content-Type": "application/json",
        }
        connection = http.client.HTTPConnection(
            self._connection.host,
            self._connection.port,
            timeout=self._connection.timeout_seconds,
        )
        try:
            connection.request(method, target, body=body, headers=headers)
            response = connection.getresponse()
            _ = response.read(_MAX_RESPONSE_BYTES)
            return response.status
        finally:
            connection.close()


@dataclass(frozen=True, slots=True)
class MediaControlRequestError(Exception):
    """Report only the private control status, never source credentials."""

    status_code: int

    @override
    def __str__(self) -> str:
        """Return a credential-free control failure."""
        return f"Media control request failed with status {self.status_code}"


def _basic_authorization(connection: MediaGatewayConnection) -> str:
    value = f"{connection.username}:{connection.password}".encode()
    return "Basic " + base64.b64encode(value).decode("ascii")
