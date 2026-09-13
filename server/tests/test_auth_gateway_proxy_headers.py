"""Regression coverage for the Uvicorn-to-auth client identity boundary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import anyio
import pytest
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from gods_watching.verification.scenarios import task12_driver

if TYPE_CHECKING:
    from uvicorn._types import (
        ASGIReceiveCallable,
        ASGIReceiveEvent,
        ASGISendCallable,
        ASGISendEvent,
        HTTPScope,
        Scope,
    )


def _client_host_after_default_proxy_headers(forwarded_for: str) -> str:
    observed_client: list[str] = []

    async def application(
        scope: Scope,
        receive: ASGIReceiveCallable,
        send: ASGISendCallable,
    ) -> None:
        del receive, send
        assert scope["type"] == "http"
        http_scope: HTTPScope = scope
        client = http_scope["client"]
        assert client is not None
        observed_client.append(client[0])

    async def receive() -> ASGIReceiveEvent:
        return {"type": "http.disconnect"}

    async def send(message: ASGISendEvent) -> None:
        del message

    async def exercise() -> None:
        wrapped = ProxyHeadersMiddleware(application)
        scope: HTTPScope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/api/session",
            "raw_path": b"/api/session",
            "query_string": b"",
            "root_path": "",
            "headers": [(b"x-forwarded-for", forwarded_for.encode("ascii"))],
            "client": ("127.0.0.1", 50123),
            "server": ("127.0.0.1", 8000),
        }
        await wrapped(scope, receive, send)

    anyio.run(exercise)
    return observed_client[0]


@pytest.mark.parametrize(
    ("forwarded_for", "expected_rewritten_client"),
    [
        ("malformed-client", "malformed-client"),
        ("2001:0db8:0:0:0:0:0:7, 192.0.2.9", "192.0.2.9"),
    ],
)
def test_default_uvicorn_proxy_headers_destroy_direct_peer_boundary(
    forwarded_for: str,
    expected_rewritten_client: str,
) -> None:
    # Given: a configured loopback gateway and an attacker-controlled forwarded header

    # When: Uvicorn's default proxy middleware receives the request before FastAPI
    rewritten_client = _client_host_after_default_proxy_headers(forwarded_for)

    # Then: the original gateway peer is unavailable to the auth policy
    assert rewritten_client == expected_rewritten_client


def test_task12_uvicorn_command_preserves_direct_peer_for_auth_policy() -> None:
    # Given: the isolated Task12 FastAPI launch boundary
    repository_root = Path("/workspace")

    # When: the Uvicorn command is assembled
    command = task12_driver.application_uvicorn_command(
        repository_root=repository_root,
        port=29191,
        keyfile=repository_root / "server.key",
        certfile=repository_root / "server.crt",
    )

    # Then: Uvicorn cannot rewrite the direct peer before configured-gateway handling
    assert "--no-proxy-headers" in command
