"""HTTP contract tests through a typed opener boundary."""

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TypeGuard, cast
from urllib.parse import urlsplit
from urllib.request import Request

import pytest
from pydantic import SecretStr

from gods_watching.journeys.http_setup import HttpSetup
from gods_watching.journeys.models import OperatorCredentials, StackError


@dataclass(slots=True)
class ServerState:
    """Record request metadata for the HTTP test double."""

    origin_headers: list[str | None] = field(default_factory=list)
    login_payloads: list[dict[str, object]] = field(default_factory=list)
    camera_payloads: list[dict[str, object]] = field(default_factory=list)
    cookies: list[str | None] = field(default_factory=list)
    camera_ids: list[str] = field(default_factory=list)
    camera_state: str = "online"


@dataclass(slots=True)
class JsonResponse:
    """Provide the small response interface consumed by HttpSetup."""

    status: int
    body: bytes

    def read(self) -> bytes:
        """Return the encoded JSON response body."""
        return self.body

    def close(self) -> None:
        """Match urllib response cleanup without holding external resources."""


class RecordingHTTPServer:
    """Expose status data to requests routed by the in-memory opener."""

    def __init__(self, state: ServerState) -> None:
        self.state: ServerState = state

    def status_payload(self) -> dict[str, object]:
        cameras = [
            {
                "camera_id": camera_id,
                "camera_session_id": None,
                "state": self.state.camera_state,
                "last_ingest_at": None,
                "frame_age_seconds": 0.2,
                "actual_framerate": 25.0,
                "detector_framerate": 25.0,
                "dropped_frames": 0,
                "detector_requests": 1,
                "detector_results": 1,
                "last_error": None,
            }
            for camera_id in self.state.camera_ids
        ]
        return {
            "state": "ready",
            "cameras": cameras,
            "inference_ready": True,
            "persistence_paused": False,
            "worker_updated_at": None,
            "storage_managed_bytes": 0,
            "storage_quota_bytes": 1_000,
            "indexing_queue_depth": 0,
            "last_searchable_latency_seconds": None,
        }


class InMemoryOpener:
    """Route typed urllib Requests to a deterministic product API double."""

    def __init__(self, server: RecordingHTTPServer) -> None:
        self.server: RecordingHTTPServer = server
        self.session_cookie: str | None = None

    def open(self, fullurl: Request, *, timeout: float) -> JsonResponse:
        """Return the product response matching a session, camera, or status route."""
        del timeout
        path = urlsplit(fullurl.full_url).path
        method = fullurl.get_method()
        if path == "/api/session" and method == "GET":
            return JsonResponse(status=401, body=b"{}")
        if path == "/api/session" and method == "POST":
            payload = _request_payload(fullurl)
            self.server.state.login_payloads.append(payload)
            self.server.state.origin_headers.append(fullurl.get_header("Origin"))
            self.session_cookie = "gw_session=test-session-token"
            return _json_response(
                200,
                {
                    "authenticated": True,
                    "idle_expires_at": None,
                    "absolute_expires_at": None,
                },
            )
        if path == "/api/cameras" and method == "POST":
            payload = _request_payload(fullurl)
            self.server.state.camera_payloads.append(payload)
            self.server.state.cookies.append(self.session_cookie)
            index = len(self.server.state.camera_ids) + 1
            camera_id = str(uuid.UUID(int=index))
            self.server.state.camera_ids.append(camera_id)
            source_url = payload.get("source_url")
            if not isinstance(source_url, str):
                return _json_response(422, {"detail": "source_url must be a string"})
            source_port = int(source_url.rsplit(":", maxsplit=1)[1].split("/", maxsplit=1)[0])
            return _json_response(
                201,
                {
                    "camera_id": camera_id,
                    "name": payload.get("name"),
                    "source_host": "127.0.0.1",
                    "source_port": source_port,
                    "detection_enabled": payload.get("detection_enabled"),
                    "detection_threshold": payload.get("detection_threshold"),
                    "version": 1,
                    "deleted_at": None,
                },
            )
        if path == "/api/status" and method == "GET":
            return _json_response(200, self.server.status_payload())
        return _json_response(404, {"detail": "not found"})


@contextmanager
def http_endpoint(
    camera_state: str = "online",
) -> Iterator[tuple[str, ServerState, InMemoryOpener]]:
    """Create a local-origin opener routed through the in-memory API double."""
    state = ServerState(camera_state=camera_state)
    opener = InMemoryOpener(RecordingHTTPServer(state))
    yield "http://journey-test.invalid", state, opener


def test_http_setup_uses_origin_cookie_and_camera_contract_fields() -> None:
    """Session login and camera creation reuse one same-origin session."""
    with http_endpoint() as (base_url, state, opener):
        setup = HttpSetup(opener=opener, base_url=base_url)
        credentials = OperatorCredentials(
            username="fixture-operator",
            password=SecretStr("fixture-password-value"),
        )

        assert setup.probe_session() == 401
        setup.login(credentials)
        camera_ids = setup.register_fixture_cameras(38554)
        setup.wait_cameras_streaming(camera_ids, timeout_s=1.0)

    assert state.origin_headers == [base_url]
    assert state.login_payloads == [
        {"username": "fixture-operator", "password": "fixture-password-value"}
    ]
    assert camera_ids == tuple(state.camera_ids)
    assert state.camera_payloads == [
        {
            "name": f"camera-{number}",
            "source_url": f"rtsp://127.0.0.1:38554/camera-{number}",
            "detection_enabled": True,
            "detection_threshold": 0.5,
        }
        for number in range(1, 5)
    ]
    assert all(cookie == "gw_session=test-session-token" for cookie in state.cookies)


def test_http_setup_times_out_when_fixture_cameras_never_come_online() -> None:
    """The authenticated status contract bounds camera-stream readiness polling."""
    with http_endpoint(camera_state="offline") as (base_url, _state, opener):
        setup = HttpSetup(opener=opener, base_url=base_url)

        with pytest.raises(StackError, match="did not become online"):
            setup.wait_cameras_streaming((str(uuid.UUID(int=1)),), timeout_s=0.01)


def _request_payload(request: Request) -> dict[str, object]:
    body = request.data
    if not isinstance(body, (bytes, bytearray)):
        return {}
    decoded = cast("object", json.loads(body))
    if not _is_json_object(decoded):
        return {}
    return decoded


def _json_response(status: int, payload: dict[str, object]) -> JsonResponse:
    return JsonResponse(status=status, body=json.dumps(payload).encode())


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))
