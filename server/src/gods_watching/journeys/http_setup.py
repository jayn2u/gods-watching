"""Establish Journey preconditions through the product's HTTP contracts."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Final, Protocol, TypeGuard, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from pydantic import ValidationError

from gods_watching.contracts.cameras import CameraCreateRequest, CameraResponse
from gods_watching.contracts.session import LoginRequest, SessionResponse
from gods_watching.contracts.status import StatusResponse

from .models import OperatorCredentials, raise_stack_error

_HTTP_TIMEOUT_S: Final = 5.0
_CAMERA_POLL_INTERVAL_S: Final = 2.0
_CAMERA_COUNT: Final = 4
_HTTP_OK: Final = 200
_HTTP_CREATED: Final = 201


class JourneyHttpResponse(Protocol):
    """Expose the response fields and methods used by HTTP setup."""

    status: int

    def read(self) -> bytes:
        """Read the response body."""
        ...

    def close(self) -> None:
        """Release the response's underlying connection resources."""
        ...


class JourneyHttpOpener(Protocol):
    """Open product API requests and return their response contracts."""

    def open(
        self,
        fullurl: Request,
        *,
        timeout: float,
    ) -> JourneyHttpResponse | None:
        """Open a prepared request with a bounded timeout."""
        ...


@dataclass(frozen=True, slots=True)
class HttpSetup:
    """Keep one cookie-aware opener for login and deterministic API setup."""

    opener: JourneyHttpOpener
    base_url: str

    def __post_init__(self) -> None:
        """Reject non-HTTP origins before constructing requests."""
        origin = urlsplit(self.base_url)
        if origin.scheme not in {"http", "https"} or not origin.netloc:
            raise_stack_error("HTTP setup base_url must use HTTP or HTTPS")

    def probe_session(self) -> int | None:
        """Return the session route status used to gate stack readiness."""
        request = Request(  # noqa: S310
            f"{self.base_url.rstrip('/')}/api/session",
            method="GET",
        )
        try:
            response = self.opener.open(request, timeout=_HTTP_TIMEOUT_S)
            if response is None:
                return None
            try:
                return response.status
            finally:
                response.close()
        except HTTPError as error:
            return error.code
        except (OSError, TimeoutError, URLError):
            return None

    def login(self, credentials: OperatorCredentials) -> None:
        """Create an operator session with the configured same-origin request."""
        request = LoginRequest.model_validate(
            {
                "username": credentials.username,
                "password": credentials.password.get_secret_value(),
            }
        )
        status_code, payload = self._request_json(
            "POST",
            "/api/session",
            {
                "username": request.username,
                "password": request.password.get_secret_value(),
            },
            origin=self.base_url,
        )
        if status_code != _HTTP_OK:
            raise_stack_error(f"operator login returned HTTP {status_code}")
        try:
            session = SessionResponse.model_validate(payload)
        except ValidationError as error:
            raise_stack_error("operator login returned an invalid session response", cause=error)
        if not session.authenticated:
            raise_stack_error("operator login did not establish an authenticated session")

    def register_fixture_cameras(self, rtsp_port: int) -> tuple[str, ...]:
        """Register the four loopback RTSP fixtures through CameraCreateRequest."""
        camera_ids: list[str] = []
        for number in range(1, _CAMERA_COUNT + 1):
            name = f"camera-{number}"
            request = CameraCreateRequest.model_validate(
                {
                    "name": name,
                    "source_url": f"rtsp://127.0.0.1:{rtsp_port}/{name}",
                    "detection_enabled": True,
                    "detection_threshold": 0.5,
                }
            )
            status_code, payload = self._request_json(
                "POST",
                "/api/cameras",
                cast("dict[str, object]", request.model_dump(mode="json")),
                origin=self.base_url,
            )
            if status_code != _HTTP_CREATED:
                raise_stack_error(f"fixture camera registration returned HTTP {status_code}")
            try:
                camera = CameraResponse.model_validate(payload)
            except ValidationError as error:
                raise_stack_error(
                    "fixture camera registration returned an invalid response",
                    cause=error,
                )
            camera_ids.append(str(camera.camera_id))
        return tuple(camera_ids)

    def wait_cameras_streaming(
        self,
        camera_ids: tuple[str, ...],
        timeout_s: float = 180,
    ) -> None:
        """Wait until all requested cameras report the API's online runtime state."""
        if not camera_ids:
            return
        expected_ids = set(camera_ids)
        deadline = monotonic() + timeout_s
        while True:
            _status_code, payload = self._request_json("GET", "/api/status")
            try:
                status = StatusResponse.model_validate(payload)
            except ValidationError as error:
                raise_stack_error(
                    "camera status endpoint returned an invalid response",
                    cause=error,
                )
            online_ids = {
                str(camera.camera_id)
                for camera in status.cameras
                if camera.state == "online"
            }
            if expected_ids.issubset(online_ids):
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise_stack_error("fixture cameras did not become online before timeout")
            sleep(min(_CAMERA_POLL_INTERVAL_S, remaining))

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        origin: str | None = None,
    ) -> tuple[int, dict[str, object]]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if origin is not None:
            headers["Origin"] = origin
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(  # noqa: S310
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            response = self.opener.open(request, timeout=_HTTP_TIMEOUT_S)
            if response is None:
                raise_stack_error(f"{method} {path} returned no response")
            try:
                status_code = response.status
                decoded = cast("object", json.loads(response.read()))
            finally:
                response.close()
        except HTTPError as error:
            raise_stack_error(f"{method} {path} returned HTTP {error.code}", cause=error)
        except (OSError, TimeoutError, URLError) as error:
            raise_stack_error(f"{method} {path} request failed", cause=error)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise_stack_error(f"{method} {path} returned invalid JSON", cause=error)
        if not _is_json_object(decoded):
            raise_stack_error(f"{method} {path} response must be a JSON object")
        return status_code, decoded


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


__all__ = ["HttpSetup"]
