"""Establish Journey preconditions through the product's HTTP contracts."""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Final, Protocol, TypeGuard, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request

from pydantic import ValidationError

from gods_watching.cameras import ProbeFailureCode
from gods_watching.contracts.cameras import CameraCreateRequest, CameraResponse
from gods_watching.contracts.session import LoginRequest, SessionResponse
from gods_watching.contracts.status import StatusResponse

from .models import OperatorCredentials, StackError, raise_stack_error

_HTTP_TIMEOUT_S: Final = 5.0
_CAMERA_CREATE_TIMEOUT_S: Final = 30.0
_CAMERA_RETRY_INTERVAL_S: Final = 5.0
_CAMERA_RETRY_BUDGET_S: Final = 180.0
_CAMERA_POLL_INTERVAL_S: Final = 2.0
_CAMERA_COUNT: Final = 4
_HTTP_OK: Final = 200
_HTTP_CREATED: Final = 201
_HTTP_CONFLICT: Final = 409
_HTTP_UNPROCESSABLE: Final = 422
_CAMERA_NAME_CONFLICT: Final = "camera_name_conflict"
_PROBE_FAILURE_CODES: Final = frozenset(code.value for code in ProbeFailureCode)


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
    clock: Callable[[], float] = monotonic
    sleep: Callable[[float], None] = sleep

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
            camera_ids.append(
                self._register_fixture_camera(
                    name,
                    cast("dict[str, object]", request.model_dump(mode="json")),
                )
            )
        return tuple(camera_ids)

    def _register_fixture_camera(self, name: str, payload: Mapping[str, object]) -> str:
        deadline = self.clock() + _CAMERA_RETRY_BUDGET_S
        last_reason = "camera creation did not complete"
        retried = False
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                _raise_camera_retry_exhausted(name, last_reason)
            try:
                status_code, response_payload = self._request_json(
                    "POST",
                    "/api/cameras",
                    payload,
                    origin=self.base_url,
                    timeout=min(_CAMERA_CREATE_TIMEOUT_S, remaining),
                    include_http_errors=True,
                )
            except StackError as error:
                cause = error.__cause__
                if not _is_retryable_connection_failure(cause):
                    raise
                last_reason = _connection_failure_reason(cause)
            else:
                if status_code == _HTTP_CREATED:
                    return _camera_id_from_response(response_payload)
                error_code, error_message = _api_error_details(response_payload)
                if (
                    retried
                    and status_code == _HTTP_CONFLICT
                    and error_code == _CAMERA_NAME_CONFLICT
                ):
                    return self._camera_id_by_name(name)
                if (
                    status_code == _HTTP_UNPROCESSABLE
                    and error_code in _PROBE_FAILURE_CODES
                ):
                    last_reason = _format_api_error(status_code, error_code, error_message)
                else:
                    reason = _format_api_error(status_code, error_code, error_message)
                    raise_stack_error(f"fixture camera {name} registration returned {reason}")

            remaining = deadline - self.clock()
            if remaining <= _CAMERA_RETRY_INTERVAL_S:
                _raise_camera_retry_exhausted(name, last_reason)
            self.sleep(_CAMERA_RETRY_INTERVAL_S)
            retried = True

    def _camera_id_by_name(self, name: str) -> str:  # noqa: RET503
        request = Request(  # noqa: S310
            f"{self.base_url.rstrip('/')}/api/cameras",
            headers={"Accept": "application/json"},
            method="GET",
        )
        try:
            response = self.opener.open(request, timeout=_HTTP_TIMEOUT_S)
            if response is None:
                raise_stack_error("GET /api/cameras returned no response")
            try:
                status_code = response.status
                decoded = cast("object", json.loads(response.read()))
            finally:
                response.close()
        except HTTPError as error:
            raise_stack_error(f"GET /api/cameras returned HTTP {error.code}", cause=error)
        except (OSError, TimeoutError, URLError) as error:
            raise_stack_error("GET /api/cameras request failed", cause=error)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise_stack_error("GET /api/cameras returned invalid JSON", cause=error)
        if status_code != _HTTP_OK:
            raise_stack_error(f"GET /api/cameras returned HTTP {status_code}")
        if not isinstance(decoded, list):
            raise_stack_error("GET /api/cameras response must be a JSON array")
        for item in cast("list[object]", decoded):
            try:
                camera = CameraResponse.model_validate(item)
            except ValidationError as error:
                raise_stack_error("GET /api/cameras returned an invalid camera", cause=error)
            if camera.name == name:
                return str(camera.camera_id)
        raise_stack_error(f"fixture camera {name} was not found after a name conflict")

    def wait_cameras_streaming(
        self,
        camera_ids: tuple[str, ...],
        timeout_s: float = 180,
    ) -> None:
        """Wait until all requested cameras report the API's online runtime state."""
        if not camera_ids:
            return
        expected_ids = set(camera_ids)
        deadline = self.clock() + timeout_s
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
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise_stack_error("fixture cameras did not become online before timeout")
            self.sleep(min(_CAMERA_POLL_INTERVAL_S, remaining))

    def _request_json(  # noqa: PLR0913
        self,
        method: str,
        path: str,
        payload: Mapping[str, object] | None = None,
        *,
        origin: str | None = None,
        timeout: float = _HTTP_TIMEOUT_S,
        include_http_errors: bool = False,
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
            response = self.opener.open(request, timeout=timeout)
            if response is None:
                raise_stack_error(f"{method} {path} returned no response")
            try:
                status_code = response.status
                decoded = cast("object", json.loads(response.read()))
            finally:
                response.close()
        except HTTPError as error:
            if not include_http_errors:
                raise_stack_error(f"{method} {path} returned HTTP {error.code}", cause=error)
            try:
                decoded = cast("object", json.loads(error.read()))
            except (OSError, UnicodeError, json.JSONDecodeError) as parse_error:
                raise_stack_error(f"{method} {path} returned invalid JSON", cause=parse_error)
            finally:
                error.close()
            status_code = error.code
        except (OSError, TimeoutError, URLError) as error:
            raise_stack_error(f"{method} {path} request failed", cause=error)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise_stack_error(f"{method} {path} returned invalid JSON", cause=error)
        if not _is_json_object(decoded):
            raise_stack_error(f"{method} {path} response must be a JSON object")
        return status_code, decoded


def _camera_id_from_response(payload: dict[str, object]) -> str:
    try:
        camera = CameraResponse.model_validate(payload)
    except ValidationError as error:
        raise_stack_error("fixture camera registration returned an invalid response", cause=error)
    return str(camera.camera_id)


def _api_error_details(payload: Mapping[str, object]) -> tuple[str | None, str | None]:
    detail = payload.get("detail")
    if not _is_json_object(detail):
        return None, None
    code = detail.get("code")
    message = detail.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


def _format_api_error(
    status_code: int,
    code: str | None,
    message: str | None,
) -> str:
    reason = f"HTTP {status_code}"
    if code is not None:
        reason = f"{reason} {code}"
    if message is not None:
        reason = f"{reason}: {message}"
    return reason


def _is_retryable_connection_failure(error: BaseException | None) -> bool:
    return isinstance(error, (OSError, TimeoutError, URLError)) and not isinstance(
        error,
        HTTPError,
    )


def _connection_failure_reason(error: BaseException | None) -> str:
    reason: object = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, TimeoutError):
        return "request timed out"
    if isinstance(reason, OSError):
        detail = reason.strerror or type(reason).__name__
        return f"connection failed: {detail}"
    if isinstance(reason, str):
        return f"connection failed: {reason}"
    return "connection failed"


def _raise_camera_retry_exhausted(name: str, last_reason: str) -> None:
    reason = f"{_CAMERA_RETRY_BUDGET_S:g} seconds: {last_reason}"
    message = f"fixture camera {name} registration retries exhausted within {reason}"
    raise_stack_error(message)


def _is_json_object(value: object) -> TypeGuard[dict[str, object]]:
    if not isinstance(value, dict):
        return False
    return all(isinstance(key, str) for key in cast("dict[object, object]", value))


__all__ = ["HttpSetup"]
