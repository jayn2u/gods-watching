import json
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import final
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Message, Scope

from gods_watching.api.appearance_routes import build_appearance_router
from gods_watching.api.camera_routes import AuthenticatedRequest, SessionDependencyFactory
from gods_watching.api.search_routes import build_search_router
from gods_watching.contracts.appearances import AppearanceResponse, BoundingBox
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.search import SearchRequest, SearchResponse, TextSearchRequest
from gods_watching.search import (
    CropChangedDuringReadError,
    CropPayload,
    CropUnavailableError,
    SearchInferenceUnavailableError,
    SearchSeedNotFoundError,
    SearchTextInvalidError,
    UnknownCameraError,
)

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
_JSON_OBJECT = TypeAdapter(dict[str, JsonValue])
_STATUS = TypeAdapter(int)
_HEADERS = TypeAdapter(list[tuple[bytes, bytes]])


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True, slots=True)
class _HttpResponse:
    status_code: int
    body: bytes
    headers: dict[str, str]


@dataclass
class _Database:
    session: AsyncSession = field(default_factory=AsyncSession)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        yield self.session


@final
class _GuardFactory:
    def __init__(self) -> None:
        self.calls: list[bool] = []

    def __call__(self, *, user_action: bool) -> Callable[..., Awaitable[AuthenticatedRequest]]:
        async def dependency(request: Request) -> AuthenticatedRequest:
            self.calls.append(user_action)
            if request.headers.get("x-test-session") != "valid":
                raise HTTPException(status_code=401, detail="authentication required")
            if user_action and request.headers.get("origin") != "https://gw.test":
                raise HTTPException(status_code=403, detail="mutation origin is not allowed")
            return AuthenticatedRequest(session_id=str(uuid4()))

        return dependency


@final
class _SearchService:
    def __init__(self, response: SearchResponse) -> None:
        self.response = response
        self.failure: (
            UnknownCameraError
            | SearchSeedNotFoundError
            | SearchInferenceUnavailableError
            | SearchTextInvalidError
            | None
        ) = None
        self.calls: list[SearchRequest] = []

    async def search(self, session: AsyncSession, request: SearchRequest) -> SearchResponse:
        del session
        self.calls.append(request)
        if self.failure is not None:
            raise self.failure
        return self.response


@final
class _AppearanceLookup:
    def __init__(self, detail: AppearanceResponse, crop: CropPayload) -> None:
        self.detail = detail
        self.crop = crop
        self.failure: CropUnavailableError | CropChangedDuringReadError | None = None
        self.detail_calls: list[UUID] = []
        self.crop_calls: list[UUID] = []

    async def get_detail(self, session: AsyncSession, appearance_id: UUID) -> AppearanceResponse:
        del session
        self.detail_calls.append(appearance_id)
        if isinstance(self.failure, CropUnavailableError):
            raise self.failure
        if isinstance(self.failure, CropChangedDuringReadError):
            raise self.failure
        return self.detail

    async def get_crop(self, session: AsyncSession, appearance_id: UUID) -> CropPayload:
        del session
        self.crop_calls.append(appearance_id)
        if self.failure is not None:
            raise self.failure
        return self.crop


def _appearance() -> AppearanceResponse:
    now = datetime(2026, 9, 8, 3, 0, tzinfo=UTC)
    camera_id = CameraId(uuid4())
    return AppearanceResponse(
        appearance_id=AppearanceId(uuid4()),
        camera_id=camera_id,
        camera_name="Archived camera",
        session_id=CameraSessionId(uuid4()),
        track_id=4,
        first_seen=now,
        last_seen=now + timedelta(seconds=5),
        ended_at=now + timedelta(seconds=5),
        representative_version=2,
        bounding_box=BoundingBox(x_min=10, y_min=20, x_max=110, y_max=220),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.91,
        crop_quality=42.5,
        model_id="fixture/clip",
        model_revision="fixture-revision",
        similarity=None,
    )


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, JsonValue] | None = None,
) -> _HttpResponse:
    payload = json.dumps(body).encode() if body is not None else b""
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    if body is not None and "content-type" not in {key.lower() for key in (headers or {})}:
        request_headers.append((b"content-type", b"application/json"))
    messages: list[Message] = []
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    scope: Scope = {
        "type": "http",
        "http_version": "1.1",
        "method": method,
        "scheme": "https",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": request_headers,
        "client": ("127.0.0.1", 50000),
        "server": ("gw.test", 443),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    status_code = _STATUS.validate_python(start.get("status"))
    raw_headers = _HEADERS.validate_python(start.get("headers"))
    response_headers = {key.decode().lower(): value.decode() for key, value in raw_headers}
    body_chunks = [
        chunk
        for message in messages
        if message["type"] == "http.response.body"
        and isinstance(chunk := message.get("body"), bytes)
    ]
    return _HttpResponse(status_code, b"".join(body_chunks), response_headers)


def _app(
    database: _Database,
    search: _SearchService,
    lookup: _AppearanceLookup,
    guard: SessionDependencyFactory,
) -> FastAPI:
    app = FastAPI()
    app.include_router(build_search_router(database=database, search=search, require_session=guard))
    app.include_router(
        build_appearance_router(database=database, lookup=lookup, require_session=guard)
    )
    return app


@pytest.mark.anyio
async def test_search_requires_auth_refreshes_activity_and_rejects_cross_origin_post() -> None:
    appearance = _appearance()
    database = _Database()
    search = _SearchService(SearchResponse(mode="browse", results=(appearance,)))
    lookup = _AppearanceLookup(appearance, CropPayload(appearance, b"jpeg", "aa/bb/key.jpg", 2))
    guard = _GuardFactory()
    app = _app(database, search, lookup, guard)

    denied = await _request(app, "POST", "/api/search", body={"mode": "browse"})
    cross_origin = await _request(
        app,
        "POST",
        "/api/search",
        headers={"x-test-session": "valid", "origin": "https://evil.test"},
        body={"mode": "browse"},
    )
    accepted = await _request(
        app,
        "POST",
        "/api/search",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"mode": "browse"},
    )
    text = await _request(
        app,
        "POST",
        "/api/search",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"mode": "text", "query": " person   near entrance "},
    )
    similar = await _request(
        app,
        "POST",
        "/api/search",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"mode": "similar", "appearance_id": str(appearance.appearance_id)},
    )

    assert denied.status_code == 401
    assert cross_origin.status_code == 403
    assert [response.status_code for response in (accepted, text, similar)] == [200, 200, 200]
    assert [request.mode for request in search.calls] == ["browse", "text", "similar"]
    text_request = search.calls[1]
    assert isinstance(text_request, TextSearchRequest)
    assert text_request.query == "person near entrance"
    assert guard.calls == [True, True, True, True, True]


@pytest.mark.anyio
async def test_search_maps_typed_service_failures_without_leaking_failure_reasons() -> None:
    appearance = _appearance()
    failures = (
        (UnknownCameraError((uuid4(),)), 422, "unknown_camera"),
        (SearchSeedNotFoundError(uuid4()), 404, "appearance_not_found"),
        (SearchInferenceUnavailableError("secret triton address"), 503, "inference_unavailable"),
        (SearchTextInvalidError("clip_text_too_many_tokens: 78 exceeds 77"), 422, "invalid_text"),
    )

    for failure, expected_status, expected_code in failures:
        database = _Database()
        search = _SearchService(SearchResponse(mode="browse", results=(appearance,)))
        search.failure = failure
        lookup = _AppearanceLookup(appearance, CropPayload(appearance, b"jpeg", "aa/bb/key.jpg", 2))
        guard = _GuardFactory()
        response = await _request(
            _app(database, search, lookup, guard),
            "POST",
            "/api/search",
            headers={"x-test-session": "valid", "origin": "https://gw.test"},
            body={"mode": "browse"},
        )

        assert response.status_code == expected_status
        payload = _JSON_OBJECT.validate_json(response.body)
        detail = payload.get("detail")
        assert isinstance(detail, dict)
        assert detail.get("code") == expected_code
        assert "secret triton address" not in response.body.decode()
        assert "78 exceeds 77" not in response.body.decode()


@pytest.mark.anyio
@pytest.mark.parametrize(
    "body",
    [
        {"mode": "text", "query": "   "},
        {"mode": "text", "query": "person\nwith bag"},
        {"mode": "text", "query": "검은 옷"},
        {
            "mode": "browse",
            "from": "2026-09-08T04:00:00+00:00",
            "to": "2026-09-08T03:00:00+00:00",
        },
    ],
)
async def test_search_boundary_validation_rejects_invalid_input_before_service_call(
    body: Mapping[str, JsonValue],
) -> None:
    appearance = _appearance()
    database = _Database()
    search = _SearchService(SearchResponse(mode="text", results=(appearance,)))
    lookup = _AppearanceLookup(appearance, CropPayload(appearance, b"jpeg", "aa/bb/key.jpg", 2))
    guard = _GuardFactory()
    app = _app(database, search, lookup, guard)

    response = await _request(
        app,
        "POST",
        "/api/search",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body=body,
    )

    assert response.status_code == 422
    assert search.calls == []


@pytest.mark.anyio
async def test_detail_and_crop_are_passive_and_crop_response_is_private_no_store() -> None:
    appearance = _appearance()
    database = _Database()
    search = _SearchService(SearchResponse(mode="browse", results=(appearance,)))
    lookup = _AppearanceLookup(
        appearance,
        CropPayload(appearance, b"real-jpeg-bytes", "aa/bb/key.jpg", 2),
    )
    guard = _GuardFactory()
    app = _app(database, search, lookup, guard)
    appearance_id = str(appearance.appearance_id)

    detail = await _request(
        app,
        "GET",
        f"/api/appearances/{appearance_id}",
        headers={"x-test-session": "valid"},
    )
    crop = await _request(
        app,
        "GET",
        f"/api/appearances/{appearance_id}/crop",
        headers={"x-test-session": "valid"},
    )

    assert detail.status_code == 200
    assert json.loads(detail.body)["camera_name"] == "Archived camera"
    assert crop.status_code == 200
    assert crop.body == b"real-jpeg-bytes"
    assert crop.headers["content-type"] == "image/jpeg"
    assert crop.headers["cache-control"] == "private, no-store, max-age=0"
    assert crop.headers["x-representative-version"] == "2"
    assert crop.headers["etag"].startswith('"')
    assert guard.calls == [False, False]
    assert lookup.detail_calls == [UUID(appearance_id)]
    assert lookup.crop_calls == [UUID(appearance_id)]


@pytest.mark.anyio
async def test_detail_and_crop_map_missing_and_pointer_race_without_serving_bytes() -> None:
    appearance = _appearance()
    database = _Database()
    search = _SearchService(SearchResponse(mode="browse", results=(appearance,)))
    lookup = _AppearanceLookup(
        appearance,
        CropPayload(appearance, b"must-not-be-served", "aa/bb/key.jpg", 2),
    )
    guard = _GuardFactory()
    app = _app(database, search, lookup, guard)
    appearance_id = str(appearance.appearance_id)

    lookup.failure = CropUnavailableError(UUID(appearance_id))
    missing = await _request(
        app,
        "GET",
        f"/api/appearances/{appearance_id}/crop",
        headers={"x-test-session": "valid"},
    )
    lookup.failure = CropChangedDuringReadError(UUID(appearance_id), 2)
    raced = await _request(
        app,
        "GET",
        f"/api/appearances/{appearance_id}/crop",
        headers={"x-test-session": "valid"},
    )

    assert missing.status_code == 404
    assert raced.status_code == 409
    assert b"must-not-be-served" not in missing.body
    assert b"must-not-be-served" not in raced.body
