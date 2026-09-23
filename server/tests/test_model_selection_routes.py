from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import TypeAdapter

from gods_watching.api.camera_routes import (
    AuthenticatedRequest,
    SessionDependency,
)
from gods_watching.api.model_routes import build_model_router
from gods_watching.contracts.model_selection import (
    ModelCatalogEntry,
    ModelSettingsResponse,
    ModelTransitionPhase,
    ModelTransitionResponse,
)
from gods_watching.model_selection.models import (
    ModelNotPreparedError,
    ModelSelectionConflictError,
    TransitionError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    from sqlalchemy.ext.asyncio import AsyncSession
    from starlette.types import Message, Scope

type JsonPrimitive = str | int | float | bool | None
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
_STATUS: Final[TypeAdapter[int]] = TypeAdapter(int)
_JSON_OBJECT: Final[TypeAdapter[dict[str, JsonValue]]] = TypeAdapter(dict[str, JsonValue])


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _catalog_response() -> ModelSettingsResponse:
    transition = ModelTransitionResponse(
        id=uuid4(),
        source_model_id="openai/clip-vit-base-patch16",
        target_model_id="openai/clip-vit-base-patch32",
        phase=ModelTransitionPhase.REINDEXING,
        processed=3,
        total=8,
        skipped=1,
        skip_reasons={"undecodable_crop": 1},
        error=None,
    )
    return ModelSettingsResponse(
        active_model_id="openai/clip-vit-base-patch16",
        maintenance=True,
        models=(
            ModelCatalogEntry(
                model_id="openai/clip-vit-base-patch16",
                display_name="OpenAI CLIP ViT-B/16",
                dimension=512,
                prepared=True,
                reason=None,
            ),
            ModelCatalogEntry(
                model_id="openai/clip-vit-base-patch32",
                display_name="OpenAI CLIP ViT-B/32",
                dimension=512,
                prepared=False,
                reason="identity marker missing",
            ),
            ModelCatalogEntry(
                model_id="openai/clip-vit-large-patch14",
                display_name="OpenAI CLIP ViT-L/14",
                dimension=768,
                prepared=True,
                reason=None,
            ),
        ),
        transition=transition,
    )


@dataclass
class _Database:
    session: AsyncSession = field(default_factory=lambda: cast("AsyncSession", object()))

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        yield self.session


@dataclass
class _ModelSelection:
    response: ModelSettingsResponse
    apply_failure: TransitionError | None = None
    get_calls: int = 0
    applied_model_ids: list[str] = field(default_factory=list)

    async def get(self, session: AsyncSession) -> ModelSettingsResponse:
        del session
        self.get_calls += 1
        return self.response

    async def apply(self, session: AsyncSession, model_id: str) -> ModelSettingsResponse:
        del session
        self.applied_model_ids.append(model_id)
        if self.apply_failure is not None:
            raise self.apply_failure
        return self.response


@dataclass
class _GuardFactory:
    built_for_user_actions: list[bool] = field(default_factory=list)
    calls: list[bool] = field(default_factory=list)

    def __call__(self, *, user_action: bool) -> SessionDependency:
        self.built_for_user_actions.append(user_action)

        async def dependency(request: Request, response: Response) -> AuthenticatedRequest:
            del response
            self.calls.append(user_action)
            if request.headers.get("x-test-session") != "valid":
                raise HTTPException(status_code=401, detail="authentication required")
            if user_action and request.headers.get("origin") != "https://gw.test":
                raise HTTPException(status_code=403, detail="mutation origin is not allowed")
            return AuthenticatedRequest(session_id="test-session")

        return dependency


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, JsonValue] | None = None,
) -> tuple[int, bytes]:
    payload = json.dumps(body).encode() if body is not None else b""
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    if body is not None:
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
    chunks = [
        chunk
        for message in messages
        if message["type"] == "http.response.body"
        and isinstance(chunk := message.get("body"), bytes)
    ]
    return status_code, b"".join(chunks)


def _app(
    model_selection: _ModelSelection,
    guard: _GuardFactory,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        build_model_router(
            database=_Database(),
            model_selection=model_selection,
            require_session=guard,
        )
    )
    return app


def _detail(body: bytes) -> dict[str, JsonValue]:
    payload = _JSON_OBJECT.validate_json(body)
    detail = payload.get("detail")
    assert isinstance(detail, dict)
    return detail


@pytest.mark.anyio
async def test_get_models_returns_the_frozen_catalog_and_transition_shape() -> None:
    service = _ModelSelection(response=_catalog_response())
    guard = _GuardFactory()

    status_code, body = await _request(
        _app(service, guard),
        "GET",
        "/api/settings/models",
        headers={"x-test-session": "valid"},
    )

    assert status_code == 200
    payload = _JSON_OBJECT.validate_json(body)
    assert set(payload) == {"active_model_id", "maintenance", "models", "transition"}
    assert payload["active_model_id"] == "openai/clip-vit-base-patch16"
    assert payload["maintenance"] is True
    models = payload["models"]
    assert isinstance(models, list)
    assert models[1] == {
        "model_id": "openai/clip-vit-base-patch32",
        "display_name": "OpenAI CLIP ViT-B/32",
        "dimension": 512,
        "prepared": False,
        "reason": "identity marker missing",
    }
    transition = payload["transition"]
    assert isinstance(transition, dict)
    assert transition["phase"] == "reindexing"
    assert transition["skip_reasons"] == {"undecodable_crop": 1}
    assert service.get_calls == 1


@pytest.mark.anyio
async def test_apply_returns_202_and_catalog_status_for_accepted_transition() -> None:
    service = _ModelSelection(response=_catalog_response())
    guard = _GuardFactory()

    status_code, body = await _request(
        _app(service, guard),
        "POST",
        "/api/settings/models/apply",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"model_id": "openai/clip-vit-base-patch32"},
    )

    assert status_code == 202
    assert _JSON_OBJECT.validate_json(body)["active_model_id"] == (
        "openai/clip-vit-base-patch16"
    )
    assert service.applied_model_ids == ["openai/clip-vit-base-patch32"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        (ModelNotPreparedError("unknown", code="unknown_model"), "unknown_model"),
        (ModelNotPreparedError("marker missing", code="model_not_prepared"), "model_not_prepared"),
    ],
)
async def test_apply_maps_unknown_and_unprepared_models_to_structured_422(
    failure: TransitionError,
    expected_code: str,
) -> None:
    service = _ModelSelection(response=_catalog_response(), apply_failure=failure)
    guard = _GuardFactory()

    status_code, body = await _request(
        _app(service, guard),
        "POST",
        "/api/settings/models/apply",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"model_id": "openai/clip-vit-base-patch32"},
    )

    assert status_code == 422
    assert _detail(body)["code"] == expected_code


@pytest.mark.anyio
async def test_apply_maps_active_transition_to_409() -> None:
    service = _ModelSelection(
        response=_catalog_response(),
        apply_failure=ModelSelectionConflictError("already active"),
    )
    guard = _GuardFactory()

    status_code, body = await _request(
        _app(service, guard),
        "POST",
        "/api/settings/models/apply",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"model_id": "openai/clip-vit-base-patch32"},
    )

    assert status_code == 409
    assert _detail(body)["code"] == "model_transition_conflict"


@pytest.mark.anyio
async def test_model_routes_require_auth_and_distinguish_passive_get_from_mutation() -> None:
    service = _ModelSelection(response=_catalog_response())
    guard = _GuardFactory()
    app = _app(service, guard)

    denied_get, _ = await _request(app, "GET", "/api/settings/models")
    denied_apply, _ = await _request(
        app,
        "POST",
        "/api/settings/models/apply",
        body={"model_id": "openai/clip-vit-base-patch32"},
    )
    cross_origin_apply, _ = await _request(
        app,
        "POST",
        "/api/settings/models/apply",
        headers={"x-test-session": "valid", "origin": "https://evil.test"},
        body={"model_id": "openai/clip-vit-base-patch32"},
    )
    accepted_get, _ = await _request(
        app,
        "GET",
        "/api/settings/models",
        headers={"x-test-session": "valid"},
    )

    assert denied_get == 401
    assert denied_apply == 401
    assert cross_origin_apply == 403
    assert accepted_get == 200
    assert guard.built_for_user_actions == [False, True]
    assert guard.calls == [False, True, True, False]
    assert service.applied_model_ids == []
