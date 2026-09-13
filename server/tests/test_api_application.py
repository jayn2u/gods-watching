from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal

import pytest
from cryptography.fernet import Fernet
from fastapi.openapi.models import (
    OpenAPI,
    Operation,
    PathItem,
    RequestBody,
    Response,
    Schema,
)
from pydantic import TypeAdapter

from gods_watching.api import ApiDependencies, ApiSettings, create_app
from gods_watching.api.camera_runtime import CameraRuntime
from gods_watching.auth import AuthService
from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.media import GatewayResponse, MediaPath, WhepProxyService
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.settings import SettingsService
from gods_watching.storage import CredentialCipher, CropObjectStore, Database, StorageRepository

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from fastapi import FastAPI
    from starlette.types import Message, Scope

_STATUS = TypeAdapter(int)
_OPENAPI_DOCUMENT: Final[TypeAdapter[OpenAPI]] = TypeAdapter(OpenAPI)
_PATH_ITEM: Final[TypeAdapter[PathItem]] = TypeAdapter(PathItem)
_OPERATION: Final[TypeAdapter[Operation]] = TypeAdapter(Operation)
_RESPONSE: Final[TypeAdapter[Response]] = TypeAdapter(Response)
_REQUEST_BODY: Final[TypeAdapter[RequestBody]] = TypeAdapter(RequestBody)
_SCHEMA: Final[TypeAdapter[Schema]] = TypeAdapter(Schema)

type HttpMethod = Literal["get", "post", "patch"]


class _Gateway:
    async def create(self, path: MediaPath, offer: bytes) -> GatewayResponse:
        del path, offer
        return GatewayResponse(status_code=500, body=b"")

    async def patch(self, location: str, fragment: bytes) -> GatewayResponse:
        del location, fragment
        return GatewayResponse(status_code=500, body=b"")

    async def delete(self, location: str) -> GatewayResponse:
        del location
        return GatewayResponse(status_code=500, body=b"")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _dependencies(tmp_path: Path) -> tuple[ApiDependencies, Database, TritonClipTransport]:
    database = Database.connect("postgresql+asyncpg://operator:secret@127.0.0.1:1/gw")
    transport = TritonClipTransport("127.0.0.1:1")
    repository = SearchRepository()
    storage = StorageRepository(CredentialCipher(Fernet.generate_key()))
    return (
        ApiDependencies(
            database=database,
            auth=AuthService(database),
            cameras=CameraService(CameraRepository(storage)),
            settings=SettingsService(),
            whep=WhepProxyService(
                gateway=_Gateway(),
                resolve_path=lambda camera_id: MediaPath(f"camera/{camera_id}"),
            ),
            camera_runtime=CameraRuntime(detector=None, media=None),
            config=ApiSettings(public_origin="https://gw.test"),
            search=SearchService(repository, ClipAdapter(transport)),
            appearance=AppearanceLookupService(repository, CropObjectStore(tmp_path / "crops")),
            clip_lifecycle=transport,
        ),
        database,
        transport,
    )


async def _request_status(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: bytes = b'{"mode":"browse"}',
) -> int:
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
    request_headers.append((b"content-type", b"application/json"))
    messages: list[Message] = []
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

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
    return _STATUS.validate_python(start.get("status"))


@pytest.mark.anyio
async def test_default_app_composes_search_and_appearance_auth_boundaries(tmp_path: Path) -> None:
    dependencies, database, transport = _dependencies(tmp_path)
    try:
        app = create_app(dependencies)
        anonymous = await _request_status(
            app,
            "POST",
            "/api/search",
            headers={"origin": "https://gw.test"},
        )
        cross_origin = await _request_status(
            app,
            "POST",
            "/api/search",
            headers={"origin": "https://evil.test", "cookie": "gw_session=installed-session"},
        )

        assert anonymous == 401
        assert cross_origin == 403
    finally:
        await transport.__aexit__(None, None, None)
        await database.close()


@pytest.mark.anyio
async def test_default_app_openapi_pins_task12_route_boundaries(tmp_path: Path) -> None:
    dependencies, database, transport = _dependencies(tmp_path)
    try:
        app = create_app(dependencies)
        document = _OPENAPI_DOCUMENT.validate_python(app.openapi())
        assert document.paths is not None
        paths = document.paths
        task12_paths = (
            "/api/session",
            "/api/session/activity",
            "/api/cameras",
            "/api/cameras/test",
            "/api/cameras/{camera_id}",
            "/api/settings",
            "/api/search",
            "/api/appearances/{appearance_id}",
            "/api/appearances/{appearance_id}/crop",
            "/api/live/{camera_id}/whep",
            "/api/live/{camera_id}/whep/{resource_id}",
        )
        assert set(task12_paths).issubset(paths)

        response_contracts: tuple[tuple[str, HttpMethod, str, str], ...] = (
            ("/api/session", "get", "200", "SessionResponse"),
            ("/api/session", "post", "200", "SessionResponse"),
            ("/api/session/activity", "post", "200", "SessionResponse"),
            ("/api/cameras", "post", "201", "CameraResponse"),
            ("/api/cameras/test", "post", "200", "CameraTestResponse"),
            ("/api/cameras/{camera_id}", "patch", "200", "CameraResponse"),
            ("/api/settings", "get", "200", "SettingsResponse"),
            ("/api/settings", "patch", "200", "SettingsResponse"),
            ("/api/search", "post", "200", "SearchResponse"),
            ("/api/appearances/{appearance_id}", "get", "200", "AppearanceResponse"),
        )
        for path, method, status_code, expected_schema in response_contracts:
            path_item = _PATH_ITEM.validate_python(paths[path])
            operation = _OPERATION.validate_python(
                {"get": path_item.get, "post": path_item.post, "patch": path_item.patch}[method]
            )
            assert operation.responses is not None
            response = _RESPONSE.validate_python(operation.responses[status_code])
            assert response.content is not None
            schema = _SCHEMA.validate_python(response.content["application/json"].schema_)
            assert schema.ref == f"#/components/schemas/{expected_schema}"

        camera_path = _PATH_ITEM.validate_python(paths["/api/cameras"])
        assert camera_path.get is not None
        camera_get = _OPERATION.validate_python(camera_path.get)
        assert camera_get.responses is not None
        camera_list_response = _RESPONSE.validate_python(camera_get.responses["200"])
        assert camera_list_response.content is not None
        camera_list_schema = _SCHEMA.validate_python(
            camera_list_response.content["application/json"].schema_
        )
        assert camera_list_schema.type == "array"
        assert camera_list_schema.items is not None
        assert _SCHEMA.validate_python(camera_list_schema.items).ref == (
            "#/components/schemas/CameraResponse"
        )

        request_contracts: tuple[tuple[str, HttpMethod, str], ...] = (
            ("/api/session", "post", "LoginRequest"),
            ("/api/cameras", "post", "CameraCreateRequest"),
            ("/api/cameras/test", "post", "CameraTestRequest"),
            ("/api/cameras/{camera_id}", "patch", "CameraPatchRequest"),
            ("/api/settings", "patch", "SettingsPatchRequest"),
        )
        for path, method, schema_name in request_contracts:
            path_item = _PATH_ITEM.validate_python(paths[path])
            operation = _OPERATION.validate_python(
                {"get": path_item.get, "post": path_item.post, "patch": path_item.patch}[method]
            )
            assert operation.requestBody is not None
            request_body = _REQUEST_BODY.validate_python(operation.requestBody)
            request_schema = _SCHEMA.validate_python(
                request_body.content["application/json"].schema_
            )
            assert request_schema.ref == f"#/components/schemas/{schema_name}"

    finally:
        await transport.__aexit__(None, None, None)
        await database.close()


@pytest.mark.anyio
async def test_default_app_openapi_pins_settings_schema_constraints(tmp_path: Path) -> None:
    dependencies, database, transport = _dependencies(tmp_path)
    try:
        document = _OPENAPI_DOCUMENT.validate_python(create_app(dependencies).openapi())
        assert document.components is not None
        assert document.components.schemas is not None
        settings_patch = _SCHEMA.validate_python(
            document.components.schemas["SettingsPatchRequest"]
        )
        assert settings_patch.type == "object"
        assert settings_patch.additionalProperties is False
        assert settings_patch.properties is not None
        assert set(settings_patch.properties) == {"retention_days", "quota_bytes", "wall_slot_ids"}
        retention_schema = _SCHEMA.validate_python(settings_patch.properties["retention_days"])
        assert retention_schema.anyOf is not None
        retention_value = _SCHEMA.validate_python(retention_schema.anyOf[0])
        assert retention_value.minimum == 1.0
        assert retention_value.type == "integer"
        quota_schema = _SCHEMA.validate_python(settings_patch.properties["quota_bytes"])
        assert quota_schema.anyOf is not None
        quota_value = _SCHEMA.validate_python(quota_schema.anyOf[0])
        assert quota_value.exclusiveMinimum == 0.0
        assert quota_value.type == "integer"
        wall_slots_schema = _SCHEMA.validate_python(settings_patch.properties["wall_slot_ids"])
        assert wall_slots_schema.anyOf is not None
        wall_slots = _SCHEMA.validate_python(wall_slots_schema.anyOf[0])
        assert wall_slots.minItems == 4
        assert wall_slots.maxItems == 4
        assert wall_slots.prefixItems is not None
        assert len(wall_slots.prefixItems) == 4

        settings_response = _SCHEMA.validate_python(document.components.schemas["SettingsResponse"])
        assert settings_response.required == ["retention_days", "quota_bytes", "wall_slot_ids"]
    finally:
        await transport.__aexit__(None, None, None)
        await database.close()
