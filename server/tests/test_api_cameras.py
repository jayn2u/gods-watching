"""HTTP camera/settings and runtime lifecycle contract tests."""

import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import final, override
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request
from pydantic import AnyUrl
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.types import Message, Scope

from gods_watching.api.camera_routes import (
    AuthenticatedRequest,
    CameraRepositoryProvider,
    CameraServiceProvider,
    SessionDependency,
    SessionDependencyFactory,
    TransactionProvider,
    build_camera_router,
)
from gods_watching.api.camera_runtime import (
    CameraRuntime,
    CameraRuntimePort,
    StaleRuntimeEffectError,
    ThresholdHandoff,
)
from gods_watching.api.settings_routes import build_settings_router
from gods_watching.cameras import (
    CameraActivationReason,
    CameraActivationRequest,
    CameraCancellationReason,
    CameraCancellationRequest,
    CameraGenerationId,
    CameraLifecyclePlan,
    CameraMutation,
    ParsedRtspSource,
    ProbeFailureCode,
    RtspProbeError,
    parse_rtsp_source,
)
from gods_watching.contracts.cameras import (
    CameraCreateRequest,
    CameraPatchRequest,
    CameraResponse,
    CameraTestResponse,
)
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.settings import SettingsPatchRequest, SettingsResponse
from gods_watching.media import SourceGenerationId

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class _FakeTransaction(TransactionProvider):
    """Caller-owned transaction surface used to isolate route composition."""

    session: AsyncSession = field(default_factory=AsyncSession)

    @asynccontextmanager
    @override
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        yield self.session


@dataclass
class _FakeRuntime(CameraRuntimePort):
    """Record ordered post-commit effects without pretending to process frames."""

    calls: list[tuple[str, str]] = field(default_factory=list)
    committed: list[tuple[CameraId, int, bool]] = field(default_factory=list)

    @override
    async def mark_committed(
        self, camera_id: CameraId, version: int, *, deleted: bool = False
    ) -> None:
        self.calls.append(("mark", str(version)))
        self.committed.append((camera_id, version, deleted))

    @override
    async def activate_source(
        self,
        camera_id: CameraId,
        version: int,
        source: ParsedRtspSource,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        del camera_id, publish_events
        self.calls.append(("media_activate", f"{version}:{source.host}"))
        return None

    @override
    async def apply_threshold(
        self,
        camera_id: CameraId,
        version: int,
        threshold: float,
    ) -> ThresholdHandoff:
        del camera_id
        self.calls.append(("threshold", f"{version}:{threshold}"))
        return ThresholdHandoff.DEFERRED

    @override
    async def activate(self, request: CameraActivationRequest) -> None:
        self.calls.append(("detector_activate", str(request.version)))

    @override
    async def cancel(self, request: CameraCancellationRequest) -> None:
        self.calls.append(("detector_cancel", str(request.version)))

    @override
    async def close_camera(self, camera_id: CameraId) -> None:
        self.calls.append(("close", str(camera_id)))


@dataclass
class _RecordingMedia:
    activations: list[tuple[CameraId, str]] = field(default_factory=list)
    deactivations: list[CameraId] = field(default_factory=list)

    async def activate(
        self,
        camera_id: CameraId,
        source_url: str,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId:
        del publish_events
        self.activations.append((camera_id, source_url))
        return SourceGenerationId(uuid4())

    async def deactivate(
        self,
        camera_id: CameraId,
        *,
        publish_events: bool = True,
    ) -> SourceGenerationId | None:
        del publish_events
        self.deactivations.append(camera_id)
        return None


@dataclass
class _RecordingDetector:
    activations: list[CameraActivationRequest] = field(default_factory=list)

    async def activate(self, request: CameraActivationRequest) -> None:
        self.activations.append(request)

    async def cancel(self, request: CameraCancellationRequest) -> None:
        del request

    async def apply_threshold(self, camera_id: CameraId, threshold: float) -> None:
        del camera_id, threshold


@dataclass
class _FakeCameraService(CameraServiceProvider):
    """Small typed service double for route-only behavior tests."""

    camera_id: CameraId = field(default_factory=lambda: CameraId(uuid4()))
    source: str = "rtsp://operator:secret@fixture:8554/live"
    version: int = 1
    probe_error: RtspProbeError | None = None
    mutation: CameraMutation | None = None
    tests: list[str] = field(default_factory=list)

    @property
    @override
    def repository(self) -> CameraRepositoryProvider:
        raise AttributeError

    @override
    async def list(self, session: AsyncSession) -> tuple[CameraResponse, ...]:
        del session
        return (self._response(),)

    @override
    async def create(
        self,
        session: AsyncSession,
        request: CameraCreateRequest,
    ) -> CameraMutation:
        del session
        self.source = str(request.source_url)
        self.version = 1
        self.mutation = CameraMutation(
            self._response(),
            CameraLifecyclePlan(
                activation=_activation(self.camera_id, self.version, request.source_url),
                cancellation=None,
            ),
        )
        return self.mutation

    @override
    async def update(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        request: CameraPatchRequest,
        *,
        expected_version: int,
    ) -> CameraMutation:
        del session, camera_id
        if expected_version != self.version:
            raise RuntimeError
        self.version += 1
        source_url = request.source_url
        if source_url is not None:
            self.source = str(source_url)
        self.mutation = CameraMutation(
            self._response(),
            CameraLifecyclePlan(
                activation=(
                    _activation(self.camera_id, self.version, source_url)
                    if source_url is not None
                    else None
                ),
                cancellation=None,
            ),
        )
        return self.mutation

    @override
    async def delete(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        *,
        expected_version: int,
    ) -> CameraMutation:
        del session, camera_id
        if expected_version != self.version:
            raise RuntimeError
        self.version += 1
        response = self._response()
        self.mutation = CameraMutation(response, CameraLifecyclePlan(None, None))
        return self.mutation

    @override
    async def test_source(self, source_url: str) -> CameraTestResponse:
        self.tests.append(source_url)
        if self.probe_error is not None:
            raise self.probe_error
        return CameraTestResponse(
            source_host="fixture",
            source_port=8554,
            codec="h264",
            width=1920,
            height=1080,
        )

    def _response(self) -> CameraResponse:
        return CameraResponse(
            camera_id=self.camera_id,
            name="Lobby",
            source_host="fixture",
            source_port=8554,
            detection_enabled=True,
            detection_threshold=0.5,
            version=self.version,
            deleted_at=None,
        )


@dataclass
@final
class _DetectionReenableCameraService(_FakeCameraService):
    @override
    async def update(
        self,
        session: AsyncSession,
        camera_id: CameraId,
        request: CameraPatchRequest,
        *,
        expected_version: int,
    ) -> CameraMutation:
        del session, camera_id, request
        if expected_version != self.version:
            raise RuntimeError
        self.version += 1
        self.mutation = CameraMutation(
            self._response(),
            CameraLifecyclePlan(
                activation=replace(
                    _activation(self.camera_id, self.version, self.source),
                    reason=CameraActivationReason.DETECTION_ENABLED,
                ),
                cancellation=None,
            ),
        )
        return self.mutation


@dataclass
class _FakeSettingsService:
    """Typed settings double that records durable route calls."""

    value: SettingsResponse = field(
        default_factory=lambda: SettingsResponse(
            retention_days=7,
            quota_bytes=100_000_000_000,
            wall_slot_ids=(None, None, None, None),
        )
    )

    async def get(self, session: AsyncSession) -> SettingsResponse:
        del session
        return self.value

    async def update(
        self,
        session: AsyncSession,
        patch: SettingsPatchRequest,
    ) -> SettingsResponse:
        del session
        self.value = SettingsResponse(
            retention_days=patch.retention_days or self.value.retention_days,
            quota_bytes=patch.quota_bytes or self.value.quota_bytes,
            wall_slot_ids=patch.wall_slot_ids or self.value.wall_slot_ids,
        )
        return self.value


def _activation(
    camera_id: CameraId,
    version: int,
    source_url: str | AnyUrl,
) -> CameraActivationRequest:
    source = parse_rtsp_source(source_url)
    return CameraActivationRequest(
        camera_id=camera_id,
        version=version,
        session_id=CameraSessionId(uuid4()),
        generation_id=CameraGenerationId(uuid4()),
        source=source,
        detection_threshold=0.5,
        reason=CameraActivationReason.CREATED,
    )


def _guard_factory() -> SessionDependencyFactory:
    """Return a guard factory that denies missing auth and cross-origin writes."""

    def factory(*, user_action: bool) -> SessionDependency:
        async def guard(request: Request) -> AuthenticatedRequest:
            if request.headers.get("x-test-session") != "valid":
                raise HTTPException(status_code=401, detail="authentication required")
            if user_action and request.headers.get("origin") != "https://gw.test":
                raise HTTPException(status_code=403, detail="mutation origin is not allowed")
            return AuthenticatedRequest(session_id=str(uuid4()))

        return guard

    return factory


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: dict[str, str] | None = None,
    body: Mapping[str, JsonValue] | None = None,
) -> tuple[int, bytes]:
    """Drive the ASGI surface without adding an HTTP client dependency."""
    messages: list[Message] = []
    payload = json.dumps(body).encode() if body is not None else b""
    request_headers = [
        (key.lower().encode(), value.encode()) for key, value in (headers or {}).items()
    ]
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
    await app(
        scope,
        receive,
        send,
    )
    status_code = _response_status(messages)
    chunks = [
        body
        for message in messages
        if message.get("type") == "http.response.body"
        and isinstance(body := message.get("body"), bytes)
    ]
    return status_code, b"".join(chunks)


def _response_status(messages: list[Message]) -> int:
    for message in messages:
        if message.get("type") != "http.response.start":
            continue
        status = message.get("status")
        if isinstance(status, int):
            return status
    raise AssertionError


@pytest.mark.anyio
async def test_camera_routes_require_guard_and_same_origin_for_mutations() -> None:
    # Given: route composition with an auth guard and a fake durable service
    database = _FakeTransaction()
    cameras = _FakeCameraService()
    runtime = _FakeRuntime()
    app = FastAPI()
    app.include_router(
        build_camera_router(
            database=database,
            cameras=cameras,
            runtime=runtime,
            require_session=_guard_factory(),
        )
    )

    # When: requests omit auth, then send a cross-origin mutation
    denied_status, _ = await _request(app, "GET", "/api/cameras")
    cross_origin_status, _ = await _request(
        app,
        "POST",
        "/api/cameras",
        headers={"x-test-session": "valid", "origin": "https://evil.test"},
        body={"name": "Lobby", "source_url": "rtsp://fixture:8554/live"},
    )

    # Then: both security boundaries fail before service effects
    assert denied_status == 401
    assert cross_origin_status == 403
    assert runtime.calls == []


@pytest.mark.anyio
async def test_camera_create_commits_before_media_and_detector_effects() -> None:
    # Given: an authenticated same-origin camera create with a detector plan
    database = _FakeTransaction()
    cameras = _FakeCameraService()
    runtime = _FakeRuntime()
    app = FastAPI()
    app.include_router(
        build_camera_router(
            database=database,
            cameras=cameras,
            runtime=runtime,
            require_session=_guard_factory(),
        )
    )

    # When: the real HTTP route receives one valid camera payload
    status, _ = await _request(
        app,
        "POST",
        "/api/cameras",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"name": "Lobby", "source_url": "rtsp://fixture:8554/live"},
    )

    # Then: the committed fence is observed before source and detector activation
    assert status == 201
    assert [name for name, _value in runtime.calls] == [
        "mark",
        "media_activate",
        "detector_activate",
    ]


@pytest.mark.anyio
async def test_camera_test_sanitizes_probe_failures_and_never_echoes_source() -> None:
    # Given: a bounded probe failure for a credential-bearing source
    cameras = _FakeCameraService(
        probe_error=RtspProbeError(
            code=ProbeFailureCode.UNREACHABLE,
            detail="RTSP source could not be reached",
        )
    )
    app = FastAPI()
    app.include_router(
        build_camera_router(
            database=_FakeTransaction(),
            cameras=cameras,
            runtime=_FakeRuntime(),
            require_session=_guard_factory(),
        )
    )

    # When: the authenticated operator runs the source test
    status, payload = await _request(
        app,
        "POST",
        "/api/cameras/test",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"source_url": "rtsp://operator:secret@offline:8554/live"},
    )

    # Then: the response contains only the stable sanitized error
    assert status == 422
    assert b"secret" not in payload
    assert b"rtsp://" not in payload


@pytest.mark.anyio
async def test_settings_routes_persist_values_and_protect_patch() -> None:
    # Given: an authenticated settings router backed by one durable service
    settings = _FakeSettingsService()
    app = FastAPI()
    app.include_router(
        build_settings_router(
            database=_FakeTransaction(),
            settings=settings,
            require_session=_guard_factory(),
        )
    )

    # When: the operator reads, then patches settings
    read_status, _ = await _request(
        app,
        "GET",
        "/api/settings",
        headers={"x-test-session": "valid"},
    )
    patched_status, patched_payload = await _request(
        app,
        "PATCH",
        "/api/settings",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"retention_days": 21, "quota_bytes": 123456789},
    )

    # Then: values are returned from the durable settings contract
    assert read_status == 200
    assert patched_status == 200
    rendered = SettingsResponse.model_validate_json(patched_payload)
    assert rendered.retention_days == 21
    assert rendered.quota_bytes == 123456789


@pytest.mark.anyio
async def test_runtime_ignores_stale_activation_after_newer_commit() -> None:
    # Given: a runtime with a committed newer camera version
    runtime = CameraRuntime(detector=None, media=None)
    camera_id = CameraId(uuid4())
    await runtime.mark_committed(camera_id, 4)
    plan = _activation(camera_id, 3, "rtsp://fixture:8554/old")

    # When / Then: an older callback cannot resurrect detector work
    with pytest.raises(StaleRuntimeEffectError):
        await runtime.activate(plan)


@pytest.mark.anyio
async def test_runtime_threshold_handoff_is_deferred_without_fake_detector() -> None:
    # Given: detection is unavailable until the Task 10b worker is supplied
    runtime = CameraRuntime(detector=None, media=None)
    result = await runtime.apply_threshold(CameraId(uuid4()), 2, 0.8)

    # Then: the typed handoff is retained as deferred work, never reported as processed
    assert result is ThresholdHandoff.DEFERRED


@pytest.mark.anyio
async def test_runtime_detection_disable_keeps_media_path_live() -> None:
    camera_id = CameraId(uuid4())
    activation = _activation(camera_id, 1, "rtsp://fixture:8554/live")
    media = _RecordingMedia()
    runtime = CameraRuntime(detector=None, media=media)
    await runtime.mark_committed(camera_id, activation.version)
    _ = await runtime.activate_bound_source(activation)

    cancellation = CameraCancellationRequest(
        camera_id=camera_id,
        version=activation.version,
        session_id=activation.session_id,
        generation_id=activation.generation_id,
        reason=CameraCancellationReason.DETECTION_DISABLED,
    )
    await runtime.cancel(cancellation)

    assert media.deactivations == []
    assert await runtime.active_binding(camera_id) is None


@pytest.mark.anyio
async def test_detection_reenable_binds_source_before_detector_activation() -> None:
    # Given: a fresh runtime and a durable detection-only re-enable activation
    cameras = _DetectionReenableCameraService()
    media = _RecordingMedia()
    detector = _RecordingDetector()
    runtime = CameraRuntime(detector=detector, media=media)
    app = FastAPI()
    app.include_router(
        build_camera_router(
            database=_FakeTransaction(),
            cameras=cameras,
            runtime=runtime,
            require_session=_guard_factory(),
        )
    )

    # When: the HTTP update effect applies the detection-only activation
    status, _ = await _request(
        app,
        "PATCH",
        f"/api/cameras/{cameras.camera_id}",
        headers={
            "x-test-session": "valid",
            "origin": "https://gw.test",
            "x-camera-version": "1",
        },
        body={"detection_enabled": True},
    )

    # Then: the source generation is bound before detector work can run
    assert status == 200
    binding = await runtime.active_binding(cameras.camera_id)
    assert binding is not None
    assert binding.camera_version == 2
    assert media.activations == [(cameras.camera_id, cameras.source)]
    assert detector.activations[0].reason is CameraActivationReason.DETECTION_ENABLED
