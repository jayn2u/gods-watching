from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final
from uuid import UUID, uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import TypeAdapter

from gods_watching.api.camera_routes import AuthenticatedRequest, SessionDependency
from gods_watching.api.training_routes import build_training_router
from gods_watching.contracts.training import (
    TrainingConfig,
    TrainingDatasetSnapshot,
    TrainingDatasetStatus,
    TrainingJobPage,
    TrainingJobResponse,
    TrainingLogPage,
    TrainingMetricPage,
    TrainingPreflightResponse,
    TrainingSplitCounts,
)
from gods_watching.training.service import (
    TrainingCursorError,
    TrainingMemoryRefusalMetadata,
    TrainingMemoryRefusedError,
    TrainingSupervisorUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from starlette.types import Message, Scope

_STATUS: Final[TypeAdapter[int]] = TypeAdapter(int)
_JSON_OBJECT: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_DETAIL_OBJECT: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_DATASET = TrainingDatasetSnapshot(
    dataset_id="cuhk-pedes",
    fingerprint="a" * 64,
    protocol="cuhk-pedes-original-splits-v1",
    split_counts={
        "train": TrainingSplitCounts(images=2, captions=4, identities=2),
        "val": TrainingSplitCounts(images=1, captions=2, identities=1),
        "test": TrainingSplitCounts(images=1, captions=2, identities=1),
    },
    image_count=4,
    caption_count=8,
    identity_count=4,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class _Guard:
    built_for_user_actions: list[bool] = field(default_factory=list)

    def __call__(self, *, user_action: bool) -> SessionDependency:
        self.built_for_user_actions.append(user_action)

        async def dependency(request: Request, response: Response) -> AuthenticatedRequest:
            del response
            if request.headers.get("x-test-session") != "valid":
                raise HTTPException(401, detail={"code": "authentication_required"})
            if user_action and request.headers.get("origin") != "https://gw.test":
                raise HTTPException(403, detail={"code": "origin_not_allowed"})
            return AuthenticatedRequest(session_id="test-session")

        return dependency


@dataclass
class _TrainingService:
    response: TrainingJobResponse
    submit_error: Exception | None = None
    preflight_error: Exception | None = None
    action_error: Exception | None = None
    submit_requests: list[object] = field(default_factory=list)

    async def datasets(self) -> TrainingDatasetStatus:
        return TrainingDatasetStatus(registered=True, valid=True, reason=None, snapshot=_DATASET)

    async def config(self) -> TrainingConfig:
        return TrainingConfig()

    async def preflight(self, config: TrainingConfig) -> TrainingPreflightResponse:
        del config
        if self.preflight_error is not None:
            raise self.preflight_error
        return TrainingPreflightResponse(
            admitted=True,
            training_peak_bytes=3,
            reserve_bytes=2,
            required_bytes=5,
            free_bytes=10,
            profile_identity="b" * 64,
            observed_at=datetime(2026, 10, 4, tzinfo=UTC),
            reason="admitted",
        )

    async def submit(self, request: object) -> TrainingJobResponse:
        self.submit_requests.append(request)
        if self.submit_error is not None:
            raise self.submit_error
        return self.response

    async def list_jobs(self, *, cursor: str | None, limit: int) -> TrainingJobPage:
        del limit
        if cursor == "bad":
            raise TrainingCursorError
        return TrainingJobPage(items=(self.response,), next_cursor=None)

    async def get_job(self, job_id: UUID) -> TrainingJobResponse:
        assert job_id == self.response.id
        return self.response

    async def cancel(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        del request_id
        assert job_id == self.response.id
        return self.response

    async def resume(self, job_id: UUID, request_id: UUID) -> TrainingJobResponse:
        del request_id
        assert job_id == self.response.id
        if self.action_error is not None:
            raise self.action_error
        return self.response

    async def metrics(
        self,
        job_id: UUID,
        *,
        cursor: str | None,
        limit: int,
    ) -> TrainingMetricPage:
        del job_id, cursor, limit
        return TrainingMetricPage(items=(), next_cursor=None)

    async def logs(
        self,
        job_id: UUID,
        *,
        cursor: str | None,
        limit: int,
    ) -> TrainingLogPage:
        del job_id, cursor, limit
        return TrainingLogPage(items=(), next_cursor=None)


def _job_response(request_id: UUID) -> TrainingJobResponse:
    return TrainingJobResponse(
        id=uuid4(),
        request_id=request_id,
        phase="starting",
        config=TrainingConfig(),
        dataset=_DATASET,
        current_epoch=0,
        current_step=0,
        owner_generation=1,
        cancel_requested=False,
        attempts=0,
        best_metric=None,
        candidate_model_id=None,
        candidate_revision=None,
        error=None,
        created_at=datetime(2026, 10, 4, tzinfo=UTC),
        updated_at=datetime(2026, 10, 4, tzinfo=UTC),
        finished_at=None,
    )


async def _request(
    app: FastAPI,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str] | None = None,
    body: Mapping[str, object] | None = None,
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
        "path": path.partition("?")[0],
        "raw_path": path.partition("?")[0].encode(),
        "query_string": path.partition("?")[2].encode(),
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


def _app(service: _TrainingService, guard: _Guard) -> FastAPI:
    app = FastAPI()
    app.include_router(build_training_router(service=service, require_session=guard))
    return app


@pytest.mark.anyio
async def test_training_routes_require_the_existing_operator_session() -> None:
    request_id = uuid4()
    service = _TrainingService(response=_job_response(request_id))
    app = _app(service, _Guard())

    status_code, _body = await _request(app, "GET", "/api/training/jobs")

    assert status_code == 401
    assert service.submit_requests == []


@pytest.mark.anyio
async def test_invalid_config_is_rejected_before_supervisor_request() -> None:
    request_id = uuid4()
    service = _TrainingService(response=_job_response(request_id))
    app = _app(service, _Guard())

    status_code, body = await _request(
        app,
        "POST",
        "/api/training/preflight",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"config": {"micro_batch_size": 1}},
    )

    assert status_code == 422
    assert b"micro_batch_size" in body


@pytest.mark.anyio
async def test_memory_refusal_preserves_required_free_and_reserve_bytes() -> None:
    request_id = uuid4()
    service = _TrainingService(
        response=_job_response(request_id),
        preflight_error=TrainingMemoryRefusedError(
            required_bytes=5 * 1024**3,
            free_bytes=4 * 1024**3,
            reserve_bytes=2 * 1024**3,
            metadata=TrainingMemoryRefusalMetadata(
                observed_at=datetime(2026, 10, 4, 12, 30, tzinfo=UTC),
                profile_identity="b" * 64,
                reason="insufficient_free_memory",
            ),
        ),
    )
    app = _app(service, _Guard())

    status_code, body = await _request(
        app,
        "POST",
        "/api/training/preflight",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"config": {}},
    )

    assert status_code == 409
    detail = _DETAIL_OBJECT.validate_python(_JSON_OBJECT.validate_json(body)["detail"])
    assert detail["required_bytes"] == 5 * 1024**3
    assert detail["free_bytes"] == 4 * 1024**3
    assert detail["reserve_bytes"] == 2 * 1024**3
    observed_at_value: object = detail["observed_at"]
    assert isinstance(observed_at_value, str)
    assert datetime.fromisoformat(observed_at_value) == datetime(2026, 10, 4, 12, 30, tzinfo=UTC)
    assert detail["profile_identity"] == "b" * 64
    assert detail["reason"] == "insufficient_free_memory"


@pytest.mark.anyio
async def test_unavailable_supervisor_returns_503_without_job_creation() -> None:
    request_id = uuid4()
    service = _TrainingService(
        response=_job_response(request_id),
        submit_error=TrainingSupervisorUnavailableError(),
    )
    app = _app(service, _Guard())

    status_code, body = await _request(
        app,
        "POST",
        "/api/training/jobs",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"request_id": str(request_id), "dataset_id": "cuhk-pedes", "config": {}},
    )

    assert status_code == 503
    assert b"training_supervisor_unavailable" in body


@pytest.mark.anyio
async def test_repeated_submit_returns_the_same_durable_job() -> None:
    request_id = uuid4()
    job = _job_response(request_id)
    service = _TrainingService(response=job)
    app = _app(service, _Guard())
    headers = {"x-test-session": "valid", "origin": "https://gw.test"}
    payload: dict[str, object] = {
        "request_id": str(request_id),
        "dataset_id": "cuhk-pedes",
        "config": {},
    }

    first_status, first = await _request(
        app, "POST", "/api/training/jobs", headers=headers, body=payload
    )
    second_status, second = await _request(
        app, "POST", "/api/training/jobs", headers=headers, body=payload
    )

    assert first_status == second_status == 202
    assert json.loads(first)["id"] == json.loads(second)["id"] == str(job.id)
    assert len(service.submit_requests) == 2


@pytest.mark.anyio
async def test_resume_memory_refusal_preserves_admission_context() -> None:
    request_id = uuid4()
    job = _job_response(request_id)
    service = _TrainingService(
        response=job,
        action_error=TrainingMemoryRefusedError(
            required_bytes=6 * 1024**3,
            free_bytes=5 * 1024**3,
            reserve_bytes=2 * 1024**3,
        ),
    )
    app = _app(service, _Guard())

    status_code, body = await _request(
        app,
        "POST",
        f"/api/training/jobs/{job.id}/resume",
        headers={"x-test-session": "valid", "origin": "https://gw.test"},
        body={"request_id": str(uuid4())},
    )

    assert status_code == 409
    detail = _JSON_OBJECT.validate_json(body)["detail"]
    assert isinstance(detail, dict)
    assert detail["required_bytes"] == 6 * 1024**3
    assert detail["free_bytes"] == 5 * 1024**3
    assert detail["reserve_bytes"] == 2 * 1024**3


@pytest.mark.anyio
async def test_malformed_history_cursor_is_422() -> None:
    request_id = uuid4()
    service = _TrainingService(response=_job_response(request_id))
    app = _app(service, _Guard())

    status_code, _body = await _request(
        app,
        "GET",
        "/api/training/jobs?cursor=bad",
        headers={"x-test-session": "valid"},
    )

    assert status_code == 422
