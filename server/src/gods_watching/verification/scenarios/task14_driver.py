"""Drive the authenticated search API over appearances published by the real worker."""

import json
import os
import secrets
import signal
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from http.client import HTTPConnection
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, cast, final
from uuid import UUID, uuid4

import anyio
from anyio import to_thread
from cryptography.fernet import Fernet

from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.storage import CredentialCipher, Database, StorageRepository
from gods_watching.verification.context import allocate_loopback_port

from .pipeline_worker_runtime import (
    WorkerStackSettings,
    age_out_ended,
    create_cameras,
    kill_worker,
    start_worker,
    stop_worker,
    visible_rows,
    wait_published,
    wait_until,
)
from .task14_errors import Task14ExecutionError
from .task14_models import SearchErrorsEvidence, SearchEvidence, Task14DriverEvidence

if TYPE_CHECKING:
    from anyio.abc import Process

_PUBLISHED_TARGET: Final = 6
_PUBLISH_TIMEOUT_SECONDS: Final = 150.0
_AGE_TIMEOUT_SECONDS: Final = 190.0
_APP_READY_TIMEOUT_SECONDS: Final = 60.0
_HTTP_TIMEOUT_SECONDS: Final = 30.0
_HTTP_OK: Final = 200
_JPEG_MAGIC: Final = b"\xff\xd8"
_TEXT_QUERY: Final = "a person walking"
_CAMERA_LABEL: Final = "task14"
_FOREIGN_ORIGIN: Final = "http://attacker.example"

type JsonObject = dict[str, object]


@dataclass(frozen=True, slots=True)
class _Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> JsonObject:
        decoded: object = json.loads(self.body) if self.body else {}
        if not isinstance(decoded, dict):
            return {}
        return {str(key): value for key, value in cast("dict[object, object]", decoded).items()}


@final
class _HttpClient:
    def __init__(self, port: int, origin: str) -> None:
        self._port: int = port
        self._origin: str = origin
        self.cookie: str | None = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def origin(self) -> str:
        return self._origin

    def request(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
        *,
        origin: str | None = None,
    ) -> _Response:
        headers = {"Origin": origin or self._origin}
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        if self.cookie is not None:
            headers["Cookie"] = self.cookie
        connection = HTTPConnection("127.0.0.1", self._port, timeout=_HTTP_TIMEOUT_SECONDS)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            response_headers = {key.lower(): value for key, value in response.getheaders()}
        finally:
            connection.close()
        set_cookie = response_headers.get("set-cookie")
        if set_cookie is not None:
            self.cookie = set_cookie.split(";", 1)[0]
        return _Response(status=response.status, headers=response_headers, body=raw)

    async def call(
        self,
        method: str,
        path: str,
        payload: JsonObject | None = None,
        *,
        origin: str | None = None,
    ) -> _Response:
        return await to_thread.run_sync(partial(self.request, method, path, payload, origin=origin))


@dataclass(frozen=True, slots=True)
class _Stack:
    worker_settings: WorkerStackSettings
    triton_container: str
    app_log: Path
    password: str


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise Task14ExecutionError(detail=f"required environment variable is missing: {name}")
    return value


def _results(response: _Response) -> list[JsonObject]:
    results = response.json().get("results")
    if not isinstance(results, list):
        return []
    return [
        {str(key): value for key, value in cast("dict[object, object]", item).items()}
        for item in cast("list[object]", results)
        if isinstance(item, dict)
    ]


def _ids(results: list[JsonObject]) -> list[str]:
    return [str(item.get("appearance_id")) for item in results]


def _ranked_desc(results: list[JsonObject]) -> bool:
    scores = [item.get("similarity") for item in results]
    numeric = [float(score) for score in scores if isinstance(score, int | float)]
    return len(numeric) == len(scores) and all(
        earlier >= later for earlier, later in pairwise(numeric)
    )


async def _start_app(stack: _Stack, port: int) -> "Process":
    settings = stack.worker_settings
    environment = {
        **os.environ,
        "GW_DATABASE_URL": settings.database_url,
        "GW_TRITON_GRPC_URL": settings.triton_url,
        "GW_CROPS_ROOT": str(settings.crop_root),
        "GW_CAMERA_CIPHER_KEY": settings.cipher_key,
        "GW_OPERATOR_PASSWORD": stack.password,
        "GW_TASK14_PUBLIC_ORIGIN": f"http://127.0.0.1:{port}",
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "gods_watching.verification.scenarios.task14_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--log-level",
        "warning",
        "--no-proxy-headers",
    ]
    with stack.app_log.open("ab") as log:
        return await anyio.open_process(command, env=environment, stdout=log, stderr=log)


async def _login(stack: _Stack, port: int) -> _HttpClient:
    client = _HttpClient(port, f"http://127.0.0.1:{port}")

    async def ready() -> bool:
        try:
            return (await client.call("GET", "/api/session")).status == _HTTP_OK
        except OSError:
            return False

    if not await wait_until(ready, _APP_READY_TIMEOUT_SECONDS):
        raise Task14ExecutionError(detail="search API did not become ready")
    login = await client.call("POST", "/api/session", {"password": stack.password})
    if login.status != _HTTP_OK or client.cookie is None:
        raise Task14ExecutionError(detail="operator login was not accepted")
    return client


async def _observe_search(
    client: _HttpClient, database: Database, cameras: tuple[str, ...]
) -> SearchEvidence:
    anonymous = _HttpClient(client.port, client.origin)
    unauthenticated = await anonymous.call("POST", "/api/search", {"mode": "browse"})
    cross_origin = await client.call(
        "POST", "/api/search", {"mode": "browse"}, origin=_FOREIGN_ORIGIN
    )

    camera_filter = _results(
        await client.call(
            "POST", "/api/search", {"mode": "browse", "camera_ids": [cameras[0]], "limit": 100}
        )
    )
    rows = await visible_rows(database)
    window_start = sorted(row.first_seen for row in rows)[len(rows) // 2]
    window_end = max(row.last_seen for row in rows)
    time_filter = _results(
        await client.call(
            "POST",
            "/api/search",
            {
                "mode": "browse",
                "from": window_start.isoformat(),
                "to": window_end.isoformat(),
                "limit": 100,
            },
        )
    )
    camera_only = bool(camera_filter) and all(
        str(item.get("camera_id")) == cameras[0] for item in camera_filter
    )
    time_only = bool(time_filter) and all(
        datetime.fromisoformat(str(item.get("last_seen"))) >= window_start
        and datetime.fromisoformat(str(item.get("first_seen"))) <= window_end
        for item in time_filter
    )

    text_request: JsonObject = {"mode": "text", "query": _TEXT_QUERY, "limit": 20}
    text_first = await client.call("POST", "/api/search", text_request)
    text_second = await client.call("POST", "/api/search", text_request)
    text_results = _results(text_first)
    seed = _ids(text_results)[0] if text_results else str(uuid4())
    similar_request: JsonObject = {"mode": "similar", "appearance_id": seed, "limit": 20}
    similar_first = await client.call("POST", "/api/search", similar_request)
    similar_second = await client.call("POST", "/api/search", similar_request)
    similar_results = _results(similar_first)
    detail = await client.call("GET", f"/api/appearances/{seed}")
    crop = await client.call("GET", f"/api/appearances/{seed}/crop")
    return SearchEvidence(
        unauthenticated_status=unauthenticated.status,
        cross_origin_status=cross_origin.status,
        browse_results=len(camera_filter) + len(time_filter),
        browse_camera_filter_only=camera_only,
        browse_time_filter_only=time_only,
        text_status=text_first.status,
        text_results=len(text_results),
        text_ranked_desc=_ranked_desc(text_results),
        text_stable=_ids(text_results) == _ids(_results(text_second)),
        similar_status=similar_first.status,
        similar_results=len(similar_results),
        similar_excludes_seed=seed not in _ids(similar_results),
        similar_ranked_desc=_ranked_desc(similar_results),
        similar_stable=_ids(similar_results) == _ids(_results(similar_second)),
        detail_status=detail.status,
        crop_status=crop.status,
        crop_jpeg=crop.headers.get("content-type") == "image/jpeg"
        and crop.body.startswith(_JPEG_MAGIC),
        crop_no_store="no-store" in crop.headers.get("cache-control", ""),
    )


async def _observe_errors(
    client: _HttpClient, stack: _Stack, database: Database, worker: "Process"
) -> SearchErrorsEvidence:
    blank = await client.call("POST", "/api/search", {"mode": "text", "query": "   "})
    unknown = await client.call(
        "POST", "/api/search", {"mode": "browse", "camera_ids": [str(uuid4())]}
    )
    aged = await age_out_ended(
        database,
        stack.worker_settings.crop_root,
        count=1,
        deadline_seconds=_AGE_TIMEOUT_SECONDS,
    )
    if aged.evicted == 0:
        raise Task14ExecutionError(detail="retention did not expire a seed appearance")
    expired = str(aged.appearance_ids[0])
    expired_seed = await client.call(
        "POST", "/api/search", {"mode": "similar", "appearance_id": expired}
    )
    expired_detail = await client.call("GET", f"/api/appearances/{expired}")
    expired_crop = await client.call("GET", f"/api/appearances/{expired}/crop")
    live = await visible_rows(database)
    live_seed = str(live[0].appearance_id) if live else str(uuid4())

    _ = await stop_worker(worker, signal.SIGTERM)
    stopped = await anyio.run_process(["docker", "stop", stack.triton_container], check=False)
    if stopped.returncode != 0:
        raise Task14ExecutionError(detail="inference container could not be stopped")
    outage_text = await client.call(
        "POST", "/api/search", {"mode": "text", "query": "a person in red clothing"}
    )
    outage_browse = await client.call("POST", "/api/search", {"mode": "browse"})
    outage_similar = await client.call(
        "POST", "/api/search", {"mode": "similar", "appearance_id": live_seed}
    )
    return SearchErrorsEvidence(
        blank_text_status=blank.status,
        unknown_camera_status=unknown.status,
        expired_seed_status=expired_seed.status,
        expired_detail_status=expired_detail.status,
        expired_crop_status=expired_crop.status,
        outage_text_status=outage_text.status,
        outage_browse_status=outage_browse.status,
        outage_similar_status=outage_similar.status,
    )


async def _drive(mode: Literal["search", "search-errors"]) -> None:
    stack = _Stack(
        worker_settings=WorkerStackSettings(
            database_url=_env("GW_TASK14_DATABASE_URL"),
            triton_url=_env("GW_TASK14_TRITON_URL"),
            rtsp_host=_env("GW_TASK14_RTSP_HOST"),
            rtsp_port=int(_env("GW_TASK14_RTSP_PORT")),
            crop_root=Path(_env("GW_TASK14_CROP_ROOT")),
            cipher_key=Fernet.generate_key().decode(),
            worker_log=Path(_env("GW_TASK14_WORKER_LOG")),
        ),
        triton_container=_env("GW_TASK14_TRITON_CONTAINER"),
        app_log=Path(_env("GW_TASK14_APP_LOG")),
        password=secrets.token_urlsafe(24),
    )
    settings = stack.worker_settings
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    worker: Process | None = None
    app: Process | None = None
    try:
        cameras = await create_cameras(
            database,
            CameraService(CameraRepository(storage)),
            settings.rtsp_host,
            label=_CAMERA_LABEL,
        )
        worker = await start_worker(settings)
        _ = await wait_published(database, _PUBLISHED_TARGET, _PUBLISH_TIMEOUT_SECONDS)
        port = allocate_loopback_port()
        app = await _start_app(stack, port)
        client = await _login(stack, port)
        camera_ids = tuple(str(UUID(str(camera))) for camera in cameras)
        if mode == "search":
            evidence = Task14DriverEvidence(
                mode=mode, search=await _observe_search(client, database, camera_ids)
            )
        else:
            evidence = Task14DriverEvidence(
                mode=mode, errors=await _observe_errors(client, stack, database, worker)
            )
        output = Path(_env("GW_TASK14_OUTPUT"))
        _ = output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    finally:
        with anyio.CancelScope(shield=True):
            for process in (worker, app):
                if process is not None and process.returncode is None:
                    _ = await kill_worker(process)
            await database.close()


async def _main() -> None:
    mode_value = _env("GW_TASK14_MODE")
    if mode_value not in ("search", "search-errors"):
        raise Task14ExecutionError(detail=f"unsupported Task 14 mode: {mode_value}")
    mode: Literal["search", "search-errors"] = mode_value
    async with anyio.create_task_group() as task_group:

        async def cancel_on_signal() -> None:
            # Harness termination cancels the drive so owned processes are killed.
            with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
                async for _signum in signals:
                    task_group.cancel_scope.cancel()
                    return

        task_group.start_soon(cancel_on_signal)
        await _drive(mode)
        task_group.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(_main)
