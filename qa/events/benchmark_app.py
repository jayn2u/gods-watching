"""Reduced real-auth HTTP app used only by the disposable export benchmark."""

from __future__ import annotations

import hmac
import inspect
import os
import resource
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_EVENTS_DIR = Path(__file__).resolve().parent
_SOURCE_DIR = Path(os.environ.get("GW_BENCHMARK_SOURCE_DIR", "/work/server/src"))
for _path in (_EVENTS_DIR, _SOURCE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from buffered_export import BufferedEventExporter
from benchmark_export import _writer_insert
from gods_watching.api.app_settings import ApiSettings
from gods_watching.api.event_routes import build_event_router
from gods_watching.api.session_routes import build_session_router
from gods_watching.api.sessionguard import require_session
from gods_watching.auth import AuthService
from gods_watching.events.export import EventExportService, PreparedCsvExport
from gods_watching.events.repository import EventRepository
from gods_watching.storage import CameraEvent, Database

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from gods_watching.contracts.events import EventExportFilters


class TimedExporter:
    """Wrap the actual exporter and record app-side prepare and chunk-pull spans."""

    def __init__(self, delegate: Any, measurements: dict[str, Any]) -> None:
        self.delegate = delegate
        self.measurements = measurements

    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        start = time.monotonic()
        prepared = await self.delegate.prepare(filters)
        end = time.monotonic()
        self.measurements["prepare_spans"].append({"start": start, "end": end})

        async def measured_chunks() -> AsyncIterator[bytes]:
            iterator = prepared.chunks.__aiter__()
            index = 0
            while True:
                pull_start = time.monotonic()
                try:
                    chunk = await anext(iterator)
                except StopAsyncIteration:
                    return
                pull_end = time.monotonic()
                self.measurements["chunk_pull_spans"].append(
                    {
                        "index": index,
                        "start": pull_start,
                        "end": pull_end,
                        "bytes": len(chunk),
                    }
                )
                index += 1
                yield chunk

        return PreparedCsvExport(
            measured_chunks(),
            prepared.aclose,
            deadline_at=prepared.deadline_at,
        )


def _peak_rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _build_app() -> FastAPI:
    database_url = os.environ["GW_DATABASE_URL"]
    ensure_url = os.environ["GW_BENCHMARK_DATABASE_NAME"]
    password = os.environ["GW_BENCHMARK_PASSWORD"]
    control_token = os.environ["GW_BENCHMARK_CONTROL_TOKEN"]
    mode = os.environ["GW_BENCHMARK_MODE"]
    database_url_value = database_url
    from benchmark_export import ensure_disposable_database

    ensure_disposable_database(database_url_value, expected_database=ensure_url)
    engine = create_async_engine(database_url, pool_size=3, max_overflow=0, pool_pre_ping=True)
    database = Database(engine, async_sessionmaker(engine, expire_on_commit=False))
    auth = AuthService(transactions=database)
    config = ApiSettings(public_origin="http://app:8000", secure_cookie=False)
    measurements: dict[str, Any] = {
        "mode": mode,
        "prepare_spans": [],
        "chunk_pull_spans": [],
        "writer": None,
        "baseline": None,
        "source_paths": {
            "AuthService": inspect.getfile(AuthService),
            "EventExportService": inspect.getfile(EventExportService),
            "EventRepository": inspect.getfile(EventRepository),
            "CameraEvent": inspect.getfile(CameraEvent),
        },
    }
    if any(
        not Path(path).resolve().is_relative_to(_SOURCE_DIR.resolve())
        for path in measurements["source_paths"].values()
    ):
        raise RuntimeError("benchmark imported project code outside the mounted worktree source")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _ = await auth.initialize_password(password)
        app.state.measurements = measurements
        yield
        await database.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_session_router(auth, config))
    def _require_session_factory(*, user_action: bool) -> Any:
        return require_session(auth, config, user_action=user_action)

    exporter = (
        BufferedEventExporter(database)
        if mode == "buffered"
        else EventExportService(database)
    )
    app.include_router(
        build_event_router(
            database=database,
            require_session=_require_session_factory,
            exporter=TimedExporter(exporter, measurements),
        )
    )

    async def _control_guard(
        benchmark_control: str | None = Header(default=None, alias="X-Benchmark-Control"),
    ) -> None:
        if benchmark_control is None or not hmac.compare_digest(benchmark_control, control_token):
            raise HTTPException(status_code=404, detail="not found")

    @app.get("/__benchmark/ready", dependencies=[Depends(_control_guard)])
    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/__benchmark/baseline", dependencies=[Depends(_control_guard)])
    async def baseline(request: Request) -> dict[str, Any]:
        state = request.app.state.measurements
        if state["baseline"] is None:
            state["baseline"] = {
                "captured_monotonic": time.monotonic(),
                "peak_rss_kib": _peak_rss_kib(),
                "note": "captured after real operator login; ru_maxrss is a process high-water mark",
            }
        return state["baseline"]

    @app.post("/__benchmark/writer", dependencies=[Depends(_control_guard)])
    async def writer(request: Request) -> dict[str, Any]:
        state = request.app.state.measurements
        if state["writer"] is not None:
            raise HTTPException(status_code=409, detail="writer already used for this sample")
        try:
            result = await _writer_insert(database)
            state["writer"] = result
            return result
        except Exception as error:
            state["writer"] = {"count": 0, "error": type(error).__name__}
            raise HTTPException(status_code=500, detail="benchmark writer failed") from error

    @app.get("/__benchmark/metrics", dependencies=[Depends(_control_guard)])
    async def metrics(request: Request) -> dict[str, Any]:
        state = request.app.state.measurements
        return {
            "mode": state["mode"],
            "baseline": state["baseline"],
            "application_peak_rss_kib": _peak_rss_kib(),
            "metrics_captured_at_utc": datetime.now(UTC).isoformat(),
            "prepare_spans": state["prepare_spans"],
            "chunk_pull_spans": state["chunk_pull_spans"],
            "writer": state["writer"],
            "source_paths": state["source_paths"],
        }

    return app


def main() -> None:
    uvicorn.run(
        _build_app(),
        host="0.0.0.0",
        port=8000,
        workers=1,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
