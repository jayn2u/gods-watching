"""Reduced real-auth HTTP app used only by the disposable export benchmark."""

from __future__ import annotations

import hmac
import inspect
import os
import resource
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast, override

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

try:
    from ._benchmark_app_bootstrap import SOURCE_DIR as _SOURCE_DIR
except ImportError:
    from _benchmark_app_bootstrap import SOURCE_DIR as _SOURCE_DIR

if TYPE_CHECKING:
    from qa.events.benchmark_export import ensure_disposable_database, write_benchmark_batch
    from qa.events.buffered_export import BufferedEventExporter
else:
    from benchmark_export import ensure_disposable_database, write_benchmark_batch
    from buffered_export import BufferedEventExporter

from gods_watching.api.app_settings import ApiSettings
from gods_watching.api.camera_routes import AuthenticatedRequest as RouteAuthenticatedRequest
from gods_watching.api.camera_routes import SessionDependency
from gods_watching.api.event_routes import build_event_router
from gods_watching.api.session_routes import build_session_router
from gods_watching.api.sessionguard import require_session
from gods_watching.auth import AuthService
from gods_watching.events.export import EventExporter, EventExportService, PreparedCsvExport
from gods_watching.events.repository import EventRepository
from gods_watching.storage import CameraEvent, Database

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from gods_watching.contracts.events import EventExportFilters


@dataclass(slots=True)
class Measurements:
    """Track one reduced app process's benchmark-only observations."""

    mode: str
    prepare_spans: list[dict[str, float]] = field(default_factory=list)
    chunk_pull_spans: list[dict[str, int | float]] = field(default_factory=list)
    writer: dict[str, object] | None = None
    baseline: dict[str, object] | None = None
    source_paths: dict[str, str] = field(default_factory=dict)


class TimedExporter(EventExporter):
    """Wrap the actual exporter and record app-side prepare and chunk-pull spans."""

    def __init__(self, delegate: EventExporter, measurements: Measurements) -> None:
        """Wrap a real exporter with an app-local measurements collector."""
        self.delegate: EventExporter = delegate
        self.measurements: Measurements = measurements

    @override
    async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
        """Measure preparation and every emitted chunk, then return safe cleanup."""
        start = time.monotonic()
        prepared = await self.delegate.prepare(filters)
        end = time.monotonic()
        self.measurements.prepare_spans.append({"start": start, "end": end})

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
                self.measurements.chunk_pull_spans.append(
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


def _request_measurements(request: Request) -> Measurements:
    """Validate the benchmark state object stored by this app's lifespan."""
    app = cast("FastAPI", request.app)
    state: object = cast("object", app.state.measurements)
    if not isinstance(state, Measurements):
        exception_message = "benchmark measurements state is unavailable"
        raise TypeError(exception_message)
    return state


def _register_writer_route(
    app: FastAPI,
    database: Database,
    control_guard: Callable[..., Awaitable[None]],
) -> None:
    """Register the one-use benchmark writer endpoint behind the control guard."""

    async def writer(request: Request) -> dict[str, object]:
        state = _request_measurements(request)
        if state.writer is not None:
            raise HTTPException(status_code=409, detail="writer already used for this sample")
        try:
            result = await write_benchmark_batch(database)
            state.writer = result
        except Exception as error:
            state.writer = {"count": 0, "error": type(error).__name__}
            raise HTTPException(status_code=500, detail="benchmark writer failed") from error
        else:
            return result

    app.add_api_route(
        "/__benchmark/writer",
        writer,
        methods=["POST"],
        dependencies=[Depends(control_guard)],
    )


def _register_control_routes(app: FastAPI, database: Database, control_token: str) -> None:
    """Register the benchmark-only endpoints behind the shared control guard."""

    async def _control_guard(
        benchmark_control: Annotated[
            str | None,
            Header(alias="X-Benchmark-Control"),
        ] = None,
    ) -> None:
        if benchmark_control is None or not hmac.compare_digest(benchmark_control, control_token):
            raise HTTPException(status_code=404, detail="not found")

    async def ready() -> dict[str, str]:
        return {"status": "ready"}

    app.add_api_route(
        "/__benchmark/ready",
        ready,
        methods=["GET"],
        dependencies=[Depends(_control_guard)],
    )

    async def baseline(request: Request) -> dict[str, object]:
        state = _request_measurements(request)
        if state.baseline is None:
            state.baseline = {
                "captured_monotonic": time.monotonic(),
                "peak_rss_kib": _peak_rss_kib(),
                "note": (
                    "captured after real operator login; ru_maxrss is a process "
                    "high-water mark"
                ),
            }
        return state.baseline

    app.add_api_route(
        "/__benchmark/baseline",
        baseline,
        methods=["POST"],
        dependencies=[Depends(_control_guard)],
    )

    _register_writer_route(app, database, _control_guard)

    async def metrics(request: Request) -> dict[str, object]:
        state = _request_measurements(request)
        return {
            "mode": state.mode,
            "baseline": state.baseline,
            "application_peak_rss_kib": _peak_rss_kib(),
            "metrics_captured_at_utc": datetime.now(UTC).isoformat(),
            "prepare_spans": state.prepare_spans,
            "chunk_pull_spans": state.chunk_pull_spans,
            "writer": state.writer,
            "source_paths": state.source_paths,
        }

    app.add_api_route(
        "/__benchmark/metrics",
        metrics,
        methods=["GET"],
        dependencies=[Depends(_control_guard)],
    )


def build_app() -> FastAPI:
    """Build the reduced authenticated HTTP app used by benchmark samples."""
    database_url = os.environ["GW_DATABASE_URL"]
    ensure_url = os.environ["GW_BENCHMARK_DATABASE_NAME"]
    password = os.environ["GW_BENCHMARK_PASSWORD"]
    control_token = os.environ["GW_BENCHMARK_CONTROL_TOKEN"]
    mode = os.environ["GW_BENCHMARK_MODE"]
    ensure_disposable_database(database_url, expected_database=ensure_url)
    engine = create_async_engine(database_url, pool_size=3, max_overflow=0, pool_pre_ping=True)
    database = Database(engine, async_sessionmaker(engine, expire_on_commit=False))
    auth = AuthService(transactions=database)
    config = ApiSettings(public_origin="http://app:8000", secure_cookie=False)
    measurements = Measurements(
        mode=mode,
        source_paths={
            "AuthService": inspect.getfile(AuthService),
            "EventExportService": inspect.getfile(EventExportService),
            "EventRepository": inspect.getfile(EventRepository),
            "CameraEvent": inspect.getfile(CameraEvent),
        },
    )
    source_root = Path(os.environ.get("GW_BENCHMARK_SOURCE_DIR", str(_SOURCE_DIR)))
    if any(
        not Path(path).resolve().is_relative_to(source_root.resolve())
        for path in measurements.source_paths.values()
    ):
        exception_message = "benchmark imported project code outside the mounted worktree source"
        raise RuntimeError(exception_message)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _ = await auth.initialize_password(password)
        app.state.measurements = measurements
        yield
        await database.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_session_router(auth, config))

    def _require_session_factory(*, user_action: bool) -> SessionDependency:
        dependency = require_session(auth, config, user_action=user_action)

        async def adapt_session(
            request: Request,
            response: Response,
        ) -> RouteAuthenticatedRequest:
            authenticated = await dependency(request, response)
            return RouteAuthenticatedRequest(session_id=str(authenticated.session.session_id))

        return adapt_session

    exporter = (
        BufferedEventExporter(database) if mode == "buffered" else EventExportService(database)
    )
    app.include_router(
        build_event_router(
            database=database,
            require_session=_require_session_factory,
            exporter=TimedExporter(exporter, measurements),
        )
    )

    _register_control_routes(app, database, control_token)
    return app


def main() -> None:
    """Serve the benchmark-only API on its private Docker network."""
    uvicorn.run(
        build_app(),
        host="0.0.0.0",  # noqa: S104 - internal Docker network; no host ports are published.
        port=8000,
        workers=1,
        access_log=False,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
