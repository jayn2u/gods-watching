"""Run the real-API person search UI spec against the worker, API, and built web app."""

import json
import os
import secrets
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast
from uuid import UUID

import anyio
from cryptography.fernet import Fernet
from sqlalchemy import select

from gods_watching.cameras import CameraRepository, CameraService
from gods_watching.storage import Camera, CredentialCipher, Database, StorageRepository
from gods_watching.verification.context import allocate_loopback_port

from .pipeline_worker_runtime import (
    WorkerStackSettings,
    create_cameras,
    kill_worker,
    start_worker,
    stop_worker,
    visible_rows,
    wait_until,
)
from .task11_runtime import wait_triton
from .task16_errors import Task16ExecutionError
from .task16_models import PlaywrightPhase, SearchUiEvidence, Task16DriverEvidence

if TYPE_CHECKING:
    from anyio.abc import Process

_REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[5]
_PUBLISH_TIMEOUT_SECONDS: Final = 180.0
_APP_READY_TIMEOUT_SECONDS: Final = 60.0
_NORMAL_PHASE_SECONDS: Final = 300.0
_OUTAGE_MARKER_SECONDS: Final = 150.0
_TRITON_RECOVERY_SECONDS: Final = 240.0
_OUTAGE_PHASE_SECONDS: Final = 360.0
_SEARCH_MODELS: Final = ("detector", "clip_image", "clip_text")
_CAMERA_LABEL: Final = "task16"


@dataclass(frozen=True, slots=True)
class _Stack:
    worker: WorkerStackSettings
    triton_container: str
    app_log: Path
    e2e_root: Path
    password: str


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise Task16ExecutionError(detail=f"required environment variable is missing: {name}")
    return value


async def _camera_names(database: Database, camera_ids: tuple[UUID, ...]) -> list[str]:
    async with database.transaction() as session:
        rows = await session.execute(
            select(Camera.id, Camera.name).where(Camera.id.in_(camera_ids))
        )
        names = dict(rows.tuples().all())
    return [names[camera_id] for camera_id in camera_ids]


async def _wait_each_camera_published(database: Database, camera_ids: tuple[UUID, ...]) -> None:
    async def every_camera() -> bool:
        published = {row.camera_id for row in await visible_rows(database)}
        return all(camera_id in published for camera_id in camera_ids)

    if not await wait_until(every_camera, _PUBLISH_TIMEOUT_SECONDS):
        raise Task16ExecutionError(detail="real worker did not publish on every camera")


async def _start_app(stack: _Stack, port: int) -> "Process":
    worker = stack.worker
    environment = {
        **os.environ,
        "GW_DATABASE_URL": worker.database_url,
        "GW_TRITON_GRPC_URL": worker.triton_url,
        "GW_CROPS_ROOT": str(worker.crop_root),
        "GW_CAMERA_CIPHER_KEY": worker.cipher_key,
        "GW_OPERATOR_PASSWORD": stack.password,
        "GW_TASK14_PUBLIC_ORIGIN": f"http://127.0.0.1:{port}",
    }
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "gods_watching.verification.scenarios.task16_app:app",
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


async def _wait_app(port: int) -> None:
    async def ready() -> bool:
        try:
            stream = await anyio.connect_tcp("127.0.0.1", port)
        except OSError:
            return False
        await stream.aclose()
        return True

    if not await wait_until(ready, _APP_READY_TIMEOUT_SECONDS):
        raise Task16ExecutionError(detail="search UI origin did not become ready")


def _playwright_environment(
    stack: _Stack, port: int, names: list[str], *, phase: str, markers: tuple[Path, Path]
) -> dict[str, str]:
    return {
        **os.environ,
        "GW_BASE_URL": f"http://127.0.0.1:{port}",
        "GW_E2E_OPERATOR_PASSWORD": stack.password,
        "GW_E2E_CAMERA_NAME": names[0],
        "GW_E2E_OTHER_CAMERA_NAME": names[1],
        "GW_E2E_SEARCH_PHASE": phase,
        "GW_E2E_EVIDENCE_ROOT": str(stack.e2e_root / phase),
        "GW_E2E_OUTAGE_MARKER": str(markers[0]),
        "GW_E2E_RECOVERY_MARKER": str(markers[1]),
    }


def _playwright_command() -> list[str]:
    return [
        "pnpm",
        "--dir",
        str(_REPOSITORY_ROOT / "web"),
        "exec",
        "playwright",
        "test",
        "e2e/search.spec.ts",
        "--workers=1",
    ]


def _phase_summary(stack: _Stack, phase: str, exit_code: int) -> PlaywrightPhase:
    results = stack.e2e_root / phase / "playwright-results.json"
    stats: dict[str, object] = {}
    if results.is_file():
        decoded = cast("object", json.loads(results.read_text(encoding="utf-8")))
        if isinstance(decoded, dict):
            raw_stats = cast("dict[str, object]", decoded).get("stats")
            if isinstance(raw_stats, dict):
                stats = cast("dict[str, object]", raw_stats)

    def count(key: str) -> int:
        value = stats.get(key)
        return value if isinstance(value, int) else -1

    return PlaywrightPhase(
        exit_code=exit_code,
        expected=count("expected"),
        unexpected=count("unexpected"),
        skipped=count("skipped"),
        flaky=count("flaky"),
    )


async def _run_phase(
    stack: _Stack, port: int, names: list[str], *, phase: str, markers: tuple[Path, Path]
) -> "Process":
    log_path = stack.e2e_root / f"playwright-{phase}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log:
        return await anyio.open_process(
            _playwright_command(),
            env=_playwright_environment(stack, port, names, phase=phase, markers=markers),
            stdout=log,
            stderr=log,
        )


async def _wait_process(process: "Process", deadline_seconds: float) -> int:
    with anyio.move_on_after(deadline_seconds):
        return await process.wait()
    _ = await kill_worker(process)
    return -1


async def _outage_phase(
    stack: _Stack, port: int, names: list[str]
) -> tuple[PlaywrightPhase, bool, bool]:
    outage_marker = stack.e2e_root / "outage-observed"
    recovery_marker = stack.e2e_root / "inference-recovered"
    stopped = await anyio.run_process(["docker", "stop", stack.triton_container], check=False)
    if stopped.returncode != 0:
        raise Task16ExecutionError(detail="inference container could not be stopped")
    process = await _run_phase(
        stack, port, names, phase="outage", markers=(outage_marker, recovery_marker)
    )

    async def marker_written() -> bool:
        return outage_marker.is_file()

    marker_seen = await wait_until(marker_written, _OUTAGE_MARKER_SECONDS)
    recovered = False
    if marker_seen:
        started = await anyio.run_process(["docker", "start", stack.triton_container], check=False)
        recovered = started.returncode == 0 and await wait_triton(
            stack.worker.triton_url, _SEARCH_MODELS
        )
        if recovered:
            _ = recovery_marker.write_text("inference recovered\n", encoding="utf-8")
    exit_code = await _wait_process(process, _OUTAGE_PHASE_SECONDS)
    return _phase_summary(stack, "outage", exit_code), marker_seen, recovered


async def _drive(stack: _Stack) -> Task16DriverEvidence:
    settings = stack.worker
    database = Database.connect(settings.database_url)
    storage = StorageRepository(CredentialCipher(settings.cipher_key.encode()))
    worker: Process | None = None
    app: Process | None = None
    try:
        cameras = CameraService(CameraRepository(storage))
        created = await create_cameras(database, cameras, settings.rtsp_host, label=_CAMERA_LABEL)
        camera_ids = tuple(UUID(str(camera_id)) for camera_id in created)
        names = await _camera_names(database, camera_ids)
        worker = await start_worker(settings)
        await _wait_each_camera_published(database, camera_ids)
        port = allocate_loopback_port()
        app = await _start_app(stack, port)
        await _wait_app(port)

        normal_process = await _run_phase(
            stack, port, names, phase="normal", markers=(Path(os.devnull), Path(os.devnull))
        )
        normal = _phase_summary(
            stack, "normal", await _wait_process(normal_process, _NORMAL_PHASE_SECONDS)
        )
        _ = await stop_worker(worker, signal.SIGTERM)
        worker = None
        outage, marker_seen, recovered = await _outage_phase(stack, port, names)
        return Task16DriverEvidence(
            ui=SearchUiEvidence(
                normal=normal,
                outage=outage,
                outage_marker_seen=marker_seen,
                inference_recovered=recovered,
            )
        )
    finally:
        with anyio.CancelScope(shield=True):
            for process in (worker, app):
                if process is not None and process.returncode is None:
                    _ = await kill_worker(process)
            await database.close()


async def _main() -> None:
    stack = _Stack(
        worker=WorkerStackSettings(
            database_url=_env("GW_TASK16_DATABASE_URL"),
            triton_url=_env("GW_TASK16_TRITON_URL"),
            rtsp_host=_env("GW_TASK16_RTSP_HOST"),
            crop_root=Path(_env("GW_TASK16_CROP_ROOT")),
            cipher_key=Fernet.generate_key().decode(),
            worker_log=Path(_env("GW_TASK16_WORKER_LOG")),
        ),
        triton_container=_env("GW_TASK16_TRITON_CONTAINER"),
        app_log=Path(_env("GW_TASK16_APP_LOG")),
        e2e_root=Path(_env("GW_TASK16_E2E_ROOT")),
        password=secrets.token_urlsafe(24),
    )
    async with anyio.create_task_group() as task_group:

        async def cancel_on_signal() -> None:
            # Harness termination cancels the drive so owned processes are killed.
            with anyio.open_signal_receiver(signal.SIGINT, signal.SIGTERM) as signals:
                async for _signum in signals:
                    task_group.cancel_scope.cancel()
                    return

        task_group.start_soon(cancel_on_signal)
        evidence = await _drive(stack)
        output = Path(_env("GW_TASK16_OUTPUT"))
        _ = output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
        task_group.cancel_scope.cancel()


if __name__ == "__main__":
    anyio.run(_main)
