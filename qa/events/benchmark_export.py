"""Reproducible HTTP benchmark for streaming and full-buffer event exports."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import http.client
import json
import os
import platform
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from http.cookies import SimpleCookie
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "server" / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from gods_watching.contracts.events import EventExportFilters
from gods_watching.events.repository import EventRepository
from gods_watching.storage import CameraEvent, Database

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

_CSV_FIELDS = ("id", "occurred_at", "event_type", "camera_id", "camera_name")
_POSTGRES_IMAGE = "pgvector/pgvector:0.8.1-pg17"
_APP_IMAGE = "gods-watching-app:local"
_SAMPLE_COUNTS = (10_000, 50_000)
_REPEATS = 3
_WRITER_COUNT = 100
_FORMULA_PREFIXES = frozenset("=+-@")
_CONTROL_PREFIXES = frozenset(("\t", "\r", "\n"))


@dataclass(frozen=True, slots=True)
class CsvValidationResult:
    """Summarize exact CSV and snapshot-ID validation without retaining output rows."""

    expected_count: int
    actual_count: int
    missing_ids: tuple[int, ...]
    duplicate_ids: tuple[int, ...]
    unexpected_ids: tuple[int, ...]
    field_mismatch_count: int
    ids_in_order: bool
    expected_ids_sha256: str
    actual_ids_sha256: str
    parse_error: str | None
    correct: bool
    performance_eligible: bool


def _id_digest(ids: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for identifier in ids:
        digest.update(str(identifier).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_csv_file(
    path: Path,
    expected_rows: Mapping[int, Mapping[str, str]],
) -> CsvValidationResult:
    """Check the downloaded CSV against the pre-insertion SELECT snapshot."""
    expected_ids = sorted(expected_rows)
    actual_ids: list[int] = []
    seen: Counter[int] = Counter()
    unexpected: set[int] = set()
    mismatch_count = 0
    parse_error: str | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if tuple(reader.fieldnames or ()) != _CSV_FIELDS:
                parse_error = "CSV header does not match the export contract"
            else:
                for row in reader:
                    if row is None or any(row.get(field) is None for field in _CSV_FIELDS):
                        parse_error = "CSV row has missing or extra columns"
                        break
                    identifier_text = row["id"]
                    try:
                        identifier = int(identifier_text)
                    except (TypeError, ValueError):
                        parse_error = "CSV event ID is not an integer"
                        break
                    actual_ids.append(identifier)
                    seen[identifier] += 1
                    expected = expected_rows.get(identifier)
                    if expected is None:
                        unexpected.add(identifier)
                        continue
                    if any(row[field] != expected[field] for field in _CSV_FIELDS):
                        mismatch_count += 1
    except (OSError, UnicodeError, csv.Error) as error:
        parse_error = f"{type(error).__name__}: {error}"

    missing = tuple(sorted(set(expected_ids) - set(seen)))
    duplicates = tuple(sorted(identifier for identifier, count in seen.items() if count > 1))
    unexpected_ids = tuple(sorted(unexpected))
    order_matches = actual_ids == expected_ids
    correct = (
        parse_error is None
        and len(actual_ids) == len(expected_ids)
        and not missing
        and not duplicates
        and not unexpected_ids
        and mismatch_count == 0
        and order_matches
    )
    return CsvValidationResult(
        expected_count=len(expected_ids),
        actual_count=len(actual_ids),
        missing_ids=missing,
        duplicate_ids=duplicates,
        unexpected_ids=unexpected_ids,
        field_mismatch_count=mismatch_count,
        ids_in_order=order_matches,
        expected_ids_sha256=_id_digest(expected_ids),
        actual_ids_sha256=_id_digest(actual_ids),
        parse_error=parse_error,
        correct=correct,
        performance_eligible=correct,
    )


def ensure_disposable_database(
    database_url: str,
    *,
    expected_database: str | None = None,
) -> None:
    """Refuse targets other than this run's local disposable benchmark database."""
    try:
        url = make_url(database_url)
    except Exception as error:
        raise ValueError("target must be a disposable benchmark database") from error
    database = url.database or ""
    if (
        url.drivername != "postgresql+asyncpg"
        or url.host not in {"127.0.0.1", "localhost", "db"}
        or url.username != "postgres"
        or re.fullmatch(r"gw_events_bench_[0-9a-f]{8}", database) is None
        or (expected_database is not None and database != expected_database)
    ):
        raise ValueError("target must be a disposable benchmark database")


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize only correct samples; an incorrect body cannot count as a gain."""
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        mode = str(sample.get("mode", "unknown"))
        values = grouped.setdefault(mode, [])
        if sample.get("correctness") is True:
            latency = sample.get("latency_seconds")
            if isinstance(latency, (float, int)) and latency >= 0:
                values.append(float(latency))
    return {
        mode: {
            "sample_count": len(latencies),
            "median_latency_seconds": statistics.median(latencies) if latencies else None,
            "latency_range_seconds": [min(latencies), max(latencies)] if latencies else None,
        }
        for mode, latencies in grouped.items()
    }


def summarize_samples_by_dataset(
    samples: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize only correctness-eligible samples within each dataset/mode cell."""
    summary: dict[str, dict[str, Any]] = {}
    metrics = {
        "latency_seconds": "latency_seconds",
        "first_byte_latency_seconds": "first_byte_latency_seconds",
        "application_peak_rss_kib": "application_peak_rss_kib",
        "startup_login_baseline_rss_kib": "startup_login_baseline_rss_kib",
    }
    for size in _SAMPLE_COUNTS:
        for mode in ("streaming", "buffered"):
            selected = [
                sample
                for sample in samples
                if sample.get("dataset_size") == size
                and sample.get("mode") == mode
                and sample.get("correctness") is True
            ]
            cell: dict[str, Any] = {"sample_count": len(selected)}
            for label, field in metrics.items():
                values = [
                    float(sample[field])
                    for sample in selected
                    if isinstance(sample.get(field), (int, float))
                ]
                cell[f"{label}_median"] = statistics.median(values) if values else None
                cell[f"{label}_range"] = [min(values), max(values)] if values else None
            summary[f"{size}:{mode}"] = cell
    return summary


def _safe_export_name(value: str) -> str:
    stripped = value.lstrip()
    if value and (
        value[0] in _CONTROL_PREFIXES
        or (stripped and stripped[0] in _FORMULA_PREFIXES)
    ):
        return "'" + value
    return value


def _seed_name(index: int) -> str:
    match index:
        case 0:
            return "=SUM(1,1)"
        case 1:
            return "\tControl prefix"
        case 2:
            return 'Quoted "camera", south'
        case 3:
            return "Line one\nLine two"
        case _:
            return f"Synthetic camera {index % 128:03d}"


def _event_json(row: Any) -> dict[str, str]:
    timestamp = row.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return {
        "id": str(row.id),
        "occurred_at": timestamp,
        "event_type": row.event_type,
        "camera_id": str(row.camera_id),
        "camera_name": _safe_export_name(row.camera_name),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _docker(arguments: Sequence[str], *, timeout: int = 60) -> str:
    try:
        completed = subprocess.run(
            ["docker", *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or f"exit code {error.returncode}"
        raise RuntimeError(f"docker {arguments[0]} failed: {detail}") from None
    return completed.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    names = ("alembic", "anyio", "argon2-cffi", "asyncpg", "fastapi", "SQLAlchemy", "uvicorn")
    return {name: metadata.version(name) for name in names}


def _docker_image_id(image: str) -> str:
    return _docker(["image", "inspect", image, "--format", "{{.Id}}"])


def _write_expected_file(path: Path, rows: Sequence[Any]) -> dict[int, dict[str, str]]:
    expected_rows = {int(row.id): _event_json(row) for row in rows}
    _write_json(path, {"rows": list(expected_rows.values())})
    return expected_rows


async def _seed_database(
    database: Database,
    *,
    row_count: int,
    expected_path: Path,
) -> dict[int, dict[str, str]]:
    """Reset and populate only the disposable table; return its exact CSV projection."""
    base_time = datetime(2026, 10, 3, tzinfo=UTC)
    event_types = ("camera.created", "camera.updated", "camera.deleted")
    async with database.engine.begin() as connection:
        _ = await connection.execute(text("TRUNCATE TABLE camera_events RESTART IDENTITY"))
        for first in range(0, row_count, 1_000):
            values = [
                {
                    "occurred_at": base_time + timedelta(seconds=index),
                    "event_type": event_types[index % len(event_types)],
                    "camera_id": UUID(int=(index % 128) + 1),
                    "camera_name": _seed_name(index),
                }
                for index in range(first, min(first + 1_000, row_count))
            ]
            _ = await connection.execute(insert(CameraEvent), values)
        result = await connection.execute(EventRepository.statement(EventExportFilters()))
        projected = result.all()
    expected = [
        _ProjectionRow(
            id=int(row[0]),
            occurred_at=cast("datetime", row[1]),
            event_type=str(row[2]),
            camera_id=cast("UUID", row[3]),
            camera_name=str(row[4]),
        )
        for row in projected
    ]
    return _write_expected_file(expected_path, expected)


@dataclass(frozen=True, slots=True)
class _ProjectionRow:
    id: int
    occurred_at: datetime
    event_type: str
    camera_id: UUID
    camera_name: str


async def _writer_insert(database: Database, *, offset: int = 0) -> dict[str, Any]:
    """Commit a fixed event batch through the production repository."""
    from gods_watching.events.repository import EventRepository

    start_mono = time.monotonic()
    start_utc = datetime.now(UTC).isoformat()
    identifiers: list[int] = []
    async with database.session_factory.begin() as session:
        for index in range(_WRITER_COUNT):
            event = await EventRepository.record(
                session,
                event_type="camera.updated",
                camera_id=UUID(int=10_000 + offset + index),
                camera_name=f"Concurrent writer event {index:03d}",
            )
            identifiers.append(int(event.id))
    commit_mono = time.monotonic()
    commit_utc = datetime.now(UTC).isoformat()
    end_mono = time.monotonic()
    end_utc = datetime.now(UTC).isoformat()
    return {
        "count": len(identifiers),
        "ids": identifiers,
        "started_monotonic": start_mono,
        "committed_monotonic": commit_mono,
        "ended_monotonic": end_mono,
        "started_at_utc": start_utc,
        "committed_at_utc": commit_utc,
        "ended_at_utc": end_utc,
        "commit_seconds": commit_mono - start_mono,
    }


def _make_engine(database_url: str) -> Database:
    ensure_disposable_database(database_url)
    engine = create_async_engine(database_url, poolclass=NullPool, pool_pre_ping=True)
    return Database(engine, async_sessionmaker(engine, expire_on_commit=False))


def _client_json_request(
    app_host: str,
    method: str,
    path: str,
    *,
    control_token: str,
    payload: Mapping[str, Any] | None = None,
    cookie: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any], http.client.HTTPMessage]:
    connection = http.client.HTTPConnection(app_host, 8000, timeout=timeout)
    headers = {"X-Benchmark-Control": control_token}
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if cookie is not None:
        headers["Cookie"] = cookie
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_payload = response.read()
        decoded = json.loads(response_payload) if response_payload else {}
        if not isinstance(decoded, dict):
            raise RuntimeError("benchmark control endpoint returned an invalid response")
        return response.status, cast("dict[str, Any]", decoded), response.headers
    finally:
        connection.close()


def _load_expected_rows(path: Path) -> dict[int, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("expected-row manifest is invalid")
    expected: dict[int, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("expected-row manifest is invalid")
        expected[int(row["id"])] = {str(field): str(value) for field, value in row.items()}
    return expected


def _wait_for_app(app_host: str, control_token: str) -> None:
    deadline = time.monotonic() + 60.0
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            status, _, _ = _client_json_request(
                app_host,
                "GET",
                "/__benchmark/ready",
                control_token=control_token,
                timeout=2.0,
            )
            if status == 200:
                return
            last_error = f"HTTP {status}"
        except OSError as error:
            last_error = type(error).__name__
        time.sleep(0.2)
    raise RuntimeError(f"benchmark app was not ready within 60 seconds ({last_error})")


def run_client_sample(
    *,
    app_host: str,
    control_token: str,
    password: str,
    expected_path: Path,
    output_csv: Path,
    result_path: Path,
    writer_enabled: bool,
) -> dict[str, Any]:
    """Login, stream the HTTP body to disk, trigger one asynchronous writer, and validate."""
    _wait_for_app(app_host, control_token)
    login_connection = http.client.HTTPConnection(app_host, 8000, timeout=30.0)
    login_connection.request(
        "POST",
        "/api/session",
        body=json.dumps({"username": "admin", "password": password}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Origin": "http://app:8000",
        },
    )
    login_response = login_connection.getresponse()
    login_body = login_response.read()
    cookie_header = login_response.getheader("Set-Cookie")
    login_connection.close()
    if login_response.status != 200 or cookie_header is None:
        raise RuntimeError(f"real operator login failed with HTTP {login_response.status}")
    cookies = SimpleCookie()
    cookies.load(cookie_header)
    session_cookie = cookies.get("gw_session")
    if session_cookie is None:
        raise RuntimeError("real operator login did not set its session cookie")
    login_payload = json.loads(login_body)
    if login_payload.get("authenticated") is not True:
        raise RuntimeError("real operator login response was not authenticated")

    baseline_status, baseline, _ = _client_json_request(
        app_host,
        "POST",
        "/__benchmark/baseline",
        control_token=control_token,
    )
    if baseline_status != 200:
        raise RuntimeError("application RSS baseline was unavailable")

    connection = http.client.HTTPConnection(app_host, 8000, timeout=120.0)
    request_started = time.monotonic()
    connection.request(
        "GET",
        "/api/events/export.csv",
        headers={"Cookie": f"gw_session={session_cookie.value}"},
    )
    response = connection.getresponse()
    if response.status != 200:
        error_body = response.read(4_096).decode("utf-8", errors="replace")
        connection.close()
        raise RuntimeError(f"HTTP export failed with status {response.status}: {error_body}")
    if response.getheader("Content-Type", "").split(";", maxsplit=1)[0] != "text/csv":
        connection.close()
        raise RuntimeError("HTTP export content type was not text/csv")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    byte_count = 0
    body_hash = hashlib.sha256()
    first_byte_latency: float | None = None
    writer_triggered_at: float | None = None
    writer_request: dict[str, Any] = {"requested": False}
    writer_thread: threading.Thread | None = None
    writer_exception: list[str] = []

    def _read_byte(stream: http.client.HTTPResponse) -> bytes:
        nonlocal first_byte_latency
        value = stream.read(1)
        if value and first_byte_latency is None:
            first_byte_latency = time.monotonic() - request_started
        return value

    def _trigger_writer() -> None:
        started = time.monotonic()
        writer_request["client_started_monotonic"] = started
        try:
            status, payload, _ = _client_json_request(
                app_host,
                "POST",
                "/__benchmark/writer",
                control_token=control_token,
                timeout=120.0,
            )
            writer_request["http_status"] = status
            writer_request["response_count"] = payload.get("count", 0)
        except Exception as error:
            writer_exception.append(type(error).__name__)
        writer_request["client_ended_monotonic"] = time.monotonic()
        writer_request["requested"] = True

    with output_csv.open("wb") as output:
        for line_number in range(2):
            while True:
                byte = _read_byte(response)
                if not byte:
                    connection.close()
                    raise RuntimeError("HTTP export ended before its first CSV data row")
                output.write(byte)
                body_hash.update(byte)
                byte_count += 1
                if byte == b"\n":
                    break
            if line_number == 1 and writer_enabled:
                writer_triggered_at = time.monotonic()
                writer_thread = threading.Thread(target=_trigger_writer, name="event-benchmark-writer")
                writer_thread.start()
        while True:
            chunk = response.read(64 * 1024)
            if not chunk:
                break
            if first_byte_latency is None:
                first_byte_latency = time.monotonic() - request_started
            output.write(chunk)
            body_hash.update(chunk)
            byte_count += len(chunk)
    transfer_ended = time.monotonic()
    connection.close()
    if writer_thread is not None:
        writer_thread.join(timeout=120.0)
        if writer_thread.is_alive():
            writer_exception.append("writer_timeout")

    metrics_status, app_metrics, _ = _client_json_request(
        app_host,
        "GET",
        "/__benchmark/metrics",
        control_token=control_token,
    )
    if metrics_status != 200:
        raise RuntimeError("application metrics endpoint was unavailable")
    expected_rows = _load_expected_rows(expected_path)
    validation = validate_csv_file(output_csv, expected_rows)
    writer_metrics = app_metrics.get("writer")
    writer_count = int(writer_metrics.get("count", 0)) if isinstance(writer_metrics, dict) else 0
    writer_ok = (not writer_enabled) or (writer_count == _WRITER_COUNT and not writer_exception)
    result: dict[str, Any] = {
        "http_request_started_monotonic": request_started,
        "http_transfer_ended_monotonic": transfer_ended,
        "latency_seconds": transfer_ended - request_started,
        "first_byte_latency_seconds": first_byte_latency,
        "bytes": byte_count,
        "body_sha256": body_hash.hexdigest(),
        "csv_validation": asdict(validation),
        "export_correctness": validation.correct,
        "writer_triggered_after_first_data_row": writer_triggered_at is not None,
        "writer_request": writer_request,
        "writer_request_errors": writer_exception,
        "writer_count": writer_count,
        "writer_correctness": writer_ok,
        "application_metrics": app_metrics,
        "startup_login_baseline_rss_kib": baseline.get("peak_rss_kib"),
    }
    result["correctness"] = validation.correct and writer_ok
    result["performance_eligible"] = result["correctness"]
    _write_json(result_path, result)
    return result


def _client_main(arguments: argparse.Namespace) -> int:
    run_client_sample(
        app_host=arguments.app_host,
        control_token=os.environ["GW_BENCHMARK_CONTROL_TOKEN"],
        password=os.environ["GW_BENCHMARK_PASSWORD"],
        expected_path=Path(arguments.expected),
        output_csv=Path(arguments.output_csv),
        result_path=Path(arguments.result),
        writer_enabled=arguments.writer,
    )
    return 0


class BenchmarkRunner:
    """Own one isolated Docker network, PostgreSQL container, and all output files."""

    def __init__(self) -> None:
        self.run_id = uuid4().hex[:8]
        self.started_at_utc = datetime.now(UTC).isoformat()
        self.database_name = f"gw_events_bench_{self.run_id}"
        self.network_name = f"gw-event-export-{self.run_id}"
        self.db_container = f"gw-event-export-db-{self.run_id}"
        self.db_password = uuid4().hex
        self.operator_password = secrets.token_urlsafe(24)
        self.control_token = secrets.token_urlsafe(32)
        self.app_database_url = (
            f"postgresql+asyncpg://postgres:{self.db_password}"
            f"@db:5432/{self.database_name}"
        )
        ensure_disposable_database(
            self.app_database_url,
            expected_database=self.database_name,
        )
        self.scratch_root = Path(tempfile.mkdtemp(prefix=f"gw-event-export-{self.run_id}-"))
        self.output_root = REPO_ROOT / "docs" / "experiments" / "2026-10-03-event-export"
        self.logs_root = self.output_root / "logs"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.commands: list[str] = []
        self.owned_containers: set[str] = set()
        self.network_created = False
        self.database_version = "unavailable"
        self.alembic_head = "unverified"
        self.app_python_version = "unavailable"
        self.implementation_source_paths: dict[str, str] = {}
        self.samples: list[dict[str, Any]] = []
        self.smoke: list[dict[str, Any]] = []
        self.isolation_trials: list[dict[str, Any]] = []
        self.failures: list[dict[str, str]] = []
        self.resource_diagnostics: list[dict[str, Any]] = []
        self.current_phase = "initialization"

    def _run_docker(self, arguments: Sequence[str], *, timeout: int = 60) -> str:
        safe_arguments: list[str] = []
        for argument in arguments:
            if argument.startswith(
                (
                    "POSTGRES_PASSWORD=",
                    "GW_DATABASE_URL=",
                    "GW_BENCHMARK_PASSWORD=",
                    "GW_BENCHMARK_CONTROL_TOKEN=",
                )
            ):
                key = argument.split("=", maxsplit=1)[0]
                safe_arguments.append(f"{key}=<redacted synthetic value>")
            else:
                safe_arguments.append(argument)
        self.commands.append("docker " + " ".join(safe_arguments))
        return _docker(arguments, timeout=timeout)

    def _database_diagnostics(self, label: str) -> dict[str, Any]:
        """Capture effective Docker limits, cgroup counters, and tmpfs state."""
        snapshot: dict[str, Any] = {"label": label, "captured_at_utc": datetime.now(UTC).isoformat()}
        try:
            inspected = json.loads(self._run_docker(["inspect", self.db_container], timeout=15))[0]
        except (RuntimeError, IndexError, json.JSONDecodeError) as error:
            snapshot["inspect_error"] = f"{type(error).__name__}: {error}"
            return snapshot
        state = inspected.get("State", {})
        host_config = inspected.get("HostConfig", {})
        snapshot["container"] = {
            "status": state.get("Status"),
            "exit_code": state.get("ExitCode"),
            "oom_killed": state.get("OOMKilled"),
            "error": state.get("Error"),
            "effective_memory_bytes": host_config.get("Memory"),
            "effective_memory_swap_bytes": host_config.get("MemorySwap"),
            "effective_nano_cpus": host_config.get("NanoCpus"),
        }
        if state.get("Running") is True:
            try:
                snapshot["cgroup_v2"] = self._run_docker(
                    [
                        "exec",
                        self.db_container,
                        "sh",
                        "-c",
                        "for item in memory.current memory.peak memory.max memory.swap.current "
                        "memory.swap.max; do printf '%s=' \"$item\"; cat \"/sys/fs/cgroup/$item\" "
                        "2>/dev/null || true; done",
                    ],
                    timeout=15,
                )
                snapshot["tmpfs_df"] = self._run_docker(
                    ["exec", self.db_container, "df", "-B1", "/var/lib/postgresql/data"],
                    timeout=15,
                )
                snapshot["database_relation_bytes"] = self._run_docker(
                    [
                        "exec",
                        self.db_container,
                        "psql",
                        "-U",
                        "postgres",
                        "-d",
                        self.database_name,
                        "-Atc",
                        "SELECT pg_database_size(current_database()), "
                        "pg_total_relation_size('camera_events')",
                    ],
                    timeout=15,
                )
            except RuntimeError as error:
                snapshot["live_probe_error"] = f"{type(error).__name__}: {error}"
        try:
            snapshot["postgres_log_tail"] = self._run_docker(
                ["logs", "--tail", "200", self.db_container], timeout=15
            )
        except RuntimeError as error:
            snapshot["postgres_log_error"] = f"{type(error).__name__}: {error}"
        return snapshot

    def _start_environment(self) -> None:
        self._run_docker(["network", "create", "--internal", self.network_name])
        self.network_created = True
        self._run_docker(
            [
                "run",
                "--pull=never",
                "--detach",
                "--name",
                self.db_container,
                "--network",
                self.network_name,
                "--network-alias",
                "db",
                "--cpus=1",
                "--memory=384m",
                "--memory-swap=384m",
                "--shm-size=64m",
                "--tmpfs",
                "/var/lib/postgresql/data:rw,noexec,nosuid,size=256m",
                "--env",
                f"POSTGRES_PASSWORD={self.db_password}",
                "--env",
                f"POSTGRES_DB={self.database_name}",
                _POSTGRES_IMAGE,
            ],
            timeout=60,
        )
        self.owned_containers.add(self.db_container)
        self._initialize_database()

    def _initialize_database(self) -> None:
        ready = False
        for _ in range(120):
            try:
                self._run_docker(
                    [
                        "exec",
                        self.db_container,
                        "pg_isready",
                        "-h",
                        "127.0.0.1",
                        "-U",
                        "postgres",
                        "-d",
                        self.database_name,
                    ],
                    timeout=10,
                )
                ready = True
                break
            except RuntimeError:
                time.sleep(0.25)
        if not ready:
            raise RuntimeError("disposable PostgreSQL did not become ready on the private network")
        self._run_utility(
            suffix="migrate",
            command=("-m", "alembic", "-c", "/work/alembic.ini", "upgrade", "head"),
            script=False,
            include_alembic=True,
            timeout=120,
        )
        self.database_version = self._run_docker(
            [
                "exec",
                self.db_container,
                "psql",
                "-U",
                "postgres",
                "-d",
                self.database_name,
                "-Atc",
                "SELECT version()",
            ],
            timeout=20,
        )
        self.alembic_head = self._run_docker(
            [
                "exec",
                self.db_container,
                "psql",
                "-U",
                "postgres",
                "-d",
                self.database_name,
                "-Atc",
                "SELECT version_num FROM alembic_version",
            ],
            timeout=20,
        )
        if self.alembic_head != "0008_camera_events":
            raise RuntimeError(f"unexpected disposable database Alembic head: {self.alembic_head}")

    def _run_utility(
        self,
        *,
        suffix: str,
        command: Sequence[str],
        script: bool = True,
        include_alembic: bool = False,
        timeout: int = 300,
    ) -> None:
        utility_name = f"gw-event-export-util-{self.run_id}-{suffix}"
        arguments = [
            "run",
            "--pull=never",
            "--rm",
            "--name",
            utility_name,
            "--network",
            self.network_name,
            "--cpus=1",
            "--memory=256m",
            "--memory-swap=256m",
            "--mount",
            f"type=bind,src={SOURCE_ROOT},dst=/work/server/src,readonly",
            "--mount",
            f"type=bind,src={REPO_ROOT / 'qa' / 'events'},dst=/work/events,readonly",
            "--mount",
            f"type=bind,src={self.scratch_root},dst=/work/results",
            "--env",
            "PYTHONPATH=/work/server/src",
            "--env",
            f"GW_DATABASE_URL={self.app_database_url}",
            "--env",
            f"GW_BENCHMARK_DATABASE_NAME={self.database_name}",
        ]
        if include_alembic:
            arguments.extend(
                [
                    "--mount",
                    f"type=bind,src={REPO_ROOT / 'server' / 'alembic'},dst=/work/server/alembic,readonly",
                    "--mount",
                    f"type=bind,src={REPO_ROOT / 'alembic.ini'},dst=/work/alembic.ini,readonly",
                ]
            )
        arguments.extend(["--workdir", "/work", _APP_IMAGE])
        arguments.append("python")
        if script:
            arguments.extend(["/work/events/benchmark_export.py", *command])
        else:
            arguments.extend(command)
        self._run_docker(arguments, timeout=timeout)

    def _start_app(self, *, mode: str, container_name: str) -> None:
        self._run_docker(
            [
                "run",
                "--pull=never",
                "--detach",
                "--name",
                container_name,
                "--network",
                self.network_name,
                "--network-alias",
                "app",
                "--cpus=1",
                "--memory=256m",
                "--memory-swap=256m",
                "--mount",
                f"type=bind,src={SOURCE_ROOT},dst=/work/server/src,readonly",
                "--mount",
                f"type=bind,src={REPO_ROOT / 'qa' / 'events'},dst=/work/events,readonly",
                "--env",
                "PYTHONPATH=/work/server/src",
                "--env",
                "GW_BENCHMARK_SOURCE_DIR=/work/server/src",
                "--env",
                f"GW_DATABASE_URL={self.app_database_url}",
                "--env",
                f"GW_BENCHMARK_DATABASE_NAME={self.database_name}",
                "--env",
                f"GW_BENCHMARK_PASSWORD={self.operator_password}",
                "--env",
                f"GW_BENCHMARK_CONTROL_TOKEN={self.control_token}",
                "--env",
                f"GW_BENCHMARK_MODE={mode}",
                "--workdir",
                "/work",
                _APP_IMAGE,
                "python",
                "/work/events/benchmark_app.py",
            ],
            timeout=60,
        )
        self.owned_containers.add(container_name)

    def _run_client(
        self,
        *,
        app_container: str,
        mode: str,
        row_count: int,
        repeat: int,
        writer_enabled: bool,
        expected_path: Path,
    ) -> dict[str, Any]:
        client_name = f"gw-event-export-client-{self.run_id}-{mode}-{row_count}-{repeat}"
        output_csv = self.scratch_root / f"{mode}-{row_count}-{repeat}.csv"
        result_path = self.scratch_root / f"{mode}-{row_count}-{repeat}.json"
        command = [
            "run",
            "--pull=never",
            "--rm",
            "--name",
            client_name,
            "--network",
            self.network_name,
            "--cpus=1",
            "--memory=256m",
            "--memory-swap=256m",
            "--mount",
            f"type=bind,src={SOURCE_ROOT},dst=/work/server/src,readonly",
            "--mount",
            f"type=bind,src={REPO_ROOT / 'qa' / 'events'},dst=/work/events,readonly",
            "--mount",
            f"type=bind,src={self.scratch_root},dst=/work/results",
            "--env",
            "PYTHONPATH=/work/server/src",
            "--env",
            f"GW_BENCHMARK_PASSWORD={self.operator_password}",
            "--env",
            f"GW_BENCHMARK_CONTROL_TOKEN={self.control_token}",
            _APP_IMAGE,
            "python",
            "/work/events/benchmark_export.py",
            "client-sample",
            "--app-host",
            "app",
            "--expected",
            f"/work/results/{expected_path.name}",
            "--output-csv",
            f"/work/results/{output_csv.name}",
            "--result",
            f"/work/results/{result_path.name}",
        ]
        if writer_enabled:
            command.append("--writer")
        self._run_docker(command, timeout=300)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["mode"] = mode
        result["dataset_size"] = row_count
        result["repeat"] = repeat
        result["writer_count_expected"] = _WRITER_COUNT if writer_enabled else 0
        result["app_container"] = app_container
        result["client_container"] = client_name
        if result.get("correctness") is not True:
            invalid_root = self.output_root / "incorrect-output"
            invalid_root.mkdir(parents=True, exist_ok=True)
            if output_csv.exists():
                shutil.copyfile(output_csv, invalid_root / output_csv.name)
        output_csv.unlink(missing_ok=True)
        return result

    def _run_http_sample(
        self,
        *,
        mode: str,
        row_count: int,
        repeat: int,
        writer_enabled: bool,
        expected_path: Path,
    ) -> dict[str, Any]:
        app_container = f"gw-event-export-app-{self.run_id}-{mode}-{row_count}-{repeat}"
        self._start_app(mode=mode, container_name=app_container)
        try:
            result = self._run_client(
                app_container=app_container,
                mode=mode,
                row_count=row_count,
                repeat=repeat,
                writer_enabled=writer_enabled,
                expected_path=expected_path,
            )
            self.app_python_version = self._run_docker(
                ["exec", app_container, "python", "--version"]
            )
            writer = result.get("application_metrics", {}).get("writer")
            writer_interval = (
                (writer.get("started_monotonic"), writer.get("committed_monotonic"))
                if isinstance(writer, dict)
                else None
            )
            http_start = result.get("http_request_started_monotonic")
            http_end = result.get("http_transfer_ended_monotonic")
            http_overlap = False
            fetch_encode_overlap: bool | None = None
            if writer_interval and isinstance(http_start, (int, float)) and isinstance(http_end, (int, float)):
                writer_start, writer_commit = writer_interval
                http_overlap = writer_start <= http_end and writer_commit >= http_start
                spans = result.get("application_metrics", {}).get("chunk_pull_spans", [])
                fetch_spans = [
                    span
                    for span in spans
                    if isinstance(span, dict)
                    and int(span.get("index", 0)) >= 2
                    and isinstance(span.get("start"), (int, float))
                    and isinstance(span.get("end"), (int, float))
                ]
                fetch_encode_overlap = any(
                    writer_start <= float(span["end"])
                    and writer_commit >= float(span["start"])
                    for span in fetch_spans
                )
            result["writer_http_transfer_overlap"] = http_overlap
            result["writer_stream_pull_fetch_encode_overlap"] = fetch_encode_overlap
            result["writer_database_fetch_overlap_status"] = (
                "not separately observable; app-side stream-pull intervals include fetch and encoding"
            )
            app_metrics = result.get("application_metrics", {})
            result["application_peak_rss_kib"] = app_metrics.get("application_peak_rss_kib")
            source_paths = app_metrics.get("source_paths")
            if isinstance(source_paths, dict):
                self.implementation_source_paths = {
                    str(key): str(value) for key, value in source_paths.items()
                }
            baseline_rss = result.get("startup_login_baseline_rss_kib")
            peak_rss = result.get("application_peak_rss_kib")
            result["application_rss_high_water_delta_kib"] = (
                peak_rss - baseline_rss
                if isinstance(peak_rss, int) and isinstance(baseline_rss, int)
                else None
            )
            self._save_app_log(app_container, mode, row_count, repeat)
            return result
        finally:
            self._run_docker(["rm", "--force", app_container], timeout=30)
            self.owned_containers.discard(app_container)

    def _save_app_log(self, container_name: str, mode: str, count: int, repeat: int) -> None:
        try:
            output = _docker(["logs", container_name], timeout=20)
        except RuntimeError as error:
            output = str(error)
        path = self.logs_root / f"{mode}-{count}-{repeat}.app.log"
        path.write_text(output + "\n", encoding="utf-8")

    def _experiment_metadata(self) -> dict[str, Any]:
        try:
            docker_version = _docker(["version", "--format", "{{.Server.Version}}"])
        except RuntimeError as error:
            docker_version = str(error)
        return {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "repository_head_at_start": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
            ).stdout.strip(),
            "os": platform.platform(),
            "python": sys.version,
            "python_version_app_image": self.app_python_version,
            "dependencies": _dependency_versions(),
            "docker_server_version": docker_version,
            "postgresql_server_version": self.database_version,
            "alembic_head": self.alembic_head,
            "implementation_source_paths": self.implementation_source_paths,
            "images": {
                _POSTGRES_IMAGE: _docker_image_id(_POSTGRES_IMAGE),
                _APP_IMAGE: _docker_image_id(_APP_IMAGE),
            },
            "database": {
                "name": self.database_name,
                "data_storage": "task-owned database container tmpfs, capped at 256MiB",
                "host_publish": "none; migration and seed utilities share the private network",
                "container_network": f"internal Docker network {self.network_name}",
            },
            "resource_limits": {
                "database": {"cpus": 1, "memory_mib": 384, "swap_mib": 384, "tmpfs_mib": 256},
                "application": {"cpus": 1, "memory_mib": 256, "swap_mib": 256},
                "client": {"cpus": 1, "memory_mib": 256, "swap_mib": 256},
                "aggregate_database_connections": "<=4; setup utility is sequential and app pool caps at three",
            },
            "composition": "real FastAPI session and export routers, real AuthService and Database; no media services",
            "authentication": "synthetic operator credential; actual HTTP login and cookie guard used",
            "download": "stdlib HTTP client streams response to a temporary file; no response.read() for full body",
            "rss": "application resource.getrusage(RUSAGE_SELF).ru_maxrss in Linux KiB; baseline is marked after login",
        }

    def _write_command_log(self) -> None:
        command_lines = [
            "Reproduction command (host):",
            "PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python qa/events/benchmark_export.py run",
            "",
            "Exact per-run Docker/ migration commands, with synthetic passwords and control tokens redacted:",
            *self.commands,
            "",
            "Measurements run sequentially; each app/client container is task-owned, resource-limited, and removed.",
        ]
        (self.output_root / "COMMANDS.md").write_text("\n".join(command_lines) + "\n", encoding="utf-8")

    def _save_results(self) -> None:
        metadata_payload = self._experiment_metadata()
        _write_json(self.output_root / "environment.json", metadata_payload)
        _write_json(
            self.output_root / "results.json",
            {
                "environment": metadata_payload,
                "smoke": self.smoke,
                "samples": self.samples,
                "sample_summary": summarize_samples_by_dataset(self.samples),
                "sample_summary_pooled_descriptive_only": summarize_samples(self.samples),
                "isolation_trials": self.isolation_trials,
                "isolation_summary": _summarize_isolation(self.isolation_trials),
                "failures": self.failures,
                "caveats": [
                    "Three repetitions support median and range only; no p95 or significance claim.",
                    "The reduced app composition excludes media services and workers.",
                    "Startup and Argon2 login contribute to process ru_maxrss; report absolute peak and marked baseline.",
                    "HTTP overlap and app-side stream-pull intervals are separate; stream-pull spans include DB fetch and CSV encoding.",
                    "A single SELECT uses one Read Committed snapshot; the RC/RR multi-SELECT probe is a separate experiment.",
                ],
            },
        )
        self._write_command_log()

    def _run_smoke(self) -> None:
        for mode in ("streaming", "buffered"):
            expected_path = self.scratch_root / "expected-smoke.json"
            self._seed_via_utility(
                row_count=100,
                expected_path=expected_path,
                suffix=f"smoke-seed-{mode}",
            )
            result = self._run_http_sample(
                mode=mode,
                row_count=100,
                repeat=0,
                writer_enabled=False,
                expected_path=expected_path,
            )
            result["category"] = "smoke"
            self.smoke.append(result)
        hashes = [sample.get("body_sha256") for sample in self.smoke]
        equal = hashes[0] == hashes[1]
        self.smoke[-1]["identical_csv_to_other_mode"] = equal
        if not all(sample.get("correctness") is True for sample in self.smoke) or not equal:
            raise RuntimeError("streaming/buffered smoke outputs differed or failed CSV validation")

    def _run_exports(self) -> None:
        for row_count in _SAMPLE_COUNTS:
            for repeat in range(1, _REPEATS + 1):
                modes = ("streaming", "buffered") if repeat % 2 == 1 else ("buffered", "streaming")
                for mode in modes:
                    expected_path = self.scratch_root / f"expected-{row_count}.json"
                    self._seed_via_utility(
                        row_count=row_count,
                        expected_path=expected_path,
                        suffix=f"seed-{mode}-{row_count}-{repeat}",
                    )
                    result = self._run_http_sample(
                        mode=mode,
                        row_count=row_count,
                        repeat=repeat,
                        writer_enabled=True,
                        expected_path=expected_path,
                    )
                    self.samples.append(result)
                    self._save_results()

    def _run_isolation(self) -> None:
        for row_count in _SAMPLE_COUNTS:
            for isolation in ("READ COMMITTED", "REPEATABLE READ"):
                for repeat in range(1, _REPEATS + 1):
                    expected_path = self.scratch_root / f"expected-isolation-{row_count}.json"
                    result_path = self.scratch_root / f"isolation-{row_count}-{repeat}.json"
                    self._run_utility(
                        suffix=f"isolation-{row_count}-{isolation.lower().replace(' ', '-')}-{repeat}",
                        command=(
                            "isolation-sample",
                            "--row-count",
                            str(row_count),
                            "--isolation",
                            isolation,
                            "--repeat",
                            str(repeat),
                            "--expected",
                            f"/work/results/{expected_path.name}",
                            "--result",
                            f"/work/results/{result_path.name}",
                        ),
                        timeout=300,
                    )
                    trial = json.loads(result_path.read_text(encoding="utf-8"))
                    self.isolation_trials.append(trial)
                    self._save_results()

    def resume_missing_isolation_trials(self) -> int:
        """Run only absent isolation coordinates from the already completed HTTP run."""
        results_path = self.output_root / "results.json"
        original = json.loads(results_path.read_text(encoding="utf-8"))
        original_trials = list(original.get("isolation_trials", []))
        if len(original.get("samples", [])) != 12 or len(original_trials) != 7:
            raise RuntimeError("resume requires the recorded 12 HTTP samples and 7 isolation trials")
        if original.get("isolation_recovery"):
            raise RuntimeError("isolation recovery is already recorded")

        expected = [
            (size, isolation, repeat)
            for size in _SAMPLE_COUNTS
            for isolation in ("READ COMMITTED", "REPEATABLE READ")
            for repeat in range(1, _REPEATS + 1)
        ]
        completed = {
            (trial.get("dataset_size"), trial.get("isolation"), trial.get("repeat"))
            for trial in original_trials
        }
        missing = [coordinate for coordinate in expected if coordinate not in completed]
        expected_missing = [
            (50_000, "READ COMMITTED", 2),
            (50_000, "READ COMMITTED", 3),
            (50_000, "REPEATABLE READ", 1),
            (50_000, "REPEATABLE READ", 2),
            (50_000, "REPEATABLE READ", 3),
        ]
        if missing != expected_missing:
            raise RuntimeError(f"unexpected missing isolation coordinates: {missing!r}")

        recovery_path = self.output_root / f"isolation-recovery-{self.run_id}.json"
        log_path = self.logs_root / f"isolation-recovery-{self.run_id}-postgres.log"
        self.isolation_trials = []

        def save_recovery(status: str) -> None:
            _write_json(
                recovery_path,
                {
                    "status": status,
                    "run_id": self.run_id,
                    "started_at_utc": self.started_at_utc,
                    "environment": self._experiment_metadata(),
                    "requested_coordinates": [list(value) for value in missing],
                    "trials": self.isolation_trials,
                    "failures": self.failures,
                    "resource_diagnostics": self.resource_diagnostics,
                },
            )

        exit_code = 1
        try:
            self.current_phase = "create isolated recovery DB and apply migrations"
            self._start_environment()
            self.resource_diagnostics.append(self._database_diagnostics("after-migration"))
            for row_count, isolation, repeat in missing:
                label = f"{row_count}-{isolation.lower().replace(' ', '-')}-{repeat}"
                self.current_phase = f"recovery isolation trial {label}"
                self.resource_diagnostics.append(self._database_diagnostics(f"before-{label}"))
                expected_path = self.scratch_root / f"expected-{label}.json"
                result_path = self.scratch_root / f"result-{label}.json"
                try:
                    self._run_utility(
                        suffix=f"recovery-{label}",
                        command=(
                            "isolation-sample",
                            "--row-count",
                            str(row_count),
                            "--isolation",
                            isolation,
                            "--repeat",
                            str(repeat),
                            "--expected",
                            f"/work/results/{expected_path.name}",
                            "--result",
                            f"/work/results/{result_path.name}",
                        ),
                        timeout=300,
                    )
                    trial = json.loads(result_path.read_text(encoding="utf-8"))
                    self.isolation_trials.append(trial)
                    self.resource_diagnostics.append(self._database_diagnostics(f"after-{label}"))
                    save_recovery("in_progress")
                    if trial.get("error") is not None or trial.get("exact_ids_match") is not True:
                        raise RuntimeError(f"isolation trial failed validation: {label}")
                except Exception as error:
                    self.failures.append(
                        {
                            "phase": self.current_phase,
                            "error_type": type(error).__name__,
                            "message": str(error)[:2_000],
                        }
                    )
                    self.resource_diagnostics.append(self._database_diagnostics(f"failure-{label}"))
                    save_recovery("failed")
                    break
            if len(self.isolation_trials) == len(missing) and not self.failures:
                exit_code = 0
            save_recovery("completed" if exit_code == 0 else "failed")
        except Exception as error:
            self.failures.append(
                {
                    "phase": self.current_phase,
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                }
            )
            self.resource_diagnostics.append(self._database_diagnostics("setup-failure"))
            save_recovery("failed")
        finally:
            if self.db_container in self.owned_containers:
                try:
                    log_path.write_text(
                        _docker(["logs", self.db_container], timeout=20) + "\n",
                        encoding="utf-8",
                    )
                except RuntimeError as error:
                    log_path.write_text(f"{error}\n", encoding="utf-8")
            for container_name in tuple(self.owned_containers):
                try:
                    self._run_docker(["rm", "--force", container_name], timeout=30)
                except RuntimeError:
                    pass
                self.owned_containers.discard(container_name)
            if self.network_created:
                try:
                    self._run_docker(["network", "rm", self.network_name], timeout=30)
                except RuntimeError:
                    pass
                self.network_created = False
            self._write_recovery_commands()
            save_recovery("completed" if exit_code == 0 else "failed")
            shutil.rmtree(self.scratch_root, ignore_errors=True)

        if exit_code != 0:
            return exit_code

        initial_path = self.output_root / "initial-run-results.json"
        if not initial_path.exists():
            shutil.copy2(results_path, initial_path)
        merged = dict(original)
        merged["sample_summary_pooled_descriptive_only"] = merged.pop("sample_summary", {})
        merged["sample_summary"] = summarize_samples_by_dataset(merged["samples"])
        merged["sample_summary_by_dataset_and_mode"] = merged["sample_summary"]
        merged["isolation_trials"] = original_trials + self.isolation_trials
        merged["isolation_summary"] = _summarize_isolation(merged["isolation_trials"])
        merged["isolation_recovery"] = {
            "run_id": self.run_id,
            "status": "completed",
            "source_run_id": original.get("environment", {}).get("run_id"),
            "resumed_trial_count": len(self.isolation_trials),
            "total_isolation_trial_count": len(merged["isolation_trials"]),
            "raw_artifact": recovery_path.name,
            "initial_run_archive": initial_path.name,
            "postgres_log": log_path.name,
            "diagnostics_are_outside_trial_timing": True,
        }
        results_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0

    def _write_recovery_commands(self) -> None:
        path = self.output_root / f"isolation-recovery-{self.run_id}-commands.md"
        path.write_text(
            "Recovery command: `PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python "
            "qa/events/benchmark_export.py resume-isolation`\n\n"
            "Exact per-run Docker commands, with synthetic credentials redacted:\n\n"
            + "\n".join(f"- `{command}`" for command in self.commands)
            + "\n",
            encoding="utf-8",
        )

    def _seed_via_utility(self, *, row_count: int, expected_path: Path, suffix: str) -> None:
        self._run_utility(
            suffix=suffix,
            command=(
                "seed",
                "--row-count",
                str(row_count),
                "--expected",
                f"/work/results/{expected_path.name}",
            ),
            timeout=300,
        )

    def run(self, *, smoke_only: bool = False) -> int:
        try:
            self.current_phase = "create disposable DB and apply migrations"
            self._start_environment()
            self.current_phase = "100-row authenticated HTTP smoke"
            self._run_smoke()
            self._save_results()
            if not smoke_only:
                self.current_phase = "12 HTTP export samples"
                self._run_exports()
                self.current_phase = "12 RC/RR isolation trials"
                self._run_isolation()
            self._save_results()
            if not all(sample.get("correctness") is True for sample in self.samples):
                return 2
            if not smoke_only and (len(self.samples) != 12 or len(self.isolation_trials) != 12):
                return 3
            return 0
        except Exception as error:
            self.failures.append(
                {
                    "phase": self.current_phase,
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                }
            )
            raise
        finally:
            self._collect_postgres_log()
            for container_name in tuple(self.owned_containers):
                try:
                    self._run_docker(["rm", "--force", container_name], timeout=30)
                except RuntimeError:
                    pass
                self.owned_containers.discard(container_name)
            if self.network_created:
                try:
                    self._run_docker(["network", "rm", self.network_name], timeout=30)
                except RuntimeError:
                    pass
                self.network_created = False
            self._save_results()
            shutil.rmtree(self.scratch_root, ignore_errors=True)

    def _collect_postgres_log(self) -> None:
        if self.db_container not in self.owned_containers:
            return
        try:
            output = _docker(["logs", self.db_container], timeout=20)
        except RuntimeError as error:
            output = str(error)
        (self.logs_root / "postgres.log").write_text(output + "\n", encoding="utf-8")


def _summarize_isolation(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for size in _SAMPLE_COUNTS:
        for isolation in ("READ COMMITTED", "REPEATABLE READ"):
            selected = [
                trial
                for trial in trials
                if trial.get("dataset_size") == size and trial.get("isolation") == isolation
            ]
            elapsed = [float(trial["elapsed_seconds"]) for trial in selected if trial.get("error") is None]
            summary[f"{size}:{isolation}"] = {
                "trial_count": len(selected),
                "initial_count": selected[0].get("initial_count") if selected else None,
                "export_counts": [trial.get("export_count") for trial in selected],
                "count_mismatches": [trial.get("count_mismatch") for trial in selected],
                "elapsed_seconds_median": statistics.median(elapsed) if elapsed else None,
                "elapsed_seconds_range": [min(elapsed), max(elapsed)] if elapsed else None,
            }
    return summary


async def _run_isolation_trial(
    database: Database,
    *,
    row_count: int,
    expected_ids: Sequence[int],
    isolation: str,
    repeat: int,
) -> dict[str, Any]:
    start = time.monotonic()
    trial: dict[str, Any] = {
        "dataset_size": row_count,
        "isolation": isolation,
        "repeat": repeat,
        "forced_interleaving": "initial COUNT, commit writer batch, then materialize repository projection in the same transaction",
        "expected_count": row_count,
        "error": None,
    }
    try:
        async with database.engine.connect() as connection:
            connection = await connection.execution_options(isolation_level=isolation)
            transaction = await connection.begin()
            try:
                initial = await connection.scalar(select(func.count()).select_from(CameraEvent))
                writer = await _writer_insert(database, offset=(row_count * repeat))
                projected = await connection.execute(EventRepository.statement(EventExportFilters()))
                exported = projected.all()
                exported_ids = [int(row[0]) for row in exported]
                await transaction.commit()
            except BaseException:
                if transaction.is_active:
                    await transaction.rollback()
                raise
        expected_exported = list(expected_ids)
        if isolation == "READ COMMITTED":
            expected_exported.extend(int(identifier) for identifier in writer["ids"])
        trial.update(
            {
                "initial_count": int(initial or 0),
                "writer_count": int(writer["count"]),
                "writer_started_at_utc": writer["started_at_utc"],
                "writer_committed_at_utc": writer["committed_at_utc"],
                "writer_ended_at_utc": writer["ended_at_utc"],
                "writer_commit_seconds": writer["commit_seconds"],
                "export_count": len(exported_ids),
                "count_mismatch": len(exported_ids) != int(initial or 0),
                "expected_export_count": len(expected_exported),
                "exact_ids_match": exported_ids == expected_exported,
                "exported_ids_sha256": _id_digest(exported_ids),
                "expected_ids_sha256": _id_digest(expected_exported),
            }
        )
    except Exception as error:
        trial["error"] = type(error).__name__
    trial["elapsed_seconds"] = time.monotonic() - start
    return trial


def _utility_main(arguments: argparse.Namespace) -> int:
    database_url = os.environ["GW_DATABASE_URL"]
    database_name = os.environ["GW_BENCHMARK_DATABASE_NAME"]
    ensure_disposable_database(database_url, expected_database=database_name)
    database = _make_engine(database_url)

    async def run_seed() -> None:
        await _seed_database(
            database,
            row_count=arguments.row_count,
            expected_path=Path(arguments.expected),
        )

    async def run_isolation() -> None:
        expected = await _seed_database(
            database,
            row_count=arguments.row_count,
            expected_path=Path(arguments.expected),
        )
        trial = await _run_isolation_trial(
            database,
            row_count=arguments.row_count,
            expected_ids=tuple(sorted(expected)),
            isolation=arguments.isolation,
            repeat=arguments.repeat,
        )
        _write_json(Path(arguments.result), trial)

    async def execute() -> None:
        try:
            if arguments.command == "seed":
                await run_seed()
            else:
                await run_isolation()
        finally:
            await database.close()

    asyncio.run(execute())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="run smoke, export samples, and isolation probe")
    run_parser.add_argument("--smoke-only", action="store_true")
    subparsers.add_parser("resume-isolation", help="run only the five absent isolation trials")
    client_parser = subparsers.add_parser("client-sample", help=argparse.SUPPRESS)
    client_parser.add_argument("--app-host", required=True)
    client_parser.add_argument("--expected", required=True)
    client_parser.add_argument("--output-csv", required=True)
    client_parser.add_argument("--result", required=True)
    client_parser.add_argument("--writer", action="store_true")
    seed_parser = subparsers.add_parser("seed", help=argparse.SUPPRESS)
    seed_parser.add_argument("--row-count", required=True, type=int)
    seed_parser.add_argument("--expected", required=True)
    isolation_parser = subparsers.add_parser("isolation-sample", help=argparse.SUPPRESS)
    isolation_parser.add_argument("--row-count", required=True, type=int)
    isolation_parser.add_argument("--isolation", required=True, choices=("READ COMMITTED", "REPEATABLE READ"))
    isolation_parser.add_argument("--repeat", required=True, type=int)
    isolation_parser.add_argument("--expected", required=True)
    isolation_parser.add_argument("--result", required=True)
    arguments = parser.parse_args()
    if arguments.command == "client-sample":
        return _client_main(arguments)
    if arguments.command in {"seed", "isolation-sample"}:
        return _utility_main(arguments)
    if arguments.command == "resume-isolation":
        return BenchmarkRunner().resume_missing_isolation_trials()
    return BenchmarkRunner().run(smoke_only=bool(arguments.smoke_only))


if __name__ == "__main__":
    raise SystemExit(main())
