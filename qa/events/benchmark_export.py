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
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from http.cookies import SimpleCookie
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, cast
from uuid import UUID, uuid4

from sqlalchemy import func, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

try:
    from ._benchmark_export_bootstrap import REPO_ROOT, SOURCE_ROOT
except ImportError:
    from _benchmark_export_bootstrap import REPO_ROOT, SOURCE_ROOT

from gods_watching.contracts.events import EventExportFilters
from gods_watching.events.repository import EventRepository
from gods_watching.storage import CameraEvent, Database

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from typing import BinaryIO

_CSV_FIELDS = ("id", "occurred_at", "event_type", "camera_id", "camera_name")
_POSTGRES_IMAGE = "pgvector/pgvector:0.8.1-pg17"
_APP_IMAGE = "gods-watching-app:local"
_SAMPLE_COUNTS = (10_000, 50_000)
_REPEATS = 3
_WRITER_COUNT = 100
_HTTP_OK_STATUS = 200
_EXPECTED_SAMPLE_COUNT = 12
_EXPECTED_ISOLATION_TRIAL_COUNT = 7
_FIRST_WRITER_CHUNK_INDEX = 2
_FORMULA_PREFIXES = frozenset("=+-@")
_CONTROL_PREFIXES = frozenset(("\t", "\r", "\n"))


@dataclass(frozen=True, slots=True)
class ControlClientConfig:
    """Connection details shared by requests to the benchmark control API."""

    app_host: str
    control_token: str


@dataclass(frozen=True, slots=True)
class _ControlRequest:
    """Request-specific values sent through the benchmark control client."""

    method: str
    path: str
    payload: Mapping[str, object] | None = None
    cookie: str | None = None
    timeout: float = 30.0


@dataclass(frozen=True, slots=True)
class _SamplePaths:
    """Paths belonging to one downloaded sample."""

    expected: Path
    output_csv: Path
    result: Path


@dataclass(frozen=True, slots=True)
class SampleCoordinate:
    """Immutable options that identify one sample and its writer behavior."""

    mode: str
    row_count: int
    repeat: int
    writer_enabled: bool


@dataclass(slots=True)
class _WriterObservation:
    """Client-side status for the asynchronous writer request."""

    triggered_at: float | None = None
    request: dict[str, object] = dataclass_field(
        default_factory=lambda: {"requested": False}
    )
    errors: list[str] = dataclass_field(default_factory=list)
    thread: threading.Thread | None = None


@dataclass(frozen=True, slots=True)
class _WriterLaunch:
    """Inputs needed to start the writer at the CSV first-row boundary."""

    client: ControlClientConfig
    enabled: bool
    observation: _WriterObservation


@dataclass(frozen=True, slots=True)
class _BodyDownload:
    """Measured body transfer and writer observation for one HTTP response."""

    first_byte_latency: float | None
    byte_count: int
    body_sha256: str
    writer: _WriterObservation


@dataclass(frozen=True, slots=True)
class _SampleTransfer:
    """HTTP request and file-transfer clock boundaries plus its streamed body."""

    request_started: float
    transfer_ended: float
    body: _BodyDownload


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


class _ReadableResponse(Protocol):
    """Minimal readable-response contract shared by HTTP and pure test streams."""

    def read(self, amt: int | None = None) -> bytes:
        """Read up to amt bytes, or to end when amt is omitted."""
        ...


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
                    if None in row or any(row.get(field) is None for field in _CSV_FIELDS):
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
        exception_message = "target must be a disposable benchmark database"
        raise ValueError(exception_message) from error
    database = url.database or ""
    if (
        url.drivername != "postgresql+asyncpg"
        or bool(url.query)
        or url.host not in {"127.0.0.1", "localhost", "db"}
        or url.username != "postgres"
        or re.fullmatch(r"gw_events_bench_[0-9a-f]{8}", database) is None
        or (expected_database is not None and database != expected_database)
    ):
        exception_message = "target must be a disposable benchmark database"
        raise ValueError(exception_message)


def summarize_samples(
    samples: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
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
    samples: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize only correctness-eligible samples within each dataset/mode cell."""
    summary: dict[str, dict[str, object]] = {}
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
            cell: dict[str, object] = {"sample_count": len(selected)}
            for label, field in metrics.items():
                values: list[float] = []
                for sample in selected:
                    metric_value = sample.get(field)
                    if isinstance(metric_value, (int, float)):
                        values.append(float(metric_value))
                cell[f"{label}_median"] = statistics.median(values) if values else None
                cell[f"{label}_range"] = [min(values), max(values)] if values else None
            summary[f"{size}:{mode}"] = cell
    return summary


def _safe_export_name(value: str) -> str:
    stripped = value.lstrip()
    if value and (value[0] in _CONTROL_PREFIXES or (stripped and stripped[0] in _FORMULA_PREFIXES)):
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


def _event_json(row: _ProjectionRow) -> dict[str, str]:
    timestamp = row.occurred_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    return {
        "id": str(row.id),
        "occurred_at": timestamp,
        "event_type": row.event_type,
        "camera_id": str(row.camera_id),
        "camera_name": _safe_export_name(row.camera_name),
    }


def _write_json(path: Path, payload: object) -> None:
    _ = path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_json_object(value: object) -> dict[str, object]:
    """Validate a JSON object and narrow dynamic keys to strings."""
    if not isinstance(value, dict):
        exception_message = "benchmark JSON value must be an object"
        raise TypeError(exception_message)
    result: dict[str, object] = {}
    for key, item in cast("dict[object, object]", value).items():
        if not isinstance(key, str):
            exception_message = "benchmark JSON object keys must be strings"
            raise TypeError(exception_message)
        result[key] = item
    return result


def load_json_object(payload: str | bytes) -> dict[str, object]:
    """Decode JSON behind an object boundary and validate its top-level shape."""
    decoded: object = cast("object", json.loads(payload))
    return parse_json_object(decoded)


def read_json_object(path: Path) -> dict[str, object]:
    """Read and validate a JSON object from a benchmark artifact."""
    return load_json_object(path.read_text(encoding="utf-8"))


def parse_json_array(payload: str | bytes) -> list[object]:
    """Decode a JSON array while containing the stdlib decoder's dynamic type."""
    decoded: object = cast("object", json.loads(payload))
    if not isinstance(decoded, list):
        exception_message = "benchmark JSON value must be an array"
        raise TypeError(exception_message)
    return cast("list[object]", decoded)


def _json_object_list(value: object) -> list[dict[str, object]]:
    """Validate a JSON list whose entries are records, without requiring fields."""
    if not isinstance(value, list):
        exception_message = "benchmark JSON value must be an array of objects"
        raise TypeError(exception_message)
    return [parse_json_object(item) for item in cast("list[object]", value)]


def _json_int(value: object) -> int:
    """Convert the scalar values accepted by int() from validated JSON."""
    if isinstance(value, (int, float, str)):
        return int(value)
    exception_message = "benchmark JSON value is not an integer scalar"
    raise TypeError(exception_message)


def _subprocess_text(value: object) -> str:
    """Narrow captured text streams from subprocess exceptions."""
    return value if isinstance(value, str) else ""


def _resolve_executable(name: str) -> str:
    """Resolve host-side command line tools only when a host helper needs them."""
    executable = shutil.which(name)
    if executable is None:
        exception_message = f"required executable {name!r} was not found on PATH"
        raise FileNotFoundError(exception_message)
    return str(Path(executable).resolve())


def _raise_invalid_isolation_trial(label: str) -> NoReturn:
    exception_message = f"isolation trial failed validation: {label}"
    raise RuntimeError(exception_message)


def _raise_invalid_writer_ids() -> NoReturn:
    exception_message = "writer result did not include its committed event IDs"
    raise TypeError(exception_message)


def docker_command(arguments: Sequence[str], *, timeout: int = 60) -> str:
    """Run a controlled Docker CLI command and return its standard output."""
    try:
        completed = subprocess.run(  # noqa: S603 - bounded, shell-free Docker argument vector.
            [_resolve_executable("docker"), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as error:
        detail = (
            _subprocess_text(cast("object", error.stderr)).strip()
            or f"exit code {error.returncode}"
        )
        exception_message = f"docker {arguments[0]} failed: {detail}"
        raise RuntimeError(exception_message) from None
    return completed.stdout.strip()


def docker_logs(container_name: str, *, tail: int | None = None) -> str:
    """Return labeled stdout and stderr from Docker's container log streams."""
    arguments = [_resolve_executable("docker"), "logs"]
    if tail is not None:
        arguments.extend(["--tail", str(tail)])
    arguments.append(container_name)
    try:
        completed = subprocess.run(  # noqa: S603 - bounded, shell-free Docker log argument vector.
            arguments,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.CalledProcessError as error:
        stdout = _subprocess_text(cast("object", error.stdout))
        stderr = _subprocess_text(cast("object", error.stderr))
        stdout_log = f"--- stdout ---\n{stdout}"
        stderr_log = f"--- stderr ---\n{stderr}"
        exception_message = f"docker logs failed:\n{stdout_log}\n{stderr_log}"
        raise RuntimeError(exception_message) from None
    return "\n".join(
        (
            "--- stdout ---",
            completed.stdout or "",
            "--- stderr ---",
            completed.stderr or "",
        )
    )


def writer_overlap_metrics(
    *,
    http_start: float,
    http_end: float,
    writer_start: float,
    writer_commit: float,
) -> dict[str, bool]:
    """Separate transaction-activity intersection from commit inside the HTTP window."""
    return {
        "transaction_activity_overlaps_http_transfer": (
            writer_start <= http_end and writer_commit >= http_start
        ),
        "commit_inside_http_transfer": http_start <= writer_commit <= http_end,
    }


def isolation_trial_is_valid(trial: Mapping[str, object]) -> bool:
    """Validate a forced-interleaving RC/RR result before timing can be summarized."""
    dataset_size = trial.get("dataset_size")
    isolation = trial.get("isolation")
    if not isinstance(dataset_size, int) or dataset_size <= 0:
        return False
    if isolation not in {"READ COMMITTED", "REPEATABLE READ"}:
        return False
    expected_export_count = dataset_size + (_WRITER_COUNT if isolation == "READ COMMITTED" else 0)
    return (
        trial.get("error") is None
        and trial.get("exact_ids_match") is True
        and trial.get("initial_count") == dataset_size
        and trial.get("export_count") == expected_export_count
        and trial.get("writer_count") == _WRITER_COUNT
    )


def _dependency_versions() -> dict[str, str]:
    names = ("alembic", "anyio", "argon2-cffi", "asyncpg", "fastapi", "SQLAlchemy", "uvicorn")
    return {name: metadata.version(name) for name in names}


def _docker_image_id(image: str) -> str:
    return docker_command(["image", "inspect", image, "--format", "{{.Id}}"])


def _write_expected_file(
    path: Path,
    rows: Sequence[_ProjectionRow],
) -> dict[int, dict[str, str]]:
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
        projected = result.tuples().all()
    expected = [
        _ProjectionRow(
            id=row[0],
            occurred_at=row[1],
            event_type=row[2],
            camera_id=row[3],
            camera_name=row[4],
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


async def write_benchmark_batch(database: Database, *, offset: int = 0) -> dict[str, object]:
    """Commit a fixed event batch through the production repository."""
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


def make_engine(database_url: str) -> Database:
    """Create an async database wrapper after validating its disposable name."""
    ensure_disposable_database(database_url)
    engine = create_async_engine(database_url, poolclass=NullPool, pool_pre_ping=True)
    return Database(engine, async_sessionmaker(engine, expire_on_commit=False))


def _client_json_request(
    client: ControlClientConfig,
    request: _ControlRequest,
) -> tuple[int, dict[str, object], http.client.HTTPMessage]:
    connection = http.client.HTTPConnection(client.app_host, 8000, timeout=request.timeout)
    headers = {"X-Benchmark-Control": client.control_token}
    body = None
    if request.payload is not None:
        body = json.dumps(request.payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if request.cookie is not None:
        headers["Cookie"] = request.cookie
    try:
        connection.request(request.method, request.path, body=body, headers=headers)
        response = connection.getresponse()
        response_payload = response.read()
        if response_payload:
            try:
                decoded = load_json_object(response_payload)
            except ValueError as error:
                exception_message = "benchmark control endpoint returned an invalid response"
                raise RuntimeError(exception_message) from error
        else:
            decoded = {}
        return response.status, decoded, response.headers
    finally:
        connection.close()


def _load_expected_rows(path: Path) -> dict[int, dict[str, str]]:
    payload = read_json_object(path)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        exception_message = "expected-row manifest is invalid"
        raise TypeError(exception_message)
    expected: dict[int, dict[str, str]] = {}
    for raw_row in cast("list[object]", rows):
        try:
            row = parse_json_object(raw_row)
        except (TypeError, ValueError) as error:
            exception_message = "expected-row manifest is invalid"
            raise RuntimeError(exception_message) from error
        expected[_json_int(row["id"])] = {field: str(value) for field, value in row.items()}
    return expected


def _wait_for_app(client: ControlClientConfig) -> None:
    deadline = time.monotonic() + 60.0
    last_error = "not ready"
    while time.monotonic() < deadline:
        try:
            status, _, _ = _client_json_request(
                client,
                _ControlRequest("GET", "/__benchmark/ready", timeout=2.0),
            )
            if status == _HTTP_OK_STATUS:
                return
            last_error = f"HTTP {status}"
        except OSError as error:
            last_error = type(error).__name__
        time.sleep(0.2)
    exception_message = f"benchmark app was not ready within 60 seconds ({last_error})"
    raise RuntimeError(exception_message)


def _authenticate_operator(
    client: ControlClientConfig,
    password: str,
) -> tuple[str, dict[str, object]]:
    """Validate the real operator login and capture the application RSS baseline."""
    _wait_for_app(client)
    login_connection = http.client.HTTPConnection(client.app_host, 8000, timeout=30.0)
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
    if login_response.status != _HTTP_OK_STATUS or cookie_header is None:
        exception_message = f"real operator login failed with HTTP {login_response.status}"
        raise RuntimeError(exception_message)
    cookies = SimpleCookie()
    cookies.load(cookie_header)
    session_cookie = cookies.get("gw_session")
    if session_cookie is None:
        exception_message = "real operator login did not set its session cookie"
        raise RuntimeError(exception_message)
    login_payload = load_json_object(login_body)
    if login_payload.get("authenticated") is not True:
        exception_message = "real operator login response was not authenticated"
        raise RuntimeError(exception_message)

    baseline_status, baseline, _ = _client_json_request(
        client,
        _ControlRequest("POST", "/__benchmark/baseline"),
    )
    if baseline_status != _HTTP_OK_STATUS:
        exception_message = "application RSS baseline was unavailable"
        raise RuntimeError(exception_message)
    return session_cookie.value, baseline


def _perform_writer_request(client: ControlClientConfig, writer: _WriterObservation) -> None:
    """Run one asynchronous writer request and retain its outcome for the sample."""
    writer.request["client_started_monotonic"] = time.monotonic()
    try:
        status, payload, _ = _client_json_request(
            client,
            _ControlRequest("POST", "/__benchmark/writer", timeout=120.0),
        )
        writer.request["http_status"] = status
        writer.request["response_count"] = payload.get("count", 0)
    except Exception as error:  # noqa: BLE001 - capture any background writer failure for this owner.
        writer.errors.append(type(error).__name__)
    writer.request["client_ended_monotonic"] = time.monotonic()
    writer.request["requested"] = True


def _launch_writer(client: ControlClientConfig, writer: _WriterObservation) -> None:
    """Start the writer immediately after the first complete CSV data row."""
    writer.triggered_at = time.monotonic()
    writer.thread = threading.Thread(
        target=_perform_writer_request,
        args=(client, writer),
        name="event-benchmark-writer",
    )
    writer.thread.start()


def _read_csv_preamble(
    response: _ReadableResponse,
    output: BinaryIO,
    request_started: float,
    launch: _WriterLaunch,
) -> tuple[bytes, float | None]:
    """Write the header and first data row one byte at a time, as before."""
    prefix = bytearray()
    first_byte_latency: float | None = None
    for line_number in range(2):
        while True:
            byte = response.read(1)
            if not byte:
                exception_message = "HTTP export ended before its first CSV data row"
                raise RuntimeError(exception_message)
            if first_byte_latency is None:
                first_byte_latency = time.monotonic() - request_started
            _ = output.write(byte)
            prefix.extend(byte)
            if byte == b"\n":
                break
        if line_number == 1 and launch.enabled:
            _launch_writer(launch.client, launch.observation)
    return bytes(prefix), first_byte_latency


def download_export_body(
    response: _ReadableResponse,
    output: BinaryIO,
    request_started: float,
    client: ControlClientConfig,
    *,
    writer_enabled: bool,
) -> _BodyDownload:
    """Stream 64 KiB body chunks, preserving the first-row writer trigger and clocks."""
    writer = _WriterObservation()
    prefix, first_byte_latency = _read_csv_preamble(
        response,
        output,
        request_started,
        _WriterLaunch(client=client, enabled=writer_enabled, observation=writer),
    )
    body_hash = hashlib.sha256()
    body_hash.update(prefix)
    byte_count = len(prefix)
    while True:
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        if first_byte_latency is None:
            first_byte_latency = time.monotonic() - request_started
        _ = output.write(chunk)
        body_hash.update(chunk)
        byte_count += len(chunk)
    return _BodyDownload(
        first_byte_latency=first_byte_latency,
        byte_count=byte_count,
        body_sha256=body_hash.hexdigest(),
        writer=writer,
    )


def download_export_to_file(
    output_path: Path,
    response: _ReadableResponse,
    request_started: float,
    client: ControlClientConfig,
    *,
    writer_enabled: bool,
) -> _SampleTransfer:
    """Close the CSV file before recording the transfer end, as the sample did."""
    with output_path.open("wb") as output:
        body = download_export_body(
            response,
            output,
            request_started,
            client,
            writer_enabled=writer_enabled,
        )
    transfer_ended = time.monotonic()
    return _SampleTransfer(request_started, transfer_ended, body)


def join_writer(writer: _WriterObservation) -> None:
    """Wait for the optional writer request and record its existing timeout outcome."""
    if writer.thread is None:
        return
    writer.thread.join(timeout=120.0)
    if writer.thread.is_alive():
        writer.errors.append("writer_timeout")


def run_client_sample(
    client: ControlClientConfig,
    password: str,
    paths: _SamplePaths,
    writer_enabled: bool,
) -> dict[str, object]:
    """Login, stream the HTTP body to disk, trigger one asynchronous writer, and validate."""
    session_cookie, baseline = _authenticate_operator(client, password)

    connection = http.client.HTTPConnection(client.app_host, 8000, timeout=120.0)
    request_started = time.monotonic()
    connection.request(
        "GET",
        "/api/events/export.csv",
        headers={"Cookie": f"gw_session={session_cookie}"},
    )
    response = connection.getresponse()
    if response.status != _HTTP_OK_STATUS:
        error_body = response.read(4_096).decode("utf-8", errors="replace")
        connection.close()
        exception_message = f"HTTP export failed with status {response.status}: {error_body}"
        raise RuntimeError(exception_message)
    if response.getheader("Content-Type", "").split(";", maxsplit=1)[0] != "text/csv":
        connection.close()
        exception_message = "HTTP export content type was not text/csv"
        raise RuntimeError(exception_message)

    paths.output_csv.parent.mkdir(parents=True, exist_ok=True)
    try:
        download = download_export_to_file(
            paths.output_csv,
            response,
            request_started,
            client,
            writer_enabled=writer_enabled,
        )
    finally:
        connection.close()
    writer = download.body.writer
    join_writer(writer)

    metrics_status, app_metrics, _ = _client_json_request(
        client,
        _ControlRequest("GET", "/__benchmark/metrics"),
    )
    if metrics_status != _HTTP_OK_STATUS:
        exception_message = "application metrics endpoint was unavailable"
        raise RuntimeError(exception_message)
    expected_rows = _load_expected_rows(paths.expected)
    validation = validate_csv_file(paths.output_csv, expected_rows)
    writer_metrics_value = app_metrics.get("writer")
    writer_metrics = (
        parse_json_object(cast("object", writer_metrics_value))
        if isinstance(writer_metrics_value, dict)
        else None
    )
    writer_count = _json_int(writer_metrics.get("count", 0)) if writer_metrics is not None else 0
    writer_ok = (not writer_enabled) or (writer_count == _WRITER_COUNT and not writer.errors)
    result: dict[str, object] = {
        "http_request_started_monotonic": download.request_started,
        "http_transfer_ended_monotonic": download.transfer_ended,
        "latency_seconds": download.transfer_ended - download.request_started,
        "first_byte_latency_seconds": download.body.first_byte_latency,
        "bytes": download.body.byte_count,
        "body_sha256": download.body.body_sha256,
        "csv_validation": cast("object", asdict(validation)),
        "export_correctness": validation.correct,
        "writer_triggered_after_first_data_row": writer.triggered_at is not None,
        "writer_request": writer.request,
        "writer_request_errors": writer.errors,
        "writer_count": writer_count,
        "writer_correctness": writer_ok,
        "application_metrics": app_metrics,
        "startup_login_baseline_rss_kib": baseline.get("peak_rss_kib"),
    }
    result["correctness"] = validation.correct and writer_ok
    result["performance_eligible"] = validation.correct and writer_ok
    _write_json(paths.result, result)
    return result


class _CommandLineArguments(argparse.Namespace):
    """Attributes supplied by the command-specific argparse subparser."""

    command: str | None = None
    smoke_only: bool = False
    app_host: str = ""
    expected: str = ""
    output_csv: str = ""
    result: str = ""
    writer: bool = False
    row_count: int = 0
    isolation: str = ""
    repeat: int = 0


def _client_main(arguments: _CommandLineArguments) -> int:
    _ = run_client_sample(
        ControlClientConfig(
            app_host=arguments.app_host,
            control_token=os.environ["GW_BENCHMARK_CONTROL_TOKEN"],
        ),
        os.environ["GW_BENCHMARK_PASSWORD"],
        _SamplePaths(
            expected=Path(arguments.expected),
            output_csv=Path(arguments.output_csv),
            result=Path(arguments.result),
        ),
        arguments.writer,
    )
    return 0


def writer_overlap_metadata(
    result: Mapping[str, object],
    app_metrics: Mapping[str, object],
) -> dict[str, object]:
    """Calculate overlap flags while retaining unknown values for absent intervals."""
    writer_value = app_metrics.get("writer")
    writer = (
        parse_json_object(cast("object", writer_value))
        if isinstance(writer_value, dict)
        else None
    )
    writer_start_value = writer.get("started_monotonic") if writer is not None else None
    writer_commit_value = writer.get("committed_monotonic") if writer is not None else None
    writer_interval = (
        (float(writer_start_value), float(writer_commit_value))
        if isinstance(writer_start_value, (int, float))
        and isinstance(writer_commit_value, (int, float))
        else None
    )
    http_start = result.get("http_request_started_monotonic")
    http_end = result.get("http_transfer_ended_monotonic")
    transaction_activity_overlap = False
    commit_inside_http = False
    fetch_encode_overlap: bool | None = None
    if (
        writer_interval is not None
        and isinstance(http_start, (int, float))
        and isinstance(http_end, (int, float))
    ):
        writer_start, writer_commit = writer_interval
        overlap = writer_overlap_metrics(
            http_start=float(http_start),
            http_end=float(http_end),
            writer_start=writer_start,
            writer_commit=writer_commit,
        )
        transaction_activity_overlap = overlap[
            "transaction_activity_overlaps_http_transfer"
        ]
        commit_inside_http = overlap["commit_inside_http_transfer"]
        spans_value = app_metrics.get("chunk_pull_spans", [])
        fetch_spans: list[tuple[float, float]] = []
        if isinstance(spans_value, list):
            for raw_span in cast("list[object]", spans_value):
                if not isinstance(raw_span, dict):
                    continue
                span = parse_json_object(cast("object", raw_span))
                span_index = span.get("index", 0)
                span_start = span.get("start")
                span_end = span.get("end")
                if (
                    isinstance(span_start, (int, float))
                    and isinstance(span_end, (int, float))
                    and _json_int(span_index) >= _FIRST_WRITER_CHUNK_INDEX
                ):
                    fetch_spans.append((float(span_start), float(span_end)))
        fetch_encode_overlap = any(
            writer_start <= span_end and writer_commit >= span_start
            for span_start, span_end in fetch_spans
        )
    return {
        "writer_http_transfer_overlap": commit_inside_http,
        "writer_commit_inside_http_transfer": commit_inside_http,
        "writer_transaction_activity_overlapped_http_transfer": transaction_activity_overlap,
        "writer_stream_pull_fetch_encode_overlap": fetch_encode_overlap,
        "writer_database_fetch_overlap_status": (
            "not separately observable; app-side stream-pull intervals include fetch and encoding"
        ),
    }


def _load_recovery_plan(
    original: dict[str, object],
) -> tuple[list[dict[str, object]], list[tuple[int, str, int]]]:
    """Validate the completed run and compute the only supported missing coordinates."""
    original_trials = _json_object_list(original.get("isolation_trials", []))
    original_samples = _json_object_list(original.get("samples", []))
    if (
        len(original_samples) != _EXPECTED_SAMPLE_COUNT
        or len(original_trials) != _EXPECTED_ISOLATION_TRIAL_COUNT
    ):
        exception_message = "resume requires the recorded 12 HTTP samples and 7 isolation trials"
        raise RuntimeError(exception_message)
    for trial in original_trials:
        trial["valid"] = isolation_trial_is_valid(trial)
    if any(not trial["valid"] for trial in original_trials):
        exception_message = "resume requires every existing isolation trial to be valid"
        raise RuntimeError(exception_message)
    if original.get("isolation_recovery"):
        exception_message = "isolation recovery is already recorded"
        raise RuntimeError(exception_message)

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
        exception_message = f"unexpected missing isolation coordinates: {missing!r}"
        raise RuntimeError(exception_message)
    return original_trials, missing


class BenchmarkRunner:
    """Own one isolated Docker network, PostgreSQL container, and all output files."""

    def __init__(self) -> None:
        """Initialize per-run state without creating containers or files."""
        self.run_id: str = uuid4().hex[:8]
        self.started_at_utc: str = datetime.now(UTC).isoformat()
        self.database_name: str = f"gw_events_bench_{self.run_id}"
        self.network_name: str = f"gw-event-export-{self.run_id}"
        self.db_container: str = f"gw-event-export-db-{self.run_id}"
        self.db_password: str = uuid4().hex
        self.operator_password: str = secrets.token_urlsafe(24)
        self.control_token: str = secrets.token_urlsafe(32)
        self.app_database_url: str = (
            f"postgresql+asyncpg://postgres:{self.db_password}@db:5432/{self.database_name}"
        )
        ensure_disposable_database(
            self.app_database_url,
            expected_database=self.database_name,
        )
        self.scratch_root: Path = Path(tempfile.mkdtemp(prefix=f"gw-event-export-{self.run_id}-"))
        self.output_root: Path = REPO_ROOT / "docs" / "experiments" / "2026-10-03-event-export"
        self.logs_root: Path = self.output_root / "logs"
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.logs_root.mkdir(parents=True, exist_ok=True)
        self.commands: list[str] = []
        self.owned_containers: set[str] = set()
        self.cleanup_failures: list[dict[str, str]] = []
        self.network_created: bool = False
        self.database_version: str = "unavailable"
        self.alembic_head: str = "unverified"
        self.app_python_version: str = "unavailable"
        self.implementation_source_paths: dict[str, str] = {}
        self.samples: list[dict[str, object]] = []
        self.smoke: list[dict[str, object]] = []
        self.isolation_trials: list[dict[str, object]] = []
        self.failures: list[dict[str, str]] = []
        self.resource_diagnostics: list[dict[str, object]] = []
        self.current_phase: str = "initialization"

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
        return docker_command(arguments, timeout=timeout)

    def _record_cleanup_failure(self, resource: str, phase: str, error: Exception) -> None:
        self.cleanup_failures.append(
            {"resource": resource, "phase": phase, "error": f"{type(error).__name__}: {error}"}
        )

    def _remove_owned_container(self, container_name: str, *, phase: str) -> None:
        if container_name not in self.owned_containers:
            return
        try:
            _ = self._run_docker(["rm", "--force", container_name], timeout=30)
        except RuntimeError as error:
            message = str(error).lower()
            if "no such container" in message or "no such object" in message:
                self.owned_containers.discard(container_name)
                return
            self._record_cleanup_failure(container_name, phase, error)
        except Exception as error:  # noqa: BLE001 - retain any unexpected cleanup failure and ownership.
            self._record_cleanup_failure(container_name, phase, error)
        else:
            self.owned_containers.discard(container_name)

    def _run_owned_container(
        self,
        container_name: str,
        arguments: Sequence[str],
        *,
        timeout: int,
    ) -> str:
        self.owned_containers.add(container_name)
        try:
            return self._run_docker(arguments, timeout=timeout)
        finally:
            self._remove_owned_container(container_name, phase="container run finalizer")

    def _cleanup_task_resources(self, *, phase: str) -> None:
        for container_name in tuple(self.owned_containers):
            self._remove_owned_container(container_name, phase=phase)
        if not self.network_created:
            return
        try:
            _ = self._run_docker(["network", "rm", self.network_name], timeout=30)
        except RuntimeError as error:
            message = str(error).lower()
            if "no such network" in message or "not found" in message:
                self.network_created = False
                return
            self._record_cleanup_failure(self.network_name, phase, error)
        except Exception as error:  # noqa: BLE001 - retain any unexpected network cleanup failure.
            self._record_cleanup_failure(self.network_name, phase, error)
        else:
            self.network_created = False

    def _capture_container_logs(
        self,
        container_name: str,
        *,
        tail: int | None = None,
    ) -> str:
        command = ["docker", "logs"]
        if tail is not None:
            command.extend(["--tail", str(tail)])
        command.append(container_name)
        self.commands.append(" ".join(command))
        return docker_logs(container_name, tail=tail)

    def _database_diagnostics(self, label: str) -> dict[str, object]:
        """Capture effective Docker limits, cgroup counters, and tmpfs state."""
        snapshot: dict[str, object] = {
            "label": label,
            "captured_at_utc": datetime.now(UTC).isoformat(),
        }
        try:
            inspections = parse_json_array(
                self._run_docker(["inspect", self.db_container], timeout=15)
            )
            inspected = parse_json_object(inspections[0])
        except (RuntimeError, IndexError, ValueError) as error:
            snapshot["inspect_error"] = f"{type(error).__name__}: {error}"
            return snapshot
        state = parse_json_object(inspected.get("State", {}))
        host_config = parse_json_object(inspected.get("HostConfig", {}))
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
                        # Pyright rejects implicit concatenation of these static command fragments.
                        (
                            "for item in memory.current memory.peak "  # noqa: ISC003
                            + "memory.max memory.swap.current memory.swap.max; "
                            + 'do printf \'%s=\' "$item"; cat "/sys/fs/cgroup/$item" '
                            + "2>/dev/null || true; done"
                        ),
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
                        # Pyright rejects implicit concatenation of this static SQL statement.
                        (
                            "SELECT pg_database_size(current_database()), "  # noqa: ISC003
                            + "pg_total_relation_size('camera_events')"
                        ),
                    ],
                    timeout=15,
                )
            except RuntimeError as error:
                snapshot["live_probe_error"] = f"{type(error).__name__}: {error}"
        try:
            snapshot["postgres_log_tail"] = self._capture_container_logs(
                self.db_container, tail=200
            )
        except Exception as error:  # noqa: BLE001 - preserve log-tail diagnostics without blocking return.
            snapshot["postgres_log_error"] = f"{type(error).__name__}: {error}"
        return snapshot

    def _start_environment(self) -> None:
        _ = self._run_docker(["network", "create", "--internal", self.network_name])
        self.network_created = True
        self.owned_containers.add(self.db_container)
        _ = self._run_docker(
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
        self._initialize_database()

    def _initialize_database(self) -> None:
        ready = False
        for _ in range(120):
            try:
                _ = self._run_docker(
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
            exception_message = "disposable PostgreSQL did not become ready on the private network"
            raise RuntimeError(exception_message)
        self.run_utility(
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
            exception_message = f"unexpected disposable database Alembic head: {self.alembic_head}"
            raise RuntimeError(exception_message)

    def run_utility(
        self,
        *,
        suffix: str,
        command: Sequence[str],
        script: bool = True,
        include_alembic: bool = False,
        timeout: int = 300,
    ) -> None:
        """Start one task-owned utility container and collect its output."""
        utility_name = f"gw-event-export-util-{self.run_id}-{suffix}"
        arguments = [
            "run",
            "--pull=never",
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
            alembic_source = REPO_ROOT / "server" / "alembic"
            arguments.extend(
                [
                    "--mount",
                    f"type=bind,src={alembic_source},dst=/work/server/alembic,readonly",
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
        _ = self._run_owned_container(utility_name, arguments, timeout=timeout)

    def _start_app(self, *, mode: str, container_name: str) -> None:
        self.owned_containers.add(container_name)
        _ = self._run_docker(
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

    def run_client(
        self,
        *,
        app_container: str,
        coordinate: SampleCoordinate,
        expected_path: Path,
    ) -> dict[str, object]:
        """Run one authenticated client sample and return its correctness record."""
        mode = coordinate.mode
        row_count = coordinate.row_count
        repeat = coordinate.repeat
        writer_enabled = coordinate.writer_enabled
        client_name = f"gw-event-export-client-{self.run_id}-{mode}-{row_count}-{repeat}"
        output_csv = self.scratch_root / f"{mode}-{row_count}-{repeat}.csv"
        result_path = self.scratch_root / f"{mode}-{row_count}-{repeat}.json"
        command = [
            "run",
            "--pull=never",
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
        _ = self._run_owned_container(client_name, command, timeout=300)
        result = read_json_object(result_path)
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
                _ = shutil.copyfile(output_csv, invalid_root / output_csv.name)
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
    ) -> dict[str, object]:
        app_container = f"gw-event-export-app-{self.run_id}-{mode}-{row_count}-{repeat}"
        try:
            self._start_app(mode=mode, container_name=app_container)
            result = self.run_client(
                app_container=app_container,
                coordinate=SampleCoordinate(
                    mode=mode,
                    row_count=row_count,
                    repeat=repeat,
                    writer_enabled=writer_enabled,
                ),
                expected_path=expected_path,
            )
            self.app_python_version = self._run_docker(
                ["exec", app_container, "python", "--version"]
            )
            app_metrics_value = result.get("application_metrics")
            app_metrics = (
                parse_json_object(cast("object", app_metrics_value))
                if isinstance(app_metrics_value, dict)
                else {}
            )
            result.update(writer_overlap_metadata(result, app_metrics))
            result["application_peak_rss_kib"] = app_metrics.get("application_peak_rss_kib")
            source_paths_value = app_metrics.get("source_paths")
            if isinstance(source_paths_value, dict):
                source_paths = parse_json_object(cast("object", source_paths_value))
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
            return result
        finally:
            self._save_app_log(app_container, mode, row_count, repeat)
            self._remove_owned_container(app_container, phase="HTTP sample finalizer")

    def _save_app_log(self, container_name: str, mode: str, count: int, repeat: int) -> None:
        try:
            output = self._capture_container_logs(container_name)
        except Exception as error:  # noqa: BLE001 - preserve app log failure without replacing sample result.
            output = str(error)
        path = self.logs_root / f"{mode}-{count}-{repeat}.app.log"
        _ = path.write_text(output + "\n", encoding="utf-8")

    def _experiment_metadata(self) -> dict[str, object]:
        try:
            docker_version = docker_command(["version", "--format", "{{.Server.Version}}"])
        except RuntimeError as error:
            docker_version = str(error)
        return {
            "run_id": self.run_id,
            "started_at_utc": self.started_at_utc,
            "repository_head_at_start": subprocess.run(  # noqa: S603 - fixed read-only Git argv, no shell.
                [_resolve_executable("git"), "rev-parse", "HEAD"],
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
                "aggregate_database_connections": """<=4; setup utility is sequential and app pool \
caps at three""",
            },
            "composition": """real FastAPI session and export routers, real AuthService and \
Database; no media services""",
            "authentication": """synthetic operator credential; actual HTTP login \
and cookie guard used""",
            "download": """stdlib HTTP client streams response to a temporary file; \
no response.read() for full body""",
            "rss": """application resource.getrusage(RUSAGE_SELF).ru_maxrss in \
Linux KiB; baseline is marked after login""",
        }

    def _write_command_log(self) -> None:
        command_lines = [
            "Reproduction command (host):",
            """PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python \
qa/events/benchmark_export.py run""",
            "",
            """Exact per-run Docker/migration commands, with synthetic \
passwords and control tokens redacted:""",
            *self.commands,
            "",
            """Measurements run sequentially; each app/client container is \
task-owned, resource-limited, and removed.""",
        ]
        _ = (self.output_root / "COMMANDS.md").write_text(
            "\n".join(command_lines) + "\n", encoding="utf-8"
        )

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
                "isolation_summary": summarize_isolation(self.isolation_trials),
                "failures": self.failures,
                "cleanup_failures": self.cleanup_failures,
                "remaining_owned_containers": sorted(self.owned_containers),
                "network_still_owned": self.network_created,
                "caveats": [
                    """Three repetitions support median and range only; no p95 or \
significance claim.""",
                    "The reduced app composition excludes media services and workers.",
                    """Startup and Argon2 login contribute to process ru_maxrss; report \
absolute peak and marked baseline.""",
                    """HTTP overlap and app-side stream-pull intervals are separate; \
stream-pull spans include DB fetch and CSV encoding.""",
                    """A single SELECT uses one Read Committed snapshot; the RC/RR \
multi-SELECT probe is a separate experiment.""",
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
            exception_message = "streaming/buffered smoke outputs differed or failed CSV validation"
            raise RuntimeError(exception_message)

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
                    isolation_slug = isolation.lower().replace(" ", "-")
                    self.run_utility(
                        suffix=f"isolation-{row_count}-{isolation_slug}-{repeat}",
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
                    trial = read_json_object(result_path)
                    trial["valid"] = isolation_trial_is_valid(trial)
                    self.isolation_trials.append(trial)
                    self._save_results()
                    if not trial["valid"]:
                        trial_label = f"{row_count} {isolation} repeat {repeat}"
                        exception_message = (
                            f"isolation trial failed validity checks: {trial_label}"
                        )
                        raise RuntimeError(exception_message)

    def _run_one_recovery_trial(
        self,
        coordinate: tuple[int, str, int],
        save_recovery: Callable[[str], None],
    ) -> bool:
        """Run and record one recovery coordinate, returning false on its first failure."""
        row_count, isolation, repeat = coordinate
        label = f"{row_count}-{isolation.lower().replace(' ', '-')}-{repeat}"
        self.current_phase = f"recovery isolation trial {label}"
        self.resource_diagnostics.append(self._database_diagnostics(f"before-{label}"))
        expected_path = self.scratch_root / f"expected-{label}.json"
        result_path = self.scratch_root / f"result-{label}.json"
        try:
            self.run_utility(
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
            trial = read_json_object(result_path)
            trial["valid"] = isolation_trial_is_valid(trial)
            self.isolation_trials.append(trial)
            self.resource_diagnostics.append(self._database_diagnostics(f"after-{label}"))
            save_recovery("in_progress")
            if not trial["valid"]:
                _raise_invalid_isolation_trial(label)
        except Exception as error:  # noqa: BLE001 - preserve any trial failure and stop recovery.
            self.failures.append(
                {
                    "phase": self.current_phase,
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                }
            )
            self.resource_diagnostics.append(self._database_diagnostics(f"failure-{label}"))
            save_recovery("failed")
            return False
        return True

    def _execute_recovery_trials(
        self,
        missing: list[tuple[int, str, int]],
        save_recovery: Callable[[str], None],
    ) -> int:
        """Own recovery setup and its fail-fast ordered trial sequence."""
        exit_code = 1
        try:
            self.current_phase = "create isolated recovery DB and apply migrations"
            self._start_environment()
            self.resource_diagnostics.append(self._database_diagnostics("after-migration"))
            for coordinate in missing:
                if not self._run_one_recovery_trial(coordinate, save_recovery):
                    break
            if len(self.isolation_trials) == len(missing) and not self.failures:
                exit_code = 0
            save_recovery("completed" if exit_code == 0 else "failed")
        except Exception as error:  # noqa: BLE001 - retain setup failure evidence before cleanup.
            self.failures.append(
                {
                    "phase": self.current_phase,
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                }
            )
            self.resource_diagnostics.append(self._database_diagnostics("setup-failure"))
            save_recovery("failed")
        return exit_code

    def _finalize_recovery(
        self,
        log_path: Path,
        save_recovery: Callable[[str], None],
        exit_code: int,
    ) -> int:
        """Capture logs, clean owned resources, and save the post-cleanup recovery state."""
        if self.db_container in self.owned_containers:
            try:
                _ = log_path.write_text(
                    self._capture_container_logs(self.db_container) + "\n",
                    encoding="utf-8",
                )
            except Exception as error:  # noqa: BLE001 - save capture failure, then still clean up.
                _ = log_path.write_text(f"{error}\n", encoding="utf-8")
        self._cleanup_task_resources(phase="isolation recovery finalizer")
        if self.cleanup_failures or self.owned_containers or self.network_created:
            exit_code = 5
        self._write_recovery_commands()
        save_recovery("completed" if exit_code == 0 else "failed")
        shutil.rmtree(self.scratch_root, ignore_errors=True)
        return exit_code

    def _merge_successful_recovery(
        self,
        results_path: Path,
        original: dict[str, object],
        original_trials: list[dict[str, object]],
        recovery_path: Path,
        log_path: Path,
    ) -> int:
        """Merge recovered rows and summaries only after all trials and cleanup pass."""
        initial_path = self.output_root / "initial-run-results.json"
        if not initial_path.exists():
            _ = shutil.copy2(results_path, initial_path)
        merged = dict(original)
        merged["sample_summary_pooled_descriptive_only"] = merged.pop("sample_summary", {})
        merged["sample_summary"] = summarize_samples_by_dataset(
            _json_object_list(merged.get("samples", []))
        )
        merged["sample_summary_by_dataset_and_mode"] = merged["sample_summary"]
        merged["isolation_trials"] = original_trials + self.isolation_trials
        if any(not isolation_trial_is_valid(trial) for trial in merged["isolation_trials"]):
            exception_message = "cannot merge invalid isolation trials into completed results"
            raise RuntimeError(exception_message)
        merged["isolation_summary"] = summarize_isolation(merged["isolation_trials"])
        merged["isolation_recovery"] = {
            "run_id": self.run_id,
            "status": "completed",
            "source_run_id": parse_json_object(original.get("environment", {})).get("run_id"),
            "resumed_trial_count": len(self.isolation_trials),
            "total_isolation_trial_count": len(merged["isolation_trials"]),
            "raw_artifact": recovery_path.name,
            "initial_run_archive": initial_path.name,
            "postgres_log": log_path.name,
            "diagnostics_are_outside_trial_timing": True,
        }
        _ = results_path.write_text(
            json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return 0

    def resume_missing_isolation_trials(self) -> int:
        """Run only absent isolation coordinates from the already completed HTTP run."""
        results_path = self.output_root / "results.json"
        original = read_json_object(results_path)
        original_trials, missing = _load_recovery_plan(original)
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
                    "cleanup_failures": self.cleanup_failures,
                    "remaining_owned_containers": sorted(self.owned_containers),
                    "network_still_owned": self.network_created,
                    "resource_diagnostics": self.resource_diagnostics,
                },
            )

        exit_code = 1
        try:
            exit_code = self._execute_recovery_trials(missing, save_recovery)
        except Exception as error:  # noqa: BLE001 - preserve owner failure evidence before cleanup.
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
            exit_code = self._finalize_recovery(log_path, save_recovery, exit_code)
        if exit_code != 0:
            return exit_code
        return self._merge_successful_recovery(
            results_path,
            original,
            original_trials,
            recovery_path,
            log_path,
        )

    def _write_recovery_commands(self) -> None:
        path = self.output_root / f"isolation-recovery-{self.run_id}-commands.md"
        python_command = "PYTHONPATH=server/src /mnt/data/gods-watching/.venv/bin/python"
        script_command = "qa/events/benchmark_export.py resume-isolation"
        recovery_command = f"{python_command} {script_command}"
        _ = path.write_text(
            "".join(
                (
                    f"Recovery command: `{recovery_command}`\n\n",
                    "Exact per-run Docker commands, with synthetic credentials redacted:\n\n",
                    "\n".join(f"- `{command}`" for command in self.commands),
                    "\n",
                )
            ),
            encoding="utf-8",
        )

    def _seed_via_utility(self, *, row_count: int, expected_path: Path, suffix: str) -> None:
        self.run_utility(
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
        """Execute one benchmark run and report its measured validation outcome."""
        exit_code = 0
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
                exit_code = 2
            if not smoke_only and (
                len(self.samples) != _EXPECTED_SAMPLE_COUNT
                or len(self.isolation_trials) != _EXPECTED_SAMPLE_COUNT
            ):
                exit_code = 3
            if not smoke_only and any(
                not isolation_trial_is_valid(trial) for trial in self.isolation_trials
            ):
                exit_code = 4
        except Exception as error:  # noqa: BLE001 - save run failure evidence before final cleanup.
            self.failures.append(
                {
                    "phase": self.current_phase,
                    "error_type": type(error).__name__,
                    "message": str(error)[:2_000],
                }
            )
            exit_code = 1
        finally:
            self._collect_postgres_log()
            self._cleanup_task_resources(phase="benchmark finalizer")
            if self.cleanup_failures or self.owned_containers or self.network_created:
                exit_code = 5
            self._save_results()
            shutil.rmtree(self.scratch_root, ignore_errors=True)
        return exit_code

    def _collect_postgres_log(self) -> None:
        if self.db_container not in self.owned_containers:
            return
        try:
            output = self._capture_container_logs(self.db_container)
        except Exception as error:  # noqa: BLE001 - log capture is best-effort during final cleanup.
            output = str(error)
        _ = (self.logs_root / "postgres.log").write_text(output + "\n", encoding="utf-8")


def summarize_isolation(
    trials: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Summarize per-isolation trial outcomes without percentile claims."""
    summary: dict[str, dict[str, object]] = {}
    for size in _SAMPLE_COUNTS:
        for isolation in ("READ COMMITTED", "REPEATABLE READ"):
            selected = [
                trial
                for trial in trials
                if trial.get("dataset_size") == size and trial.get("isolation") == isolation
            ]
            valid_trials = [trial for trial in selected if isolation_trial_is_valid(trial)]
            elapsed = [
                float(elapsed_value)
                for trial in valid_trials
                if isinstance(
                    elapsed_value := trial.get("elapsed_seconds"),
                    (int, float),
                )
            ]
            summary[f"{size}:{isolation}"] = {
                "trial_count": len(selected),
                "valid_trial_count": len(valid_trials),
                "invalid_trial_count": len(selected) - len(valid_trials),
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
) -> dict[str, object]:
    start = time.monotonic()
    trial: dict[str, object] = {
        "dataset_size": row_count,
        "isolation": isolation,
        "repeat": repeat,
        "forced_interleaving": """initial COUNT, commit writer batch, then materialize \
repository projection in the same transaction""",
        "expected_count": row_count,
        "error": None,
    }
    writer_ids: list[int] = []
    try:
        async with database.engine.connect() as connection:
            transaction_connection = await connection.execution_options(isolation_level=isolation)
            transaction = await transaction_connection.begin()
            try:
                initial = await transaction_connection.scalar(
                    select(func.count()).select_from(CameraEvent)
                )
                writer = await write_benchmark_batch(database, offset=(row_count * repeat))
                writer_ids_value = writer["ids"]
                if not isinstance(writer_ids_value, list):
                    _raise_invalid_writer_ids()
                writer_ids = [
                    _json_int(identifier) for identifier in cast("list[object]", writer_ids_value)
                ]
                projected = await transaction_connection.execute(
                    EventRepository.statement(EventExportFilters())
                )
                exported = projected.tuples().all()
                exported_ids = [int(row[0]) for row in exported]
                await transaction.commit()
            except BaseException:
                if transaction.is_active:
                    await transaction.rollback()
                raise
        expected_exported = list(expected_ids)
        if isolation == "READ COMMITTED":
            expected_exported.extend(writer_ids)
        trial.update(
            {
                "initial_count": int(initial or 0),
                "writer_count": _json_int(writer["count"]),
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
    except Exception as error:  # noqa: BLE001 - isolate invalid trial data from later summaries.
        trial["error"] = type(error).__name__
    trial["elapsed_seconds"] = time.monotonic() - start
    return trial


def _utility_main(arguments: _CommandLineArguments) -> int:
    database_url = os.environ["GW_DATABASE_URL"]
    database_name = os.environ["GW_BENCHMARK_DATABASE_NAME"]
    ensure_disposable_database(database_url, expected_database=database_name)
    database = make_engine(database_url)

    async def run_seed() -> None:
        _ = await _seed_database(
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
    """Dispatch the benchmark CLI command selected by its arguments."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="run smoke, export samples, and isolation probe")
    _ = run_parser.add_argument("--smoke-only", action="store_true")
    _ = subparsers.add_parser("resume-isolation", help="run only the five absent isolation trials")
    client_parser = subparsers.add_parser("client-sample", help=argparse.SUPPRESS)
    _ = client_parser.add_argument("--app-host", required=True)
    _ = client_parser.add_argument("--expected", required=True)
    _ = client_parser.add_argument("--output-csv", required=True)
    _ = client_parser.add_argument("--result", required=True)
    _ = client_parser.add_argument("--writer", action="store_true")
    seed_parser = subparsers.add_parser("seed", help=argparse.SUPPRESS)
    _ = seed_parser.add_argument("--row-count", required=True, type=int)
    _ = seed_parser.add_argument("--expected", required=True)
    isolation_parser = subparsers.add_parser("isolation-sample", help=argparse.SUPPRESS)
    _ = isolation_parser.add_argument("--row-count", required=True, type=int)
    _ = isolation_parser.add_argument(
        "--isolation", required=True, choices=("READ COMMITTED", "REPEATABLE READ")
    )
    _ = isolation_parser.add_argument("--repeat", required=True, type=int)
    _ = isolation_parser.add_argument("--expected", required=True)
    _ = isolation_parser.add_argument("--result", required=True)
    arguments = parser.parse_args(namespace=_CommandLineArguments())
    if arguments.command == "client-sample":
        return _client_main(arguments)
    if arguments.command in {"seed", "isolation-sample"}:
        return _utility_main(arguments)
    if arguments.command == "resume-isolation":
        return BenchmarkRunner().resume_missing_isolation_trials()
    return BenchmarkRunner().run(smoke_only=bool(arguments.smoke_only))


if __name__ == "__main__":
    raise SystemExit(main())
