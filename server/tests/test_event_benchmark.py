"""Correctness gates for the actual HTTP export benchmark."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast, final, override

import pytest
from qa.events import benchmark_app, benchmark_export
from qa.events.benchmark_app import Measurements, TimedExporter
from qa.events.benchmark_export import (
    BenchmarkRunner,
    ensure_disposable_database,
    summarize_isolation,
    summarize_samples,
    summarize_samples_by_dataset,
    validate_csv_file,
)
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine

from gods_watching.api.sessionguard import require_session as original_require_session
from gods_watching.contracts.events import EventExportFilters
from gods_watching.events.export import EventExportService, PreparedCsvExport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from fastapi import FastAPI
    from sqlalchemy.ext.asyncio import AsyncEngine
    from starlette.types import Message, Scope

    from gods_watching.api.app_settings import ApiSettings
    from gods_watching.api.sessionguard import AuthCapability
    from gods_watching.api.sessionguard import AuthenticatedRequest as SessionAuthenticatedRequest

_FIELDS: tuple[str, ...] = ("id", "occurred_at", "event_type", "camera_id", "camera_name")
_SYNTHETIC_LOGIN = "synthetic-password"
_SYNTHETIC_GUARD = "synthetic-control-token"
_EXPECTED = {
    1: {
        "id": "1",
        "occurred_at": "2026-10-03T00:00:00Z",
        "event_type": "camera.created",
        "camera_id": "00000000-0000-0000-0000-000000000001",
        "camera_name": "Front door",
    },
    2: {
        "id": "2",
        "occurred_at": "2026-10-03T00:00:01Z",
        "event_type": "camera.updated",
        "camera_id": "00000000-0000-0000-0000-000000000002",
        "camera_name": "Parking, east",
    },
}


@final
class _TrackingOutput(io.BytesIO):
    name: str
    closed_bytes: bytes

    def __init__(self) -> None:
        super().__init__()
        self.name = "benchmark-test-output"
        self.closed_bytes = b""

    @override
    def close(self) -> None:
        self.closed_bytes = self.getvalue()
        super().close()


async def _asgi_status(
    app: FastAPI,
    path: str,
    *,
    headers: dict[str, str] | None = None,
) -> int:
    scope: Scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "root_path": "",
        "headers": [
            (key.lower().encode("ascii"), value.encode("latin-1"))
            for key, value in (headers or {}).items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    messages: list[Message] = []
    request_sent = False

    async def receive() -> Message:
        nonlocal request_sent
        if request_sent:
            return {"type": "http.disconnect"}
        request_sent = True
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        messages.append(message)

    await app(scope, receive, send)
    for message in messages:
        if message["type"] == "http.response.start":
            status = cast("object", message["status"])
            if isinstance(status, int):
                return status
    exception_message = "ASGI app returned without an HTTP response"
    raise AssertionError(exception_message)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDS, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def test_missing_duplicate_and_unexpected_ids_fail_correctness_and_eligibility(
    tmp_path: Path,
) -> None:
    rows = [
        _EXPECTED[1],
        _EXPECTED[1],
        {
            "id": "3",
            "occurred_at": "2026-10-03T00:00:02Z",
            "event_type": "camera.deleted",
            "camera_id": "00000000-0000-0000-0000-000000000003",
            "camera_name": "Unexpected",
        },
    ]
    output = tmp_path / "bad.csv"
    _write_rows(output, rows)

    result = validate_csv_file(output, _EXPECTED)

    assert result.correct is False
    assert result.missing_ids == (2,)
    assert result.duplicate_ids == (1,)
    assert result.unexpected_ids == (3,)
    assert result.performance_eligible is False


def test_field_mismatch_and_invalid_csv_fail_correctness(tmp_path: Path) -> None:
    mismatched = dict(_EXPECTED[1], camera_name="Wrong name")
    output = tmp_path / "mismatched.csv"
    _write_rows(output, [mismatched, _EXPECTED[2]])

    result = validate_csv_file(output, _EXPECTED)

    assert result.correct is False
    assert result.field_mismatch_count == 1
    assert result.performance_eligible is False

    invalid = tmp_path / "invalid.csv"
    _ = invalid.write_text("id,event_type\n1,camera.created\n", encoding="utf-8")
    invalid_result = validate_csv_file(invalid, _EXPECTED)
    assert invalid_result.correct is False
    assert invalid_result.parse_error is not None
    assert invalid_result.performance_eligible is False


def test_csv_row_with_surplus_column_fails_correctness(tmp_path: Path) -> None:
    output = tmp_path / "extra-column.csv"
    csv_content = """\
id,occurred_at,event_type,camera_id,camera_name
1,2026-10-03T00:00:00Z,camera.created,00000000-0000-0000-0000-000000000001,Front door,unexpected
2,2026-10-03T00:00:01Z,camera.updated,00000000-0000-0000-0000-000000000002,"Parking, east"
"""
    _ = output.write_text(
        csv_content,
        encoding="utf-8",
    )

    result = validate_csv_file(output, _EXPECTED)

    assert result.correct is False
    assert result.parse_error == "CSV row has missing or extra columns"
    assert result.performance_eligible is False


def test_writer_commit_overlap_is_distinct_from_transaction_activity_overlap() -> None:
    calculate_metrics = getattr(benchmark_export, "writer_overlap_metrics", None)
    assert callable(calculate_metrics), "writer overlap metrics helper should be defined"
    metrics = calculate_metrics(
        http_start=10.0,
        http_end=20.0,
        writer_start=8.0,
        writer_commit=20.001,
    )

    assert metrics == {
        "transaction_activity_overlaps_http_transfer": True,
        "commit_inside_http_transfer": False,
    }


def test_download_triggers_writer_after_first_row_and_drains_64k_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"id,name\n1,first row\n2,remaining row\n"
    first_row_end = content.index(b"\n", content.index(b"\n") + 1) + 1
    events: list[tuple[str, int | float]] = []

    @final
    class FakeResponse:
        def __init__(self) -> None:
            self.position: int = 0
            self.read_sizes: list[int] = []

        def read(self, amt: int | None = None) -> bytes:
            size = amt if amt is not None else -1
            self.read_sizes.append(size)
            value = content[self.position : self.position + size]
            self.position += len(value)
            return value

    @final
    class FakeThread:
        target: Callable[..., object] | None
        args: tuple[object, ...]
        name: str
        alive: bool

        def __init__(
            self,
            *,
            target: Callable[..., object] | None,
            args: tuple[object, ...],
            name: str,
        ) -> None:
            self.target = target
            self.args = args
            self.name = name
            self.alive = False

        def start(self) -> None:
            events.append(("writer_started_at_byte", response.position))

        def join(self, *, timeout: float) -> None:
            events.append(("join_timeout", timeout))

        def is_alive(self) -> bool:
            return self.alive

    response = FakeResponse()
    output_path = tmp_path / "stream.csv"
    tracked_output = _TrackingOutput()
    clock_calls = 0
    clock_values = iter((10.25, 10.5, 12.0))

    def monotonic() -> float:
        nonlocal clock_calls
        clock_calls += 1
        value = next(clock_values)
        if clock_calls == 3:
            events.append(("output_closed_at_transfer_end", int(tracked_output.closed)))
        return value

    def tracked_open(
        path: Path,
        mode: str = "r",
    ) -> _TrackingOutput:
        assert path == output_path
        assert mode == "wb"
        return tracked_output

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(threading, "Thread", FakeThread)
    monkeypatch.setattr(Path, "open", tracked_open)
    downloaded = benchmark_export.download_export_to_file(
        output_path,
        response,
        10.0,
        benchmark_export.ControlClientConfig("app", "token"),
        writer_enabled=True,
    )

    assert tracked_output.closed_bytes == content
    assert response.read_sizes == [1] * first_row_end + [64 * 1024, 64 * 1024]
    assert downloaded.body.first_byte_latency == 0.25
    assert downloaded.body.writer.triggered_at == 10.5
    assert downloaded.transfer_ended == 12.0
    assert events == [
        ("writer_started_at_byte", first_row_end),
        ("output_closed_at_transfer_end", 1),
    ]
    benchmark_export.join_writer(downloaded.body.writer)
    assert events[-1] == ("join_timeout", 120.0)


def testwriter_overlap_metadata_preserves_false_and_unknown_states() -> None:
    no_writer = benchmark_export.writer_overlap_metadata(
        {"http_request_started_monotonic": 10.0, "http_transfer_ended_monotonic": 20.0},
        {},
    )
    assert no_writer["writer_transaction_activity_overlapped_http_transfer"] is False
    assert no_writer["writer_commit_inside_http_transfer"] is False
    assert no_writer["writer_stream_pull_fetch_encode_overlap"] is None

    overlaps = benchmark_export.writer_overlap_metadata(
        {"http_request_started_monotonic": 10.0, "http_transfer_ended_monotonic": 20.0},
        {
            "writer": {"started_monotonic": 8.0, "committed_monotonic": 20.001},
            "chunk_pull_spans": [
                {"index": 1, "start": 10.0, "end": 20.0},
                {"index": 2, "start": 9.0, "end": 11.0},
            ],
        },
    )
    assert overlaps["writer_transaction_activity_overlapped_http_transfer"] is True
    assert overlaps["writer_commit_inside_http_transfer"] is False
    assert overlaps["writer_stream_pull_fetch_encode_overlap"] is True


def test_isolation_validity_requires_snapshot_count_writer_and_ids() -> None:
    valid_rc = {
        "dataset_size": 10_000,
        "isolation": "READ COMMITTED",
        "error": None,
        "initial_count": 10_000,
        "writer_count": 100,
        "export_count": 10_100,
        "exact_ids_match": True,
        "elapsed_seconds": 0.2,
    }
    is_valid = getattr(benchmark_export, "isolation_trial_is_valid", None)
    assert callable(is_valid), "isolation validity predicate should be defined"
    assert is_valid(valid_rc) is True

    for field, value in (
        ("error", "DatabaseError"),
        ("initial_count", 9_999),
        ("writer_count", 99),
        ("export_count", 10_000),
        ("exact_ids_match", False),
    ):
        invalid = dict(valid_rc, **{field: value})
        assert is_valid(invalid) is False


def test_recovery_finalizer_runs_when_trial_owner_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_trials = [
        {
            "dataset_size": size,
            "isolation": isolation,
            "repeat": repeat,
            "error": None,
            "initial_count": size,
            "writer_count": 100,
            "export_count": size + (100 if isolation == "READ COMMITTED" else 0),
            "exact_ids_match": True,
        }
        for size, isolation, repeats in (
            (10_000, "READ COMMITTED", (1, 2, 3)),
            (10_000, "REPEATABLE READ", (1, 2, 3)),
            (50_000, "READ COMMITTED", (1,)),
        )
        for repeat in repeats
    ]
    _ = (tmp_path / "results.json").write_text(
        json.dumps({"samples": [{}] * 12, "isolation_trials": original_trials}),
        encoding="utf-8",
    )

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.output_root = tmp_path
    runner.logs_root = tmp_path
    runner.run_id = "deadbeef"
    runner.started_at_utc = "2026-10-03T00:00:00+00:00"
    runner.isolation_trials = []
    runner.failures = []
    runner.cleanup_failures = []
    runner.owned_containers = set()
    runner.network_created = False
    runner.resource_diagnostics = []
    finalizer_calls: list[tuple[Path, int]] = []

    def interrupt_trials(
        missing: list[tuple[int, str, int]],
        save_recovery: Callable[[str], None],
    ) -> int:
        del missing, save_recovery
        raise KeyboardInterrupt

    def record_finalizer(
        log_path: Path,
        save_recovery: Callable[[str], None],
        exit_code: int,
    ) -> int:
        del save_recovery
        finalizer_calls.append((log_path, exit_code))
        return exit_code

    monkeypatch.setattr(runner, "_execute_recovery_trials", interrupt_trials)
    monkeypatch.setattr(runner, "_finalize_recovery", record_finalizer)

    with pytest.raises(KeyboardInterrupt):
        _ = runner.resume_missing_isolation_trials()

    assert finalizer_calls == [(tmp_path / "isolation-recovery-deadbeef-postgres.log", 1)]


def test_invalid_isolation_trials_remain_recorded_but_are_excluded_from_median() -> None:
    valid = {
        "dataset_size": 10_000,
        "isolation": "READ COMMITTED",
        "error": None,
        "initial_count": 10_000,
        "writer_count": 100,
        "export_count": 10_100,
        "exact_ids_match": True,
        "elapsed_seconds": 0.2,
    }
    invalid = dict(valid, export_count=10_000, exact_ids_match=False, elapsed_seconds=0.001)

    summary = summarize_isolation([valid, invalid])["10000:READ COMMITTED"]

    assert summary["trial_count"] == 2
    assert summary["valid_trial_count"] == 1
    assert summary["invalid_trial_count"] == 1
    assert summary["elapsed_seconds_median"] == 0.2


def test_log_capture_keeps_stdout_and_stderr_while_structured_docker_keeps_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="structured stdout",
            stderr="server diagnostics",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    def test_which(name: str) -> str | None:
        return f"test-tools/{name}"

    monkeypatch.setattr(shutil, "which", test_which)
    logs = benchmark_export.docker_logs("db", tail=20)

    assert "structured stdout" in logs
    assert "server diagnostics" in logs
    assert "stdout" in logs.lower()
    assert "stderr" in logs.lower()
    assert Path(calls[-1][0]).is_absolute()
    assert calls[-1][1:3] == ["logs", "--tail"]
    assert benchmark_export.docker_command(["ps"]) == "structured stdout"
    assert Path(calls[-1][0]).name == "docker"


def test_missing_host_executable_fails_before_process_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_executable(_: str) -> str | None:
        return None

    monkeypatch.setattr(shutil, "which", missing_executable)

    with pytest.raises(FileNotFoundError, match="required executable 'docker'"):
        _ = benchmark_export.docker_command(["version"])


def _timeout_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_remove: bool
) -> tuple[BenchmarkRunner, list[str]]:
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.run_id = "deadbeef"
    runner.network_name = "gw-event-export-deadbeef"
    runner.scratch_root = tmp_path
    runner.app_database_url = (
        "postgresql+asyncpg://postgres:secret@db:5432/gw_events_bench_deadbeef"
    )
    runner.database_name = "gw_events_bench_deadbeef"
    runner.owned_containers = set()
    runner.cleanup_failures = []
    calls: list[str] = []

    def fake_run_docker(arguments: list[str], *, timeout: int = 60) -> str:
        if arguments[0] == "run":
            name = arguments[arguments.index("--name") + 1]
            calls.append(f"run:{name}:registered={name in runner.owned_containers}")
            raise subprocess.TimeoutExpired(["docker", *arguments], timeout)
        if arguments[:2] == ["rm", "--force"]:
            name = arguments[2]
            calls.append(f"remove:{name}")
            if fail_remove:
                exception_message = "docker rm failed: daemon unavailable"
                raise RuntimeError(exception_message)
            return name
        exception_message = f"unexpected docker command: {arguments[:2]}"
        raise AssertionError(exception_message)

    monkeypatch.setattr(runner, "_run_docker", fake_run_docker)
    return runner, calls


def test_timed_out_utility_is_registered_and_force_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, calls = _timeout_runner(tmp_path, monkeypatch, fail_remove=False)

    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_utility(suffix="timeout", command=("seed",), timeout=1)

    assert calls == [
        "run:gw-event-export-util-deadbeef-timeout:registered=True",
        "remove:gw-event-export-util-deadbeef-timeout",
    ]
    assert not runner.owned_containers
    assert runner.cleanup_failures == []


def test_timed_out_utility_cleanup_failure_is_recorded_and_ownership_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, calls = _timeout_runner(tmp_path, monkeypatch, fail_remove=True)

    with pytest.raises(subprocess.TimeoutExpired):
        runner.run_utility(suffix="timeout", command=("seed",), timeout=1)

    name = "gw-event-export-util-deadbeef-timeout"
    assert calls[-1] == f"remove:{name}"
    assert name in runner.owned_containers
    assert runner.cleanup_failures[0]["resource"] == name
    assert "daemon unavailable" in runner.cleanup_failures[0]["error"]


def test_timed_out_client_is_registered_and_force_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, calls = _timeout_runner(tmp_path, monkeypatch, fail_remove=False)
    runner.operator_password = _SYNTHETIC_LOGIN
    runner.control_token = _SYNTHETIC_GUARD

    with pytest.raises(subprocess.TimeoutExpired):
        _ = runner.run_client(
            app_container="gw-event-export-app-deadbeef-streaming-10000-1",
            coordinate=benchmark_export.SampleCoordinate(
                mode="streaming",
                row_count=10_000,
                repeat=1,
                writer_enabled=True,
            ),
            expected_path=tmp_path / "expected.json",
        )

    name = "gw-event-export-client-deadbeef-streaming-10000-1"
    assert calls == [f"run:{name}:registered=True", f"remove:{name}"]
    assert not runner.owned_containers
    assert runner.cleanup_failures == []


def test_incorrect_samples_are_excluded_from_comparison_summary() -> None:
    summary = summarize_samples(
        [
            {"mode": "streaming", "latency_seconds": 4.0, "correctness": True},
            {"mode": "streaming", "latency_seconds": 0.2, "correctness": False},
            {"mode": "buffered", "latency_seconds": 5.0, "correctness": True},
        ]
    )

    assert summary["streaming"]["sample_count"] == 1
    assert summary["streaming"]["median_latency_seconds"] == 4.0
    assert summary["buffered"]["sample_count"] == 1
    assert summary["buffered"]["median_latency_seconds"] == 5.0


def test_dataset_summary_keeps_sizes_separate_and_reports_three_sample_ranges() -> None:
    samples = [
        {
            "dataset_size": size,
            "mode": mode,
            "correctness": True,
            "latency_seconds": latency,
            "first_byte_latency_seconds": latency / 10,
            "application_peak_rss_kib": peak,
            "startup_login_baseline_rss_kib": peak,
        }
        for size, mode, values in (
            (10_000, "streaming", [(0.1, 10), (0.2, 20), (0.3, 30)]),
            (10_000, "buffered", [(0.4, 40), (0.5, 50), (0.6, 60)]),
            (50_000, "streaming", [(1.0, 100), (2.0, 200), (3.0, 300)]),
            (50_000, "buffered", [(4.0, 400), (5.0, 500), (6.0, 600)]),
        )
        for latency, peak in values
    ]
    samples.append(
        {
            "dataset_size": 50_000,
            "mode": "streaming",
            "correctness": False,
            "latency_seconds": 0.001,
        }
    )

    summary = summarize_samples_by_dataset(samples)

    assert summary["10000:streaming"]["sample_count"] == 3
    assert summary["10000:streaming"]["latency_seconds_median"] == 0.2
    assert summary["10000:streaming"]["latency_seconds_range"] == [0.1, 0.3]
    assert summary["50000:streaming"]["sample_count"] == 3
    assert summary["50000:streaming"]["latency_seconds_median"] == 2.0
    assert summary["50000:streaming"]["latency_seconds_range"] == [1.0, 3.0]


def test_summaries_skip_missing_and_non_numeric_optional_timings() -> None:
    samples = [
        {"dataset_size": 10_000, "mode": "streaming", "correctness": True},
        {
            "dataset_size": 10_000,
            "mode": "streaming",
            "correctness": True,
            "latency_seconds": "not numeric",
        },
    ]

    summary = summarize_samples_by_dataset(samples)["10000:streaming"]

    assert summary["sample_count"] == 2
    assert summary["latency_seconds_median"] is None
    assert summary["first_byte_latency_seconds_median"] is None


def test_json_object_boundary_preserves_optional_values_and_rejects_other_shapes() -> None:
    decoded = benchmark_export.load_json_object(b'{"optional": null, "count": 2}')

    assert decoded == {"optional": None, "count": 2}
    with pytest.raises(TypeError, match="must be an object"):
        _ = benchmark_export.load_json_object("[]")


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_timed_exporter_preserves_chunks_measurements_and_cleanup() -> None:
    class _FakeExporter:
        def __init__(self) -> None:
            self.closed: bool = False

        async def prepare(self, filters: EventExportFilters) -> PreparedCsvExport:
            del filters

            async def chunks() -> AsyncIterator[bytes]:
                yield b"first"
                yield b"second"

            async def close() -> None:
                self.closed = True

            return PreparedCsvExport(chunks(), close, deadline_at=123.0)

    delegate = _FakeExporter()
    measurements = Measurements(mode="streaming")
    prepared = await TimedExporter(delegate, measurements).prepare(EventExportFilters())

    observed_chunks = [chunk async for chunk in prepared.chunks]
    await prepared.aclose()

    assert observed_chunks == [b"first", b"second"]
    assert delegate.closed is True
    assert prepared.deadline_at == 123.0
    assert len(measurements.prepare_spans) == 1
    assert [span["index"] for span in measurements.chunk_pull_spans] == [0, 1]


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_benchmark_app_keeps_session_and_control_guards_before_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_name = "gw_events_bench_deadbeef"
    monkeypatch.setenv(
        "GW_DATABASE_URL",
        f"postgresql+asyncpg://postgres:synthetic@127.0.0.1:5432/{database_name}",
    )
    monkeypatch.setenv("GW_BENCHMARK_DATABASE_NAME", database_name)
    monkeypatch.setenv("GW_BENCHMARK_PASSWORD", "synthetic-password")
    monkeypatch.setenv("GW_BENCHMARK_CONTROL_TOKEN", "synthetic-control-token")
    monkeypatch.setenv("GW_BENCHMARK_MODE", "streaming")
    monkeypatch.setenv(
        "GW_BENCHMARK_SOURCE_DIR",
        str(Path(__file__).resolve().parents[1] / "src"),
    )

    user_action_policies: list[bool] = []

    def record_session_policy(
        auth: AuthCapability,
        config: ApiSettings,
        *,
        user_action: bool,
        mutation: bool = False,
    ) -> Callable[..., Awaitable[SessionAuthenticatedRequest]]:
        user_action_policies.append(user_action)
        return original_require_session(
            auth,
            config,
            user_action=user_action,
            mutation=mutation,
        )

    monkeypatch.setattr(benchmark_app, "require_session", record_session_policy)

    engines: list[AsyncEngine] = []

    def track_engine(
        database_url: str,
        *,
        pool_size: int,
        max_overflow: int,
        pool_pre_ping: bool,
    ) -> AsyncEngine:
        engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=pool_pre_ping,
        )
        engines.append(engine)
        return engine

    monkeypatch.setattr(benchmark_app, "create_async_engine", track_engine)
    prepare_calls = 0

    async def count_prepare(
        self: EventExportService,
        filters: EventExportFilters,
    ) -> PreparedCsvExport:
        nonlocal prepare_calls
        del self, filters
        prepare_calls += 1
        exception_message = "export preparation must not run before session authentication"
        raise AssertionError(exception_message)

    monkeypatch.setattr(EventExportService, "prepare", count_prepare)
    app = benchmark_app.build_app()
    try:
        assert False in user_action_policies
        assert await _asgi_status(app, "/__benchmark/ready") == 404
        assert (
            await _asgi_status(
                app,
                "/__benchmark/ready",
                headers={"X-Benchmark-Control": "incorrect-token"},
            )
            == 404
        )
        assert await _asgi_status(app, "/api/events/export.csv") == 401
        assert prepare_calls == 0
    finally:
        for engine in engines:
            await engine.dispose()


@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql+asyncpg://postgres:secret@db:5432/gods_watching_prod",
        "postgresql+asyncpg://postgres:secret@127.0.0.1:5432/gods_watching_test",
        "postgresql+asyncpg://postgres:secret@example.com:5432/gw_events_bench_deadbeef",
    ],
)
def test_non_disposable_or_nonlocal_database_targets_are_rejected(database_url: str) -> None:
    with pytest.raises(ValueError, match="disposable benchmark database"):
        ensure_disposable_database(database_url)


def test_disposable_database_name_can_be_bound_to_run_id() -> None:
    database_url = "postgresql+asyncpg://postgres:secret@db:5432/gw_events_bench_deadbeef"

    ensure_disposable_database(database_url, expected_database="gw_events_bench_deadbeef")

    with pytest.raises(ValueError, match="disposable benchmark database"):
        ensure_disposable_database(database_url, expected_database="gw_events_bench_cafebabe")


@pytest.mark.parametrize(
    ("query_option", "effective_key", "effective_value"),
    [
        ("host=example.invalid", "host", "example.invalid"),
        ("database=real_application_db", "database", "real_application_db"),
        ("user=application", "user", "application"),
    ],
)
def test_disposable_database_rejects_query_overrides_before_engine_creation(
    query_option: str,
    effective_key: str,
    effective_value: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_url = (
        "postgresql+asyncpg://postgres:synthetic@127.0.0.1:5432/"
        f"gw_events_bench_deadbeef?{query_option}"
    )
    _, effective_connect_options = PGDialect_asyncpg().create_connect_args(make_url(database_url))
    assert effective_connect_options[effective_key] == effective_value

    engine_creation_attempted = False

    def fail_if_engine_created(*engine_args: object, **engine_kwargs: object) -> None:
        nonlocal engine_creation_attempted
        assert engine_args
        assert engine_kwargs
        engine_creation_attempted = True
        raise AssertionError

    monkeypatch.setattr(benchmark_export, "create_async_engine", fail_if_engine_created)

    with pytest.raises(ValueError, match="disposable benchmark database"):
        _ = benchmark_export.make_engine(database_url)

    assert engine_creation_attempted is False
