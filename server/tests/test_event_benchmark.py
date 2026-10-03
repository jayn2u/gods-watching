"""Correctness gates for the actual HTTP export benchmark."""

from __future__ import annotations

import csv
import subprocess
from types import SimpleNamespace
from pathlib import Path

import pytest
from sqlalchemy.dialects.postgresql.asyncpg import PGDialect_asyncpg
from sqlalchemy.engine import make_url

import qa.events.benchmark_export as benchmark_export
from qa.events.benchmark_export import (
    BenchmarkRunner,
    ensure_disposable_database,
    _summarize_isolation,
    summarize_samples,
    summarize_samples_by_dataset,
    validate_csv_file,
)


_FIELDS = ("id", "occurred_at", "event_type", "camera_id", "camera_name")
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
    invalid.write_text("id,event_type\n1,camera.created\n", encoding="utf-8")
    invalid_result = validate_csv_file(invalid, _EXPECTED)
    assert invalid_result.correct is False
    assert invalid_result.parse_error is not None
    assert invalid_result.performance_eligible is False


def test_csv_row_with_surplus_column_fails_correctness(tmp_path: Path) -> None:
    output = tmp_path / "extra-column.csv"
    output.write_text(
        "id,occurred_at,event_type,camera_id,camera_name\n"
        "1,2026-10-03T00:00:00Z,camera.created,"
        "00000000-0000-0000-0000-000000000001,Front door,unexpected\n"
        "2,2026-10-03T00:00:01Z,camera.updated,"
        '00000000-0000-0000-0000-000000000002,"Parking, east"\n',
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

    summary = _summarize_isolation([valid, invalid])["10000:READ COMMITTED"]

    assert summary["trial_count"] == 2
    assert summary["valid_trial_count"] == 1
    assert summary["invalid_trial_count"] == 1
    assert summary["elapsed_seconds_median"] == 0.2


def test_log_capture_keeps_stdout_and_stderr_while_structured_docker_keeps_stdout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="structured stdout", stderr="server diagnostics")

    monkeypatch.setattr(benchmark_export.subprocess, "run", fake_run)
    capture = getattr(benchmark_export, "_docker_logs", None)
    assert callable(capture), "log-specific capture helper should be defined"

    logs = capture("db", tail=20)

    assert "structured stdout" in logs
    assert "server diagnostics" in logs
    assert "stdout" in logs.lower() and "stderr" in logs.lower()
    assert calls[-1][:3] == ["docker", "logs", "--tail"]
    assert benchmark_export._docker(["ps"]) == "structured stdout"


def _timeout_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fail_remove: bool) -> tuple[BenchmarkRunner, list[str]]:
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.run_id = "deadbeef"
    runner.network_name = "gw-event-export-deadbeef"
    runner.scratch_root = tmp_path
    runner.app_database_url = "postgresql+asyncpg://postgres:secret@db:5432/gw_events_bench_deadbeef"
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
                raise RuntimeError("docker rm failed: daemon unavailable")
            return name
        raise AssertionError(f"unexpected docker command: {arguments[:2]}")

    monkeypatch.setattr(runner, "_run_docker", fake_run_docker)
    return runner, calls


def test_timed_out_utility_is_registered_and_force_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, calls = _timeout_runner(tmp_path, monkeypatch, fail_remove=False)

    with pytest.raises(subprocess.TimeoutExpired):
        runner._run_utility(suffix="timeout", command=("seed",), timeout=1)

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
        runner._run_utility(suffix="timeout", command=("seed",), timeout=1)

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
    runner.operator_password = "synthetic-password"
    runner.control_token = "synthetic-control-token"

    with pytest.raises(subprocess.TimeoutExpired):
        runner._run_client(
            app_container="gw-event-export-app-deadbeef-streaming-10000-1",
            mode="streaming",
            row_count=10_000,
            repeat=1,
            writer_enabled=True,
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
    _, effective_connect_options = PGDialect_asyncpg().create_connect_args(
        make_url(database_url)
    )
    assert effective_connect_options[effective_key] == effective_value

    engine_creation_attempted = False

    def fail_if_engine_created(*engine_args: object, **engine_kwargs: object) -> None:
        nonlocal engine_creation_attempted
        assert engine_args and engine_kwargs
        engine_creation_attempted = True
        raise AssertionError

    monkeypatch.setattr(benchmark_export, "create_async_engine", fail_if_engine_created)

    with pytest.raises(ValueError, match="disposable benchmark database"):
        _ = benchmark_export._make_engine(database_url)

    assert engine_creation_attempted is False
