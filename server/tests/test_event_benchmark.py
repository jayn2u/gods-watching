"""Correctness gates for the actual HTTP export benchmark."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from qa.events.benchmark_export import (
    ensure_disposable_database,
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
