from __future__ import annotations

import csv
import io
from datetime import UTC, datetime, timedelta, timezone
from uuid import UUID

import pytest
from sqlalchemy.dialects import postgresql

from gods_watching.contracts.events import EventExportFilters, EventExportRow
from gods_watching.events import EventRepository
from gods_watching.events.export import CSV_HEADER, encode_csv_row


def _row(camera_name: str = "North Hall") -> EventExportRow:
    return EventExportRow(
        id=37,
        occurred_at=datetime(2026, 10, 3, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),
        event_type="camera.updated",
        camera_id=UUID("00000000-0000-0000-0000-000000000037"),
        camera_name=camera_name,
    )


def test_event_statement_is_a_filtered_five_column_id_ordered_projection() -> None:
    camera_id = UUID("00000000-0000-0000-0000-000000000037")
    since = datetime(2026, 10, 1, tzinfo=UTC)
    until = datetime(2026, 10, 4, tzinfo=UTC)
    statement = EventRepository.statement(
        EventExportFilters(camera_id=camera_id, since=since, until=until)
    )
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert tuple(column.key for column in statement.selected_columns) == (
        "id",
        "occurred_at",
        "event_type",
        "camera_id",
        "camera_name",
    )
    assert "ORDER BY camera_events.id" in compiled
    assert "camera_events.camera_id =" in compiled
    assert "camera_events.occurred_at >=" in compiled
    assert "camera_events.occurred_at <" in compiled


def test_csv_header_has_the_contract_column_order() -> None:
    assert CSV_HEADER == b"id,occurred_at,event_type,camera_id,camera_name\r\n"


def test_csv_row_is_utf8_utc_and_round_trips_standard_quoting() -> None:
    row = _row('Café, "North"\nHall')

    encoded = encode_csv_row(row)
    cells = next(csv.reader(io.StringIO(encoded.decode("utf-8"))))

    assert cells == [
        "37",
        "2026-10-03T06:30:00Z",
        "camera.updated",
        "00000000-0000-0000-0000-000000000037",
        'Café, "North"\nHall',
    ]
    assert "rtsp://" not in encoded.decode("utf-8")
    assert "synthetic-token" not in encoded.decode("utf-8")


@pytest.mark.parametrize(
    "name",
    [
        "=1+1",
        "+SUM(A1:A2)",
        "-1+2",
        "@SUM(A1)",
        "\t=1+1",
        "\r=1+1",
        "\n=1+1",
        " =1+1",
        " \t@SUM(A1)",
    ],
)
def test_csv_row_prefixes_dangerous_spreadsheet_cells(name: str) -> None:
    encoded = encode_csv_row(_row(name))
    cells = next(csv.reader(io.StringIO(encoded.decode("utf-8"))))

    assert cells[4] == "'" + name


def test_export_filters_reject_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone"):
        _ = EventExportFilters(since=datetime.fromisoformat("2026-10-03T00:00:00"))


def test_export_filters_reject_an_inverted_time_range() -> None:
    with pytest.raises(ValueError, match="since"):
        _ = EventExportFilters(
            since=datetime(2026, 10, 4, tzinfo=UTC),
            until=datetime(2026, 10, 3, tzinfo=UTC),
        )
