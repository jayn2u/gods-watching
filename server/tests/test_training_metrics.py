from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gods_watching.contracts.training import TrainingMetric
from gods_watching.training.metrics import (
    MetricHistoryError,
    append_log,
    append_metric,
    read_log_page,
    read_metric_page,
    text_to_image_recall_at_k,
)

if TYPE_CHECKING:
    from pathlib import Path


def _metric(epoch: int, recall: float) -> TrainingMetric:
    return TrainingMetric(
        epoch=epoch,
        step=epoch * 10,
        training_loss=1.0 / epoch,
        validation_recall_at_1=recall,
        observed_at=datetime(2026, 10, 4, tzinfo=UTC),
    )


def test_text_to_image_recall_uses_stable_tie_breaks() -> None:
    image_embeddings = [[1.0, 0.0], [1.0, 0.0]]
    text_embeddings = [[1.0, 0.0], [0.0, 1.0]]

    score = text_to_image_recall_at_k(
        image_embeddings,
        text_embeddings,
        [7, 9],
        [9, 7],
        k=1,
    )

    assert score == 0.5


def test_text_to_image_recall_handles_native_validation_embedding_shape() -> None:
    image_embeddings = np.zeros((3078, 512), dtype=np.float32)
    text_embeddings = np.zeros((6158, 512), dtype=np.float32)
    image_embeddings[:, 0] = 1.0
    text_embeddings[:, 0] = 1.0
    image_ids = [8] * 3078
    image_ids[0] = 7
    text_ids = [7] * 6158

    score = text_to_image_recall_at_k(
        image_embeddings,
        text_embeddings,
        image_ids,
        text_ids,
        k=1,
    )

    assert score == 1.0


def test_metric_and_log_pages_are_append_only_and_cursor_bounded(tmp_path: Path) -> None:
    metric_path = tmp_path / "metrics.jsonl"
    log_path = tmp_path / "logs.jsonl"
    append_metric(metric_path, _metric(1, 0.2))
    append_metric(metric_path, _metric(2, 0.4))
    append_log(log_path, level="info", message="epoch one\ntraining details")
    append_log(log_path, level="info", message="epoch two")

    first_metrics = read_metric_page(metric_path, cursor=None, limit=1)
    second_metrics = read_metric_page(metric_path, cursor=first_metrics.next_cursor, limit=1)
    first_logs = read_log_page(log_path, cursor=None, limit=1)
    second_logs = read_log_page(log_path, cursor=first_logs.next_cursor, limit=1)

    assert [first_metrics.items[0].epoch, second_metrics.items[0].epoch] == [1, 2]
    assert first_metrics.next_cursor is not None
    assert first_logs.items[0].message == "epoch one training details"
    assert second_logs.items[0].message == "epoch two"
    assert second_logs.next_cursor is None


def test_metric_log_history_rejects_symlinks_and_bad_cursors(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    _ = target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "metrics.jsonl"
    link.symlink_to(target)

    with pytest.raises(MetricHistoryError, match="cannot be opened"):
        _ = read_metric_page(link, cursor=None, limit=10)
    with pytest.raises(ValueError, match="cursor"):
        _ = read_metric_page(target, cursor="bad", limit=10)


def test_incomplete_final_metric_record_is_hidden_and_recovered_on_append(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    append_metric(path, _metric(1, 0.2))
    with path.open("ab") as output:
        _ = output.write(b'{"epoch":')

    visible = read_metric_page(path, cursor=None, limit=10)
    assert [metric.epoch for metric in visible.items] == [1]
    assert visible.next_cursor is None

    append_metric(path, _metric(2, 0.4))
    repaired = read_metric_page(path, cursor=None, limit=10)
    assert [metric.epoch for metric in repaired.items] == [1, 2]


def test_complete_malformed_metric_record_fails_with_a_bounded_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metrics.jsonl"
    _ = path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(MetricHistoryError, match="invalid JSON"):
        _ = read_metric_page(path, cursor=None, limit=10)
