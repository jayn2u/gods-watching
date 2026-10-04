"""Durable redacted metrics/logs and deterministic text-to-image validation recall."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import base64
import fcntl
import importlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast

from gods_watching.contracts.training import (
    TrainingLogEntry,
    TrainingLogPage,
    TrainingMetric,
    TrainingMetricPage,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO


class _NumpyArray(Protocol):
    """Array operations used by the optional NumPy recall calculation."""

    @property
    def shape(self) -> Sequence[int]: ...

    @property
    def ndim(self) -> int: ...

    def transpose(self) -> _NumpyArray: ...

    def __getitem__(self, index: object) -> _NumpyArray: ...

    def __matmul__(self, other: _NumpyArray) -> _NumpyArray: ...

    def __neg__(self) -> _NumpyArray: ...

    def all(self) -> bool: ...

    def any(self, *, axis: int) -> _NumpyArray: ...

    def sum(self) -> int | float: ...


class _NumpyApi(Protocol):
    """Small runtime surface for lazily imported NumPy."""

    float32: object

    def asarray(self, value: object, *, dtype: object = ...) -> _NumpyArray: ...

    def isfinite(self, values: _NumpyArray) -> _NumpyArray: ...

    def equal(self, first: _NumpyArray, second: _NumpyArray) -> _NumpyArray: ...

    def argsort(
        self,
        values: _NumpyArray,
        *,
        axis: int,
        kind: str,
    ) -> _NumpyArray: ...


class _TorchBackedArray(Protocol):
    """Tensor methods used when callers pass Torch embeddings directly."""

    def detach(self) -> _TorchBackedArray: ...

    def cpu(self) -> _TorchBackedArray: ...

    def numpy(self) -> object: ...

_MAX_PAGE_SIZE = 200
_MAX_LOG_MESSAGE = 500
_RECALL_QUERY_CHUNK_SIZE = 128
_EMBEDDING_MATRIX_RANK = 2
_APPEND_FLAGS = os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK


class MetricHistoryError(RuntimeError):
    """A durable metric/log record could not be safely read or appended."""


class MetricCursorError(ValueError):
    """A page cursor or requested history page size is malformed."""


def append_metric(path: Path, metric: TrainingMetric) -> None:
    """Append and fsync one safe epoch metric record."""
    _append_json_line(path, metric.model_dump(mode="json"))


def append_log(
    path: Path,
    *,
    level: Literal["debug", "info", "warning", "error"],
    message: str,
    observed_at: datetime | None = None,
) -> None:
    """Append one bounded log entry with control characters removed."""
    clean_message = " ".join(message.split())[:_MAX_LOG_MESSAGE]
    if not clean_message:
        return
    entry = TrainingLogEntry(
        cursor="pending",
        level=level,
        message=clean_message,
        observed_at=observed_at or datetime.now(UTC),
    )
    _append_json_line(path, entry.model_dump(mode="json", exclude={"cursor"}))


def read_metric_page(
    path: Path,
    *,
    cursor: str | None,
    limit: int,
) -> TrainingMetricPage:
    """Read an append-only metric page using a line-number keyset cursor."""
    records, next_cursor = _read_page(path, cursor=cursor, limit=limit)
    try:
        items = tuple(TrainingMetric.model_validate(record) for _line, record in records)
    except ValueError as error:
        raise MetricHistoryError("metric history row does not match its contract") from error
    return TrainingMetricPage(items=items, next_cursor=next_cursor)


def read_log_page(
    path: Path,
    *,
    cursor: str | None,
    limit: int,
) -> TrainingLogPage:
    """Read a bounded log page while exposing only the safe log contract."""
    records, next_cursor = _read_page(path, cursor=cursor, limit=limit)
    try:
        items = tuple(
            TrainingLogEntry.model_validate({**record, "cursor": _encode_cursor(line)})
            for line, record in records
        )
    except ValueError as error:
        raise MetricHistoryError("log history row does not match its contract") from error
    return TrainingLogPage(items=items, next_cursor=next_cursor)


def text_to_image_recall_at_k(
    image_embeddings: object,
    text_embeddings: object,
    image_ids: Sequence[int],
    text_ids: Sequence[int],
    *,
    k: int,
) -> float:
    """Return macro text-query recall with stable gallery-index tie breaking."""
    if type(k) is not int or k < 1:
        raise ValueError("recall k must be a positive integer")
    numpy = cast("_NumpyApi", cast("object", importlib.import_module("numpy")))
    images = _embedding_matrix(image_embeddings, numpy)
    texts = _embedding_matrix(text_embeddings, numpy)
    if (
        images.shape[0] == 0
        or texts.shape[0] == 0
        or images.shape[0] != len(image_ids)
        or texts.shape[0] != len(text_ids)
    ):
        raise ValueError("embedding and identity counts must be non-empty and aligned")
    if images.shape[1] < 1 or images.shape[1] != texts.shape[1]:
        raise ValueError("image and text embedding dimensions must match")
    gallery_ids = numpy.asarray(image_ids)
    query_ids = numpy.asarray(text_ids)
    correct = 0
    top_k = min(k, images.shape[0])
    for start in range(0, texts.shape[0], _RECALL_QUERY_CHUNK_SIZE):
        stop = min(start + _RECALL_QUERY_CHUNK_SIZE, texts.shape[0])
        scores = texts[start:stop] @ images.transpose()
        # Stable sorting preserves ascending gallery index for tied similarity.
        ranked = numpy.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        matches = numpy.equal(
            gallery_ids[ranked],
            query_ids[start:stop, None],
        ).any(axis=1)
        correct += int(matches.sum())
    return correct / texts.shape[0]


def _embedding_matrix(value: object, numpy: _NumpyApi) -> _NumpyArray:
    if hasattr(value, "detach"):
        value = cast("_TorchBackedArray", value).detach().cpu().numpy()
    matrix = numpy.asarray(value, dtype=numpy.float32)
    if matrix.ndim != _EMBEDDING_MATRIX_RANK:
        raise ValueError("embeddings must be a two-dimensional sequence")
    if not bool(numpy.isfinite(matrix).all()):
        raise ValueError("embeddings must contain only finite values")
    return matrix


def _append_json_line(path: Path, payload: Mapping[str, object]) -> None:
    destination = Path(path)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    encoded = json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":"))
    data = (encoded + "\n").encode("utf-8")
    descriptor = os.open(destination, _APPEND_FLAGS, 0o600)
    with os.fdopen(descriptor, "r+b") as output:
        fcntl.flock(output.fileno(), fcntl.LOCK_EX)
        _truncate_partial_tail(output)
        _ = output.seek(0, os.SEEK_END)
        _ = output.write(data)
        output.flush()
        os.fsync(output.fileno())
        fcntl.flock(output.fileno(), fcntl.LOCK_UN)


def _read_page(  # noqa: C901
    path: Path,
    *,
    cursor: str | None,
    limit: int,
) -> tuple[list[tuple[int, dict[str, object]]], str | None]:
    if type(limit) is not int or not 1 <= limit <= _MAX_PAGE_SIZE:
        raise MetricCursorError("metric/log page size is outside the allowed range")
    after_line = _decode_cursor(cursor) if cursor is not None else 0
    file_path = Path(path)
    try:
        descriptor = os.open(file_path, _READ_FLAGS)
    except FileNotFoundError:
        return [], None
    except OSError as error:
        raise MetricHistoryError("metric/log history file cannot be opened") from error
    selected: list[tuple[int, dict[str, object]]] = []
    try:
        with os.fdopen(descriptor, "r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.endswith("\n"):
                    # A crash may leave one incomplete append at EOF. Do not expose
                    # it or advance the cursor; a later writer truncates that tail.
                    break
                if line_number <= after_line or not line.strip():
                    continue
                try:
                    decoded = cast("object", json.loads(line))
                except json.JSONDecodeError as error:
                    raise MetricHistoryError("metric/log history contains invalid JSON") from error
                if not isinstance(decoded, dict):
                    raise MetricHistoryError("metric/log history row is not an object")
                selected.append((line_number, cast("dict[str, object]", decoded)))
                if len(selected) > limit:
                    break
    except UnicodeDecodeError as error:
        raise MetricHistoryError("metric/log history is not valid UTF-8") from error
    except OSError as error:
        raise MetricHistoryError("metric/log history cannot be read") from error
    next_cursor = _encode_cursor(selected[limit - 1][0]) if len(selected) > limit else None
    return selected[:limit], next_cursor


def _truncate_partial_tail(file: BinaryIO) -> None:
    _ = file.seek(0, os.SEEK_END)
    size = file.tell()
    if size == 0:
        return
    _ = file.seek(size - 1)
    if file.read(1) == b"\n":
        return
    remaining = size
    while remaining:
        chunk_start = max(0, remaining - 4096)
        _ = file.seek(chunk_start)
        chunk = file.read(remaining - chunk_start)
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            _ = file.truncate(chunk_start + newline + 1)
            return
        remaining = chunk_start
    _ = file.truncate(0)


def _encode_cursor(line_number: int) -> str:
    return base64.urlsafe_b64encode(str(line_number).encode("ascii")).decode("ascii").rstrip("=")


def _decode_cursor(cursor: str) -> int:
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = int(base64.urlsafe_b64decode(padded).decode("ascii"))
    except (ValueError, UnicodeDecodeError) as error:
        raise MetricCursorError("metric/log cursor is malformed") from error
    if value < 1:
        raise MetricCursorError("metric/log cursor is malformed")
    return value


__all__ = [
    "MetricCursorError",
    "MetricHistoryError",
    "append_log",
    "append_metric",
    "read_log_page",
    "read_metric_page",
    "text_to_image_recall_at_k",
]
