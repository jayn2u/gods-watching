"""Atomic, public-safe readiness shared by the training supervisor and API."""

# ruff: noqa: TRY003, EM101, PLR0911

from __future__ import annotations

import json
import os
import stat
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from gods_watching.contracts.training import TrainingSupervisorStatus

_STATUS_MAX_BYTES = 4096
_STATUS_MAX_AGE = timedelta(seconds=15)
_STATUS_FILENAME = "supervisor-status.json"


def supervisor_status_path(training_root: Path) -> Path:
    """Return the only shared status file beneath the configured run root."""
    return training_root / _STATUS_FILENAME


def write_supervisor_status(
    training_root: Path,
    status: TrainingSupervisorStatus,
) -> None:
    """Atomically publish one bounded readiness record without exposing host data."""
    training_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(status.model_dump(mode="json"), allow_nan=False, sort_keys=True)
    if len(payload.encode("utf-8")) > _STATUS_MAX_BYTES:
        raise ValueError("training supervisor status exceeds its size bound")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=training_root,
        prefix=".supervisor-status-",
        suffix=".partial",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as temporary:
            _ = temporary.write(payload)
            _ = temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        _ = temporary_path.replace(supervisor_status_path(training_root))
        directory_fd = os.open(training_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_supervisor_status(
    training_root: Path,
    *,
    source_fingerprint: str,
    dataset_fingerprint: str | None,
    now: datetime | None = None,
) -> TrainingSupervisorStatus:
    """Read only a fresh status matching this process' source and dataset identity."""
    unavailable = TrainingSupervisorStatus(
        state="unavailable",
        reason="training_supervisor_unavailable",
    )
    path = supervisor_status_path(training_root)
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size > _STATUS_MAX_BYTES:
            return unavailable
        status = TrainingSupervisorStatus.model_validate_json(path.read_bytes())
    except (OSError, ValueError):
        return unavailable

    if status.observed_at is None:
        return unavailable
    current_time = now or datetime.now(UTC)
    if status.observed_at.utcoffset() is None or current_time.utcoffset() is None:
        return unavailable
    age = current_time - status.observed_at
    if age < timedelta(seconds=-5) or age > _STATUS_MAX_AGE:
        return TrainingSupervisorStatus(
            state="unavailable",
            reason="training_supervisor_status_stale",
            observed_at=status.observed_at,
            source_fingerprint=status.source_fingerprint,
            dataset_fingerprint=status.dataset_fingerprint,
        )
    if status.source_fingerprint != source_fingerprint:
        return TrainingSupervisorStatus(
            state="unavailable",
            reason="training_supervisor_source_changed",
            observed_at=status.observed_at,
            source_fingerprint=status.source_fingerprint,
            dataset_fingerprint=status.dataset_fingerprint,
        )
    if dataset_fingerprint is not None and status.dataset_fingerprint != dataset_fingerprint:
        return TrainingSupervisorStatus(
            state="unavailable",
            reason="training_supervisor_dataset_changed",
            observed_at=status.observed_at,
            source_fingerprint=status.source_fingerprint,
            dataset_fingerprint=status.dataset_fingerprint,
        )
    return status


__all__ = ["read_supervisor_status", "supervisor_status_path", "write_supervisor_status"]
