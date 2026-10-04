"""Fixed gated training child entrypoint; Torch is imported only after ownership."""

# ruff: noqa: TRY003, EM101, EM102

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

from gods_watching.contracts.training import TrainingConfig, TrainingMetric
from gods_watching.storage import Database
from gods_watching.training.evaluation import EvaluationCancelledError
from gods_watching.training.metrics import append_log, append_metric
from gods_watching.training.models import TrainingJob, TrainingPhase
from gods_watching.training.repository import TrainingRepository, TrainingRequestLeaseLostError
from gods_watching.training.settings import TrainingSettings
from gods_watching.training.supervisor import process_start_time

if TYPE_CHECKING:
    from typing import BinaryIO

    from gods_watching.training.publishing import TrainingJobIdentity

_MAX_SAFE_ERROR_CLASS_LENGTH = 80


class TrainingChildIdentityError(RuntimeError):
    """The child no longer matches the supervisor's persisted process identity."""


@dataclass(frozen=True, slots=True)
class _RunnerArguments:
    job_id: UUID
    owner_generation: int
    wait_for_owner: bool


def ownership_gate_is_open(stdin: BinaryIO) -> bool:
    """Accept only the one fixed supervisor release record before engine imports."""
    return stdin.readline(32) == b"owned\n"


class _RunnerReporter:
    _database: Database
    _repository: TrainingRepository
    _job_id: UUID
    _generation: int
    _run_directory: Path
    _loop: asyncio.AbstractEventLoop
    _pid: int
    _start_time: int
    _epoch: int
    _step: int

    def __init__(  # noqa: PLR0913
        self,
        database: Database,
        repository: TrainingRepository,
        job: TrainingJob,
        run_directory: Path,
        loop: asyncio.AbstractEventLoop,
        *,
        pid: int,
        start_time: int,
    ) -> None:
        self._database = database
        self._repository = repository
        self._job_id = job.id
        self._generation = job.owner_generation
        self._run_directory = run_directory
        self._loop = loop
        self._pid = pid
        self._start_time = start_time
        self._epoch = job.current_epoch
        self._step = job.current_step

    def progress(self, row: dict[str, object]) -> None:
        epoch = _non_negative_int(row.get("epoch"), field="epoch")
        step = _non_negative_int(row.get("optimizer_step"), field="optimizer step")
        self._epoch = epoch
        self._step = step
        self._persist_progress()

    def metric(self, metric: TrainingMetric) -> None:
        self._epoch = metric.epoch
        self._step = metric.step
        future = asyncio.run_coroutine_threadsafe(self._append_metric(metric), self._loop)
        if not future.result(timeout=30):
            raise TrainingRequestLeaseLostError("training child lost its owner generation")

    def checkpoint(self, path: Path, best_metric: float) -> None:
        if not path.is_absolute() or not path.is_relative_to(self._run_directory):
            raise ValueError("checkpoint path escaped the job run directory")
        self._persist_progress(checkpoint_path=path, best_metric=best_metric)

    async def log(self, level: str, message: str) -> None:
        if level not in {"debug", "info", "warning", "error"}:
            level = "info"
        async with self._database.transaction() as session:
            current = await self._repository.report_training_progress(
                session,
                self._job_id,
                owner_generation=self._generation,
                pid=self._pid,
                start_time=self._start_time,
                epoch=self._epoch,
                step=self._step,
            )
            if not current:
                raise TrainingRequestLeaseLostError("training child lost its owner generation")
            append_log(
                self._run_directory / "logs.jsonl",
                level=cast("Literal['debug', 'info', 'warning', 'error']", level),
                message=message,
            )

    async def _append_metric(self, metric: TrainingMetric) -> bool:
        async with self._database.transaction() as session:
            current = await self._repository.report_training_progress(
                session,
                self._job_id,
                owner_generation=self._generation,
                pid=self._pid,
                start_time=self._start_time,
                epoch=self._epoch,
                step=self._step,
            )
            if not current:
                return False
            append_metric(self._run_directory / "metrics.jsonl", metric)
            return True

    def _persist_progress(
        self,
        *,
        checkpoint_path: Path | None = None,
        best_metric: float | None = None,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            self._write_progress(
                checkpoint_path=checkpoint_path,
                best_metric=best_metric,
            ),
            self._loop,
        )
        if not future.result(timeout=30):
            raise TrainingRequestLeaseLostError("training child lost its owner generation")

    async def _write_progress(
        self,
        *,
        checkpoint_path: Path | None,
        best_metric: float | None,
    ) -> bool:
        async with self._database.transaction() as session:
            return await self._repository.report_training_progress(
                session,
                self._job_id,
                owner_generation=self._generation,
                pid=self._pid,
                start_time=self._start_time,
                epoch=self._epoch,
                step=self._step,
                checkpoint_path=checkpoint_path,
                best_metric=best_metric,
            )


async def run_training_child(  # noqa: C901, PLR0911, PLR0912, PLR0915
    job_id: UUID,
    owner_generation: int,
) -> int:
    """Validate the released child, then train without ending durable jobs early."""
    database_url = os.environ.get("GW_DATABASE_URL")
    if not database_url:
        raise RuntimeError("training worker requires GW_DATABASE_URL")
    database = Database.connect(database_url)
    repository = TrainingRepository()
    settings = TrainingSettings()
    pid = os.getpid()
    start_time = process_start_time(pid)
    reporter: _RunnerReporter | None = None
    try:
        async with database.transaction() as session:
            job = await repository.get_job(session, job_id)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.child_pid != pid
            or job.child_start_time != start_time
        ):
            raise TrainingChildIdentityError(  # noqa: TRY301
                "runner PID/start-time does not own this job"
            )
        if job.phase == TrainingPhase.CANCELLING.value or job.cancel_requested:
            return 0
        if job.phase != TrainingPhase.STARTING.value:
            raise TrainingChildIdentityError(  # noqa: TRY301
                "training job is not waiting at the child gate"
            )
        dataset_root = settings.require_dataset_root()
        run_directory = settings.training_root / "jobs" / str(job.id)
        run_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        async with database.transaction() as session:
            started = await repository.mark_training_started(
                session,
                job.id,
                owner_generation=owner_generation,
                pid=pid,
                start_time=start_time,
            )
        if not started:
            return 0

        from gods_watching.training.engine import (  # noqa: PLC0415
            TrainingPaths,
            TrainingRunSnapshot,
            run_training,
        )

        config = TrainingConfig.model_validate(job.config_snapshot)
        snapshot = TrainingRunSnapshot(
            job_id=job.id,
            owner_generation=owner_generation,
            config=config,
            dataset_fingerprint=job.dataset_fingerprint,
            source_fingerprint=job.source_fingerprint,
            current_epoch=job.current_epoch,
            best_metric=job.best_metric,
            checkpoint_path=Path(job.checkpoint_path) if job.checkpoint_path else None,
        )
        paths = TrainingPaths(
            run_directory=run_directory,
            dataset_root=dataset_root,
            model_root=Path(os.environ.get("GW_TRAINING_MODEL_ROOT", "/models/clip")),
            model_lock_path=settings.model_lock_path,
        )
        loop = asyncio.get_running_loop()
        reporter = _RunnerReporter(
            database,
            repository,
            job,
            run_directory,
            loop,
            pid=pid,
            start_time=start_time,
        )
        cancellation = threading.Event()
        monitor_stop = asyncio.Event()
        monitor = asyncio.create_task(
            _monitor_cancel(
                database,
                repository,
                job.id,
                owner_generation,
                cancellation,
                monitor_stop,
            )
        )
        _install_shutdown_signal_handlers(loop, cancellation)
        try:
            result = await asyncio.to_thread(
                run_training,
                snapshot,
                paths,
                reporter,
                cancellation,
            )
            if result.cancelled:
                await reporter.log("info", "training cancellation reached an optimizer boundary")
                return 0
            if not result.completed:
                await reporter.log("error", "training engine did not complete")
                return 1
            if result.best_checkpoint is None:
                raise RuntimeError("completed training run has no validation-best checkpoint")

            async with database.transaction() as session:
                staged = await repository.mark_engine_staging(
                    session,
                    job.id,
                    owner_generation=owner_generation,
                    pid=pid,
                    start_time=start_time,
                )
            if not staged:
                return 0 if await _is_cancellation_requested(
                    database, repository, job.id, owner_generation
                ) else 1
            await reporter.log(
                "info",
                "training completed; scoring validation-best weights on test",
            )

            from gods_watching.training.evaluation import evaluate_best_checkpoint  # noqa: PLC0415
            from gods_watching.training.publishing import publish_candidate  # noqa: PLC0415

            evaluation_report = await asyncio.to_thread(
                evaluate_best_checkpoint,
                snapshot,
                paths,
                result.best_checkpoint,
                cancellation=cancellation,
            )
            if cancellation.is_set():
                await reporter.log("info", "final evaluation stopped after cancellation")
                return 0

            async with database.transaction() as session:
                publishing = await repository.mark_publishing(
                    session,
                    job.id,
                    owner_generation=owner_generation,
                    pid=pid,
                    start_time=start_time,
                )
            if not publishing:
                return 0 if await _is_cancellation_requested(
                    database, repository, job.id, owner_generation
                ) else 1

            candidate = await asyncio.to_thread(
                publish_candidate,
                cast("TrainingJobIdentity", cast("object", job)),
                result.best_checkpoint,
                evaluation_report,
                settings.model_assets_root,
            )
            if cancellation.is_set():
                await reporter.log(
                    "info",
                    "publication completed after cancellation; candidate is uncommitted",
                )
                return 0
            async with database.transaction() as session:
                recorded = await repository.record_candidate_publication(
                    session,
                    job.id,
                    owner_generation=owner_generation,
                    pid=pid,
                    start_time=start_time,
                    candidate_model_id=candidate.model_id,
                    candidate_revision=candidate.revision,
                    evaluation=candidate.evaluation,
                )
            if not recorded:
                return 0 if await _is_cancellation_requested(
                    database, repository, job.id, owner_generation
                ) else 1
            await reporter.log("info", "candidate package and held-out summary committed")
            return 0
        finally:
            monitor_stop.set()
            _ = monitor.cancel()
            with suppress(asyncio.CancelledError):
                _ = await monitor
    except EvaluationCancelledError:
        # The evaluator checks the durable cancel flag at bounded batch edges.
        # Treat that cooperative stop as a clean child exit so confirmed process
        # exit can finalize the durable job as cancelled.
        return 0
    except Exception as error:  # noqa: BLE001
        if reporter is not None:
            await reporter.log("error", _safe_error_message(error))
            await _record_child_error(
                database,
                repository,
                job_id,
                owner_generation,
                pid,
                start_time,
                error,
            )
        return 1
    finally:
        await database.close()


async def _monitor_cancel(  # noqa: PLR0913
    database: Database,
    repository: TrainingRepository,
    job_id: UUID,
    owner_generation: int,
    cancellation: threading.Event,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        async with database.transaction() as session:
            job = await repository.get_job(session, job_id)
        if (
            job is None
            or job.owner_generation != owner_generation
            or job.cancel_requested
            or job.phase == TrainingPhase.CANCELLING.value
            or TrainingPhase(job.phase).terminal
        ):
            cancellation.set()
            return
        try:
            _ = await asyncio.wait_for(stop.wait(), timeout=0.25)
        except TimeoutError:
            continue


async def _record_child_error(  # noqa: PLR0913
    database: Database,
    repository: TrainingRepository,
    job_id: UUID,
    owner_generation: int,
    pid: int,
    start_time: int,
    error: Exception,
) -> None:
    async with database.transaction() as session:
        _ = await repository.report_training_error(
            session,
            job_id,
            owner_generation=owner_generation,
            pid=pid,
            start_time=start_time,
            error=_safe_error_message(error),
        )


async def _is_cancellation_requested(
    database: Database,
    repository: TrainingRepository,
    job_id: UUID,
    owner_generation: int,
) -> bool:
    """Read durable cancellation after a fenced stage transition lost a race."""
    async with database.transaction() as session:
        job = await repository.get_job(session, job_id)
    return (
        job is None
        or job.owner_generation != owner_generation
        or job.cancel_requested
        or job.phase == TrainingPhase.CANCELLING.value
        or TrainingPhase(job.phase).terminal
    )


def _safe_error_message(error: Exception) -> str:
    known_messages = {
        "TrainingOutOfMemoryError": "training ran out of GPU memory",
        "TrainingNumericalError": "training stopped after a numerical error",
        "TrainingEngineError": "training engine failed",
        "CheckpointSpaceError": "training checkpoint storage is full",
        "CheckpointError": "training checkpoint failed validation",
        "DatasetValidationError": "training dataset failed validation",
        "TrainingRequestLeaseLostError": "training ownership changed",
    }
    error_type = type(error).__name__
    safe_message = known_messages.get(error_type)
    if safe_message is not None:
        return safe_message
    if error_type.isidentifier() and len(error_type) <= _MAX_SAFE_ERROR_CLASS_LENGTH:
        return error_type
    return "TrainingError"


def _non_negative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"runner progress {field} is invalid")
    return value


def _install_shutdown_signal_handlers(
    loop: asyncio.AbstractEventLoop,
    cancellation: threading.Event,
) -> None:
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, cancellation.set)
        except NotImplementedError:
            continue


def _arguments(argv: list[str] | None = None) -> _RunnerArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--job-id", type=UUID, required=True)
    _ = parser.add_argument("--owner-generation", type=int, required=True)
    _ = parser.add_argument("--wait-for-owner", action="store_true", required=True)
    parsed = parser.parse_args(argv)
    job_id: object = cast("object", parsed.job_id)
    owner_generation: object = cast("object", parsed.owner_generation)
    wait_for_owner: object = cast("object", parsed.wait_for_owner)
    if (
        not isinstance(job_id, UUID)
        or type(owner_generation) is not int
        or type(wait_for_owner) is not bool
    ):
        raise ValueError("training runner arguments are malformed")
    return _RunnerArguments(job_id, owner_generation, wait_for_owner)


def main(argv: list[str] | None = None) -> int:
    """Wait at the inherited stdin gate before opening the GPU engine module."""
    args = _arguments(argv)
    if not ownership_gate_is_open(sys.stdin.buffer):
        return 1
    return asyncio.run(run_training_child(args.job_id, args.owner_generation))


__all__ = ["main", "ownership_gate_is_open", "run_training_child"]


if __name__ == "__main__":
    raise SystemExit(main())
