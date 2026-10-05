"""The isolated GPU training loop and its deterministic optimizer-step recipe."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from gods_watching.contracts.training import TrainingConfig, TrainingMetric
from gods_watching.training.calibration import optimizer_update_succeeded
from gods_watching.training.checkpoints import (
    build_training_checkpoint,
    load_checkpoint_verified,
    model_state_cpu,
    restore_training_checkpoint,
    save_checkpoint_atomic,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from uuid import UUID

    from gods_watching.training.engine_api import (
        TrainingBackend,
        TrainingDevice,
        TrainingModel,
        TrainingOptimizer,
        TrainingScaler,
        TrainingScheduler,
    )

_MAX_CONSECUTIVE_SKIPPED_STEPS = 16
_TRAINING_RECIPE_BETAS = (0.9, 0.999)
_TRAINING_RECIPE_EPSILON = 1e-8


class TrainingEngineError(RuntimeError):
    """Base class for a stopped or invalid training run."""


class TrainingNumericalError(TrainingEngineError):
    """A non-finite training or validation value made safe updates impossible."""


class TrainingOutOfMemoryError(TrainingEngineError):
    """The admitted worker ran out of device memory during model training."""


class _TrainingCancelledBeforeInitializationError(Exception):
    """A cancellation arrived after CPU model loading but before CUDA allocation."""


@dataclass(frozen=True, slots=True)
class TrainingRunSnapshot:
    """Immutable database inputs passed to the isolated training child."""

    job_id: UUID
    owner_generation: int
    config: TrainingConfig
    dataset_fingerprint: str
    source_fingerprint: str
    current_epoch: int = 0
    best_metric: float | None = None
    checkpoint_path: Path | None = None

    def checkpoint_identity(self) -> dict[str, object]:
        """Return only immutable identities required to resume this exact run."""
        return {
            "config_snapshot": cast("dict[str, object]", self.config.model_dump(mode="json")),
            "dataset_fingerprint": self.dataset_fingerprint,
            "source_fingerprint": self.source_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TrainingPaths:
    """Server-selected paths; no run or model path comes from a request."""

    run_directory: Path
    dataset_root: Path
    model_root: Path = Path("/models/clip")
    model_lock_path: Path = Path("/opt/gods-watching/assets/models.lock.json")

    @property
    def last_checkpoint(self) -> Path:
        """Return the latest complete epoch checkpoint destination."""
        return self.run_directory / "last.pt"

    @property
    def best_checkpoint(self) -> Path:
        """Return the best validation checkpoint destination."""
        return self.run_directory / "best.pt"


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """Already-tokenized image/text inputs for one identity-distinct micro-batch."""

    model_inputs: Mapping[str, object]


class TrainingCancellation(Protocol):
    """Thread-safe cooperative cancellation signal supplied by the runner."""

    def is_set(self) -> bool:
        """Return whether the operator asked this run to stop."""
        ...


class TrainingReporter(Protocol):
    """Small reporting seam used by the async database/file reporter."""

    def progress(self, row: dict[str, object]) -> None:
        """Publish transient epoch/micro-batch progress."""
        ...

    def metric(self, metric: TrainingMetric) -> None:
        """Persist one completed epoch's durable metric record."""
        ...

    def checkpoint(self, path: Path, best_metric: float) -> None:
        """Publish a committed checkpoint path and best metric to the job row."""
        ...


class _ClipComponentLoader(Protocol):
    """Deferred data loader kept out of the API import graph."""

    def load_clip_components(
        self,
        snapshot: TrainingRunSnapshot,
        paths: TrainingPaths,
    ) -> TrainingComponents: ...


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    """Model/data adapter seam for the production CLIP loader and tiny tests."""

    model: TrainingModel
    train_batch_count: Callable[[int], int]
    iter_train_batches: Callable[[int], Iterable[TrainingBatch]]
    validation_recall_at_1: Callable[[TrainingModel, int], float]
    device: str | None = None
    scaler_factory: Callable[[bool], object] | None = None


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Completed engine work, distinct from final test evaluation and publication."""

    completed: bool
    cancelled: bool
    epochs_completed: int
    optimizer_steps: int
    best_metric: float | None
    best_checkpoint: Path | None


@dataclass(slots=True)
class _TrainingRuntime:
    backend: TrainingBackend
    components: TrainingComponents
    model: TrainingModel
    device: TrainingDevice
    fp16: bool
    optimizer: TrainingOptimizer
    scheduler: TrainingScheduler
    scaler: TrainingScaler
    batch_counts: list[int]
    next_epoch: int
    optimizer_steps: int
    best_metric: float | None
    best_checkpoint: Path | None
    bad_epochs: int
    best_epoch: int
    best_model_state: dict[str, object] | None
    early_stopped: bool
    consecutive_skips: int = 0


@dataclass(frozen=True, slots=True)
class _EpochTraining:
    loss: float
    batches: int
    cancelled: bool = False


def accumulation_group_size(
    micro_batch_index: int,
    total_micro_batches: int,
    accumulation_steps: int,
) -> int:
    """Return the true size of this accumulation group, including its final remainder."""
    if total_micro_batches < 1 or accumulation_steps < 1:
        raise ValueError("micro-batch and accumulation counts must be positive")
    if not 0 <= micro_batch_index < total_micro_batches:
        raise ValueError("micro-batch index is outside the epoch")
    group_start = (micro_batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, total_micro_batches - group_start)


def warmup_cosine_multiplier(
    optimizer_step: int,
    *,
    total_steps: int,
    warmup_steps: int,
) -> float:
    """Compute a zero-based warmup/cosine multiplier in optimizer-step units."""
    if total_steps < 1 or warmup_steps < 0 or warmup_steps > total_steps:
        raise ValueError("scheduler step counts are invalid")
    if optimizer_step < 0:
        raise ValueError("optimizer step cannot be negative")
    if warmup_steps and optimizer_step < warmup_steps:
        return (optimizer_step + 1) / warmup_steps
    cosine_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (optimizer_step - warmup_steps) / cosine_steps))
    return 0.5 * (1 + math.cos(math.pi * progress))


def run_training(
    job_snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    reporter: TrainingReporter,
    cancellation: TrainingCancellation,
    *,
    components: TrainingComponents | None = None,
) -> TrainingResult:
    """Train CLIP for complete epochs and atomically save only fully resumable states."""
    from gods_watching.training.torch_backend import create_torch_training_backend  # noqa: PLC0415

    backend = create_torch_training_backend()
    backend.seed_everything(job_snapshot.config.seed)
    runtime: _TrainingRuntime | None = None
    try:
        if cancellation.is_set():
            return _snapshot_cancelled_result(job_snapshot, paths)
        runtime = _initialize_runtime(
            job_snapshot,
            paths,
            components,
            cancellation,
            backend,
        )
        paths.run_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        return _run_epochs(job_snapshot, paths, reporter, cancellation, runtime)
    except _TrainingCancelledBeforeInitializationError:
        return _snapshot_cancelled_result(job_snapshot, paths)
    except Exception as error:
        if not backend.is_out_of_memory(error):
            raise
        backend.clear_cuda_cache()
        raise TrainingOutOfMemoryError("CUDA ran out of memory during training") from error
    finally:
        if runtime is None:
            backend.clear_cuda_cache()
        else:
            runtime.optimizer.zero_grad(set_to_none=True)
            if runtime.backend.is_cuda_device(runtime.device):
                runtime.backend.clear_cuda_cache()


def _initialize_runtime(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    components: TrainingComponents | None,
    cancellation: TrainingCancellation,
    backend: TrainingBackend,
) -> _TrainingRuntime:
    config = snapshot.config
    if components is None:
        components = _load_clip_components(snapshot, paths)
    if cancellation.is_set():
        raise _TrainingCancelledBeforeInitializationError
    device_name = components.device or "cuda:0"
    if components.device is None and not backend.cuda_available():
        raise TrainingEngineError("training worker requires an available CUDA device")
    device = backend.create_device(device_name)
    fp16 = config.mixed_precision == "fp16"
    if fp16 and not backend.is_cuda_device(device):
        raise TrainingEngineError("FP16 training requires CUDA")
    model = components.model.to(device)
    model.train()
    batch_counts = [components.train_batch_count(epoch) for epoch in range(config.epochs)]
    if any(count < 1 for count in batch_counts):
        raise TrainingEngineError("training split has no valid contrastive micro-batches")
    total_steps = sum(math.ceil(count / config.gradient_accumulation) for count in batch_counts)
    warmup_steps = math.ceil(total_steps * config.warmup_ratio)
    optimizer, scheduler = backend.create_optimizer_and_scheduler(
        model,
        learning_rate=config.learning_rate,
        betas=_TRAINING_RECIPE_BETAS,
        epsilon=_TRAINING_RECIPE_EPSILON,
        weight_decay=config.weight_decay,
        lr_multiplier=lambda step: warmup_cosine_multiplier(
            step,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        ),
    )
    scaler = backend.create_scaler(fp16, components.scaler_factory)
    next_epoch = snapshot.current_epoch
    optimizer_steps = 0
    best_metric = snapshot.best_metric
    best_checkpoint: Path | None = (
        paths.best_checkpoint if paths.best_checkpoint.is_file() else None
    )
    bad_epochs = 0
    best_epoch = 0
    best_model_state: dict[str, object] | None = None
    early_stopped = False
    if snapshot.checkpoint_path is not None:
        checkpoint = load_checkpoint_verified(
            snapshot.checkpoint_path,
            snapshot.checkpoint_identity(),
        )
        restored = restore_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            torch_module=backend,
        )
        next_epoch = restored.next_epoch
        optimizer_steps = restored.optimizer_steps
        best_metric = restored.best_metric
        bad_epochs = restored.bad_epochs
        best_epoch = restored.best_epoch
        best_model_state = restored.best_model
        early_stopped = restored.early_stopped
        if best_model_state is not None and best_metric is not None:
            _save_best_checkpoint(
                snapshot,
                paths,
                best_model_state,
                best_metric,
                best_epoch,
            )
            best_checkpoint = paths.best_checkpoint
        else:
            best_checkpoint = None
    return _TrainingRuntime(
        backend=backend,
        components=components,
        model=model,
        device=device,
        fp16=fp16,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        batch_counts=batch_counts,
        next_epoch=next_epoch,
        optimizer_steps=optimizer_steps,
        best_metric=best_metric,
        best_checkpoint=best_checkpoint,
        bad_epochs=bad_epochs,
        best_epoch=best_epoch,
        best_model_state=best_model_state,
        early_stopped=early_stopped,
    )


def _run_epochs(  # noqa: C901, PLR0911
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    reporter: TrainingReporter,
    cancellation: TrainingCancellation,
    runtime: _TrainingRuntime,
) -> TrainingResult:
    if cancellation.is_set():
        return _cancelled_result(runtime)
    if runtime.early_stopped:
        if runtime.optimizer_steps == 0:
            raise TrainingNumericalError("training run has no successful optimizer update")
        return _completed_result(runtime)
    for epoch in range(runtime.next_epoch, snapshot.config.epochs):
        if cancellation.is_set():
            return _cancelled_result(runtime)
        trained = _train_epoch(snapshot.config, epoch, reporter, cancellation, runtime)
        if trained.cancelled:
            return _cancelled_result(runtime)
        if cancellation.is_set():
            return _cancelled_result(runtime)
        cancelled, should_stop = _complete_epoch(
            snapshot,
            paths,
            reporter,
            cancellation,
            runtime,
            epoch,
            trained,
        )
        if cancelled or cancellation.is_set():
            return _cancelled_result(runtime)
        if should_stop:
            break
    if cancellation.is_set():
        return _cancelled_result(runtime)
    if runtime.optimizer_steps == 0:
        raise TrainingNumericalError("training run has no successful optimizer update")
    return _completed_result(runtime)


def _completed_result(runtime: _TrainingRuntime) -> TrainingResult:
    return TrainingResult(
        completed=True,
        cancelled=False,
        epochs_completed=runtime.next_epoch,
        optimizer_steps=runtime.optimizer_steps,
        best_metric=runtime.best_metric,
        best_checkpoint=runtime.best_checkpoint,
    )


def _train_epoch(
    config: TrainingConfig,
    epoch: int,
    reporter: TrainingReporter,
    cancellation: TrainingCancellation,
    runtime: _TrainingRuntime,
) -> _EpochTraining:
    expected_batches = runtime.batch_counts[epoch]
    optimizer_steps_before_epoch = runtime.optimizer_steps
    total_loss = 0.0
    observed_batches = 0
    runtime.model.train()
    for index, batch in enumerate(runtime.components.iter_train_batches(epoch)):
        if index >= expected_batches:
            raise TrainingEngineError("training batch iterator exceeded its planned size")
        if index % config.gradient_accumulation == 0 and cancellation.is_set():
            return _EpochTraining(total_loss, observed_batches, cancelled=True)
        divisor = accumulation_group_size(
            index,
            expected_batches,
            config.gradient_accumulation,
        )
        with runtime.backend.autocast(runtime.device, enabled=runtime.fp16):
            loss = runtime.backend.contrastive_loss(
                runtime.model,
                batch.model_inputs,
                runtime.device,
            )
        if not runtime.backend.is_finite(loss):
            raise TrainingNumericalError("non-finite contrastive loss; no checkpoint saved")
        total_loss += runtime.backend.scalar_value(loss)
        observed_batches += 1
        runtime.scaler.scale(loss / divisor).backward()
        finished = (
            (index + 1) % config.gradient_accumulation == 0
            or index + 1 == expected_batches
        )
        if finished:
            _optimizer_update(config, epoch, index, expected_batches, reporter, runtime)
            if cancellation.is_set() and index + 1 == expected_batches:
                return _EpochTraining(total_loss, observed_batches, cancelled=True)
    if observed_batches != expected_batches:
        raise TrainingEngineError("training batch iterator ended before its planned size")
    if runtime.optimizer_steps == optimizer_steps_before_epoch:
        raise TrainingNumericalError("epoch has no successful optimizer update")
    return _EpochTraining(total_loss, observed_batches)


def _optimizer_update(  # noqa: PLR0913
    config: TrainingConfig,
    epoch: int,
    batch_index: int,
    total_batches: int,
    reporter: TrainingReporter,
    runtime: _TrainingRuntime,
) -> None:
    runtime.scaler.unscale_(runtime.optimizer)
    parameters = tuple(runtime.model.parameters())
    if not runtime.fp16 and not runtime.backend.gradients_are_finite(parameters):
        raise TrainingNumericalError("non-finite gradient before FP32 optimizer update")
    gradient_norm = runtime.backend.clip_grad_norm(
        parameters,
        max_norm=config.gradient_clipping_norm,
    )
    if not runtime.fp16 and not math.isfinite(gradient_norm):
        raise TrainingNumericalError("non-finite FP32 gradient norm before optimizer update")
    step_before = runtime.backend.optimizer_step_count(runtime.optimizer)
    scale_before = float(runtime.scaler.get_scale())
    runtime.scaler.step(runtime.optimizer)
    runtime.scaler.update()
    succeeded = optimizer_update_succeeded(
        scale_before,
        float(runtime.scaler.get_scale()),
        step_before,
        runtime.backend.optimizer_step_count(runtime.optimizer),
    )
    if succeeded:
        runtime.scheduler.step()
        runtime.optimizer_steps += 1
        runtime.consecutive_skips = 0
    else:
        runtime.consecutive_skips += 1
    runtime.optimizer.zero_grad(set_to_none=True)
    if runtime.consecutive_skips >= _MAX_CONSECUTIVE_SKIPPED_STEPS:
        raise TrainingNumericalError("AMP skipped too many consecutive optimizer updates")
    reporter.progress(
        {
            "epoch": epoch + 1,
            "micro_batch": batch_index + 1,
            "micro_batches": total_batches,
            "optimizer_step": runtime.optimizer_steps,
            "learning_rate": runtime.backend.learning_rate(runtime.optimizer),
        }
    )


def _complete_epoch(  # noqa: PLR0913
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    reporter: TrainingReporter,
    cancellation: TrainingCancellation,
    runtime: _TrainingRuntime,
    epoch: int,
    trained: _EpochTraining,
) -> tuple[bool, bool]:
    runtime.model.eval()
    with runtime.backend.no_grad():
        recall = float(runtime.components.validation_recall_at_1(runtime.model, epoch))
    if cancellation.is_set():
        return True, False
    if not math.isfinite(recall) or not 0 <= recall <= 1:
        raise TrainingNumericalError("validation recall is non-finite or outside [0, 1]")
    improved = runtime.best_metric is None or recall > runtime.best_metric
    if improved:
        runtime.best_metric = recall
        runtime.bad_epochs = 0
        runtime.best_checkpoint = paths.best_checkpoint
        runtime.best_epoch = epoch + 1
        runtime.best_model_state = model_state_cpu(runtime.model)
    else:
        runtime.bad_epochs += 1
        if runtime.best_checkpoint is None:
            raise TrainingEngineError("best validation checkpoint is missing")

    if runtime.best_metric is None or runtime.best_model_state is None:
        raise TrainingEngineError("best validation checkpoint state is missing")
    runtime.next_epoch = epoch + 1
    patience = snapshot.config.early_stopping_patience
    runtime.early_stopped = patience is not None and runtime.bad_epochs >= patience
    state = build_training_checkpoint(
        snapshot.checkpoint_identity(),
        model=runtime.model,
        optimizer=runtime.optimizer,
        scheduler=runtime.scheduler,
        scaler=runtime.scaler,
        torch_module=runtime.backend,
        seed=snapshot.config.seed,
        next_epoch=runtime.next_epoch,
        optimizer_steps=runtime.optimizer_steps,
        best_metric=runtime.best_metric,
        bad_epochs=runtime.bad_epochs,
        best_epoch=runtime.best_epoch,
        best_model=runtime.best_model_state,
        early_stopped=runtime.early_stopped,
    )
    save_checkpoint_atomic(state, paths.last_checkpoint)
    if improved:
        _save_best_checkpoint(
            snapshot,
            paths,
            runtime.best_model_state,
            runtime.best_metric,
            runtime.best_epoch,
        )
    reporter.checkpoint(paths.last_checkpoint, runtime.best_metric)
    reporter.metric(
        TrainingMetric(
            epoch=runtime.next_epoch,
            step=runtime.optimizer_steps,
            training_loss=trained.loss / trained.batches,
            validation_recall_at_1=recall,
            observed_at=datetime.now(UTC),
        )
    )
    reporter.progress(
        {
            "epoch": runtime.next_epoch,
            "optimizer_step": runtime.optimizer_steps,
            "epochs": snapshot.config.epochs,
            "validation_recall_at_1": recall,
        }
    )
    return cancellation.is_set(), runtime.early_stopped


def _save_best_checkpoint(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    best_model: dict[str, object],
    best_metric: float,
    best_epoch: int,
) -> None:
    save_checkpoint_atomic(
        {
            "identity": snapshot.checkpoint_identity(),
            "model": best_model,
            "training": {"best_epoch": best_epoch, "best_metric": best_metric},
        },
        paths.best_checkpoint,
    )


def _snapshot_cancelled_result(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
) -> TrainingResult:
    return TrainingResult(
        completed=False,
        cancelled=True,
        epochs_completed=snapshot.current_epoch,
        optimizer_steps=0,
        best_metric=snapshot.best_metric,
        best_checkpoint=paths.best_checkpoint if paths.best_checkpoint.is_file() else None,
    )


def _cancelled_result(runtime: _TrainingRuntime) -> TrainingResult:
    return TrainingResult(
        completed=False,
        cancelled=True,
        epochs_completed=runtime.next_epoch,
        optimizer_steps=runtime.optimizer_steps,
        best_metric=runtime.best_metric,
        best_checkpoint=runtime.best_checkpoint,
    )


def _load_clip_components(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
) -> TrainingComponents:
    loader = cast(
        "_ClipComponentLoader",
        cast(
            "object",
            importlib.import_module("gods_watching.training.engine_data"),
        ),
    )
    return loader.load_clip_components(snapshot, paths)


__all__ = [
    "TrainingBatch",
    "TrainingCancellation",
    "TrainingComponents",
    "TrainingEngineError",
    "TrainingNumericalError",
    "TrainingOutOfMemoryError",
    "TrainingPaths",
    "TrainingReporter",
    "TrainingResult",
    "TrainingRunSnapshot",
    "accumulation_group_size",
    "run_training",
    "warmup_cosine_multiplier",
]
