from __future__ import annotations

import importlib
import math
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast
from uuid import uuid4

import numpy as np
import pytest

from gods_watching.contracts.training import TrainingConfig, TrainingMetric
from gods_watching.training.engine import (
    TrainingBatch,
    TrainingComponents,
    TrainingNumericalError,
    TrainingOutOfMemoryError,
    TrainingPaths,
    TrainingResult,
    TrainingRunSnapshot,
    accumulation_group_size,
    run_training,
    warmup_cosine_multiplier,
)
from gods_watching.training.engine_api import TrainingModel, TrainingTensor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from contextlib import AbstractContextManager

    from gods_watching.training.engine_api import TrainingOptimizer


class _EngineModuleForTests(Protocol):
    save_checkpoint_atomic: Callable[..., None]
    _save_best_checkpoint: Callable[..., None]


class _TrainableTensor(TrainingTensor, Protocol):
    def copy_(self, other: TrainingTensor) -> None: ...

    def register_hook(self, hook: Callable[[TrainingTensor], TrainingTensor]) -> object: ...


class _ParameterFactory(Protocol):
    def __call__(self, tensor: TrainingTensor) -> _TrainableTensor: ...


class _NeuralNetworkApi(Protocol):
    Parameter: _ParameterFactory


class _CudaTestApi(Protocol):
    OutOfMemoryError: type[Exception]


class _TorchTestApi(Protocol):
    float32: object
    int64: object
    nn: _NeuralNetworkApi
    cuda: _CudaTestApi

    def tensor(self, value: object, *, dtype: object | None = None) -> TrainingTensor: ...

    def rand(self, shape: tuple[int, int], *, dtype: object) -> TrainingTensor: ...

    def ones(
        self,
        shape: tuple[int, int],
        *,
        dtype: object | None = None,
    ) -> TrainingTensor: ...

    def full_like(self, tensor: TrainingTensor, fill_value: float) -> TrainingTensor: ...

    def no_grad(self) -> AbstractContextManager[object]: ...

    def equal(self, left: TrainingTensor, right: TrainingTensor) -> bool: ...

    def manual_seed(self, seed: int) -> None: ...

    def load(
        self,
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> dict[str, object]: ...


class _TinyClipModel(TrainingModel, Protocol):
    weight: _TrainableTensor
    non_finite_gradient: bool


@dataclass(frozen=True, slots=True)
class _ModelOutput:
    image_embeds: TrainingTensor
    text_embeds: TrainingTensor


def _checkpoint_section(checkpoint: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = checkpoint[name]
    if not isinstance(value, dict):
        message = f"checkpoint section {name} is not a mapping"
        raise TypeError(message)
    return cast("Mapping[str, object]", value)


@dataclass
class _Reporter:
    progress_rows: list[dict[str, object]] = field(default_factory=list)
    metrics: list[TrainingMetric] = field(default_factory=list)
    checkpoints: list[Path] = field(default_factory=list)
    on_progress: Callable[[dict[str, object]], None] | None = None
    on_metric: Callable[[TrainingMetric], None] | None = None

    def progress(self, row: dict[str, object]) -> None:
        self.progress_rows.append(row)
        if self.on_progress is not None:
            self.on_progress(row)

    def metric(self, metric: TrainingMetric) -> None:
        self.metrics.append(metric)
        if self.on_metric is not None:
            self.on_metric(metric)

    def checkpoint(self, path: Path, best_metric: float) -> None:
        _ = best_metric
        self.checkpoints.append(path)


class _SkippingScaler:
    def scale(self, loss: object) -> object:
        return loss

    def unscale_(self, _optimizer: object) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0

    def step(self, _optimizer: object) -> None:
        return None

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {"scale": 1.0}

    def load_state_dict(self, _state: dict[str, object]) -> None:
        return None


class _EpochAwareScaler:
    _should_skip: Callable[[], bool]

    def __init__(self, should_skip: Callable[[], bool]) -> None:
        self._should_skip = should_skip

    def scale(self, loss: TrainingTensor) -> TrainingTensor:
        return loss

    def unscale_(self, _optimizer: TrainingOptimizer) -> None:
        return None

    def get_scale(self) -> float:
        return 1.0

    def step(self, optimizer: TrainingOptimizer) -> None:
        if not self._should_skip():
            optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, _state: Mapping[str, object]) -> None:
        return None


def _torch() -> _TorchTestApi:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        pytest.skip("run in the dedicated training environment")
    return cast("_TorchTestApi", cast("object", torch))


def _snapshot(
    config: TrainingConfig,
    *,
    checkpoint_path: Path | None = None,
) -> TrainingRunSnapshot:
    return TrainingRunSnapshot(
        job_id=uuid4(),
        owner_generation=1,
        config=config,
        dataset_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
        current_epoch=0,
        best_metric=None,
        checkpoint_path=checkpoint_path,
    )


def _paths(tmp_path: Path) -> TrainingPaths:
    return TrainingPaths(
        run_directory=tmp_path,
        dataset_root=tmp_path / "dataset",
        model_root=tmp_path / "model",
    )


def _tiny_model(  # noqa: C901
    torch: _TorchTestApi,
    *,
    non_finite: bool = False,
    raise_oom: bool = False,
) -> _TinyClipModel:
    class TinyModel:
        weight: _TrainableTensor
        logit_scale: _TrainableTensor
        training: bool
        non_finite_gradient: bool

        def __init__(self) -> None:
            self.weight = torch.nn.Parameter(
                torch.tensor([[0.1, 0.4], [0.4, 0.1]], dtype=torch.float32)
            )
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
            self.training = True
            self.non_finite_gradient = False

        def parameters(self) -> list[_TrainableTensor]:
            return [self.weight, self.logit_scale]

        def train(self) -> None:
            self.training = True

        def eval(self) -> None:
            self.training = False

        def to(self, _device: object) -> TinyModel:
            return self

        def __call__(self, **inputs: TrainingTensor) -> _ModelOutput:
            if raise_oom:
                message = "synthetic OOM"
                raise torch.cuda.OutOfMemoryError(message)
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            image_embeds = pixel_values @ self.weight
            text_embeds = input_ids @ self.weight
            if non_finite:
                image_embeds = image_embeds * torch.tensor(float("nan"))
            if self.non_finite_gradient:
                _ = self.weight.register_hook(
                    lambda gradient: torch.full_like(gradient, float("inf"))
                )
            return _ModelOutput(image_embeds=image_embeds, text_embeds=text_embeds)

        def state_dict(self) -> dict[str, TrainingTensor]:
            return {
                "weight": self.weight.detach().clone(),
                "logit_scale": self.logit_scale.detach().clone(),
            }

        def load_state_dict(self, state: Mapping[str, object]) -> None:
            with torch.no_grad():
                self.weight.copy_(cast("TrainingTensor", state["weight"]))
                self.logit_scale.copy_(cast("TrainingTensor", state["logit_scale"]))

    return cast("_TinyClipModel", cast("object", TinyModel()))


def _oversized_finite_gradient_model(torch: _TorchTestApi) -> TrainingModel:
    class OversizedModel:
        image_features: _TrainableTensor
        text_features: _TrainableTensor
        logit_scale: _TrainableTensor

        def __init__(self) -> None:
            self.image_features = torch.nn.Parameter(torch.ones((2, 64)))
            self.text_features = torch.nn.Parameter(torch.ones((2, 64)))
            self.logit_scale = torch.nn.Parameter(torch.tensor(0.1))
            _ = self.image_features.register_hook(
                lambda gradient: torch.full_like(gradient, 2e38)
            )

        def parameters(self) -> list[_TrainableTensor]:
            return [self.image_features, self.text_features, self.logit_scale]

        def train(self) -> None:
            return None

        def eval(self) -> None:
            return None

        def to(self, _device: object) -> OversizedModel:
            return self

        def __call__(self, **_inputs: TrainingTensor) -> _ModelOutput:
            return _ModelOutput(
                image_embeds=self.image_features,
                text_embeds=self.text_features,
            )

        def state_dict(self) -> dict[str, TrainingTensor]:
            return {
                "image_features": self.image_features.detach().clone(),
                "text_features": self.text_features.detach().clone(),
                "logit_scale": self.logit_scale.detach().clone(),
            }

        def load_state_dict(self, state: Mapping[str, object]) -> None:
            with torch.no_grad():
                self.image_features.copy_(cast("TrainingTensor", state["image_features"]))
                self.text_features.copy_(cast("TrainingTensor", state["text_features"]))
                self.logit_scale.copy_(cast("TrainingTensor", state["logit_scale"]))

    return cast("TrainingModel", cast("object", OversizedModel()))


def _batch(torch: _TorchTestApi, *, random_values: bool = False) -> TrainingBatch:
    if random_values:
        pixel_values = torch.rand((2, 2), dtype=torch.float32)
        input_ids = torch.rand((2, 2), dtype=torch.float32)
    else:
        pixel_values = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        input_ids = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
    return TrainingBatch(
        model_inputs={
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": torch.ones((2, 2), dtype=torch.int64),
        }
    )


def _components(
    model: TrainingModel,
    batches: int,
    batch_factory: Callable[[int], TrainingBatch],
    *,
    validation: Callable[[TrainingModel, int], float] | None = None,
) -> TrainingComponents:
    return TrainingComponents(
        model=model,
        train_batch_count=lambda _epoch: batches,
        iter_train_batches=lambda epoch: (batch_factory(epoch) for _ in range(batches)),
        validation_recall_at_1=validation or (lambda _model, _epoch: 0.5),
        device="cpu",
    )


def _config(*, epochs: int = 1, accumulation: int = 3) -> TrainingConfig:
    return TrainingConfig(
        epochs=epochs,
        learning_rate=1e-3,
        micro_batch_size=2,
        gradient_accumulation=accumulation,
        warmup_ratio=0.0,
        mixed_precision="fp32",
        gradient_checkpointing=False,
        early_stopping_patience=None,
    )


def test_accumulation_uses_actual_size_for_a_partial_final_group() -> None:
    assert [accumulation_group_size(index, 5, 3) for index in range(5)] == [3, 3, 3, 2, 2]


def test_warmup_cosine_schedule_counts_optimizer_steps() -> None:
    multipliers = [
        warmup_cosine_multiplier(step, total_steps=4, warmup_steps=2) for step in range(4)
    ]

    assert all(
        math.isclose(actual, expected, abs_tol=1e-9)
        for actual, expected in zip(multipliers, [0.5, 1.0, 1.0, 0.5], strict=True)
    )


def test_tiny_contrastive_model_weights_change_and_partial_group_steps(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    before = model.weight.detach().clone()
    components = _components(model, 5, lambda _epoch: _batch(torch))
    reporter = _Reporter()

    result = run_training(
        _snapshot(_config()),
        _paths(tmp_path),
        reporter,
        threading.Event(),
        components=components,
    )

    assert isinstance(result, TrainingResult)
    assert result.optimizer_steps == 2
    assert not torch.equal(before, model.weight)
    assert reporter.metrics[0].epoch == 1
    assert reporter.metrics[0].validation_recall_at_1 == 0.5


def test_best_checkpoint_is_not_overwritten_by_a_worse_validation_epoch(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    recall = iter((0.8, 0.1))
    paths = _paths(tmp_path)

    _ = run_training(
        _snapshot(_config(epochs=2, accumulation=2)),
        paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            model,
            2,
            lambda _epoch: _batch(torch),
            validation=lambda _model, _epoch: next(recall),
        ),
    )

    best = torch.load(paths.best_checkpoint, map_location="cpu", weights_only=True)
    last = torch.load(paths.last_checkpoint, map_location="cpu", weights_only=True)
    assert _checkpoint_section(best, "training")["best_epoch"] == 1
    assert _checkpoint_section(last, "training")["epoch"] == 2


def test_config_seed_makes_tiny_training_repeatable(tmp_path: Path) -> None:
    torch = _torch()
    results: list[TrainingTensor] = []
    for index, external_seed in enumerate((17, 991)):
        torch.manual_seed(external_seed)
        model = _tiny_model(torch)
        _ = run_training(
            _snapshot(_config(epochs=1, accumulation=2)),
            _paths(tmp_path / str(index)),
            _Reporter(),
            threading.Event(),
            components=_components(
                model,
                2,
                lambda _epoch: _batch(torch, random_values=True),
            ),
        )
        results.append(model.weight.detach().clone())

    assert torch.equal(results[0], results[1])


def test_non_finite_contrastive_loss_fails_and_clears_gradients(tmp_path: Path) -> None:
    torch = _torch()
    model = _tiny_model(torch, non_finite=True)
    components = _components(model, 1, lambda _epoch: _batch(torch))

    with pytest.raises(TrainingNumericalError):
        _ = run_training(
            _snapshot(_config()),
            _paths(tmp_path),
            _Reporter(),
            threading.Event(),
            components=components,
        )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_finite_loss_with_non_finite_fp32_gradients_fails_before_step(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    model.non_finite_gradient = True

    with pytest.raises(TrainingNumericalError, match="non-finite gradient"):
        _ = run_training(
            _snapshot(_config()),
            _paths(tmp_path),
            _Reporter(),
            threading.Event(),
            components=_components(model, 1, lambda _epoch: _batch(torch)),
        )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_finite_fp32_gradients_with_overflowing_norm_fail_before_step(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _oversized_finite_gradient_model(torch)
    batch = _batch(torch)

    with pytest.raises(TrainingNumericalError, match="non-finite FP32 gradient norm"):
        _ = run_training(
            _snapshot(_config()),
            _paths(tmp_path),
            _Reporter(),
            threading.Event(),
            components=_components(model, 1, lambda _epoch: batch),
        )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_cancellation_at_optimizer_boundary_keeps_only_complete_epoch_checkpoints(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    cancellation = threading.Event()
    reporter = _Reporter(on_progress=lambda _row: cancellation.set())
    result = run_training(
        _snapshot(_config(epochs=3, accumulation=2)),
        _paths(tmp_path),
        reporter,
        cancellation,
        components=_components(model, 4, lambda _epoch: _batch(torch)),
    )

    assert result.cancelled
    assert result.epochs_completed == 0
    assert not (_paths(tmp_path).last_checkpoint).exists()
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cancellation_after_the_final_optimizer_group_does_not_report_completion(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    cancellation = threading.Event()
    reporter = _Reporter(on_progress=lambda _row: cancellation.set())

    result = run_training(
        _snapshot(_config(epochs=1, accumulation=2)),
        _paths(tmp_path),
        reporter,
        cancellation,
        components=_components(model, 2, lambda _epoch: _batch(torch)),
    )

    assert result.cancelled
    assert not result.completed
    assert result.epochs_completed == 0
    assert not _paths(tmp_path).last_checkpoint.exists()


def test_out_of_memory_fails_after_cleanup(tmp_path: Path) -> None:
    torch = _torch()
    model = _tiny_model(torch, raise_oom=True)

    with pytest.raises(TrainingOutOfMemoryError):
        _ = run_training(
            _snapshot(_config()),
            _paths(tmp_path),
            _Reporter(),
            threading.Event(),
            components=_components(model, 1, lambda _epoch: _batch(torch)),
        )

    assert all(parameter.grad is None for parameter in model.parameters())


def test_epoch_with_only_overflow_skips_is_not_reported_as_completed(
    tmp_path: Path,
) -> None:
    torch = _torch()
    model = _tiny_model(torch)
    components = _components(model, 2, lambda _epoch: _batch(torch))
    components = TrainingComponents(
        model=components.model,
        train_batch_count=components.train_batch_count,
        iter_train_batches=components.iter_train_batches,
        validation_recall_at_1=components.validation_recall_at_1,
        device="cpu",
        scaler_factory=lambda _enabled: _SkippingScaler(),
    )

    with pytest.raises(TrainingNumericalError, match="no successful optimizer update"):
        _ = run_training(
            _snapshot(_config(epochs=1, accumulation=2)),
            _paths(tmp_path),
            _Reporter(),
            threading.Event(),
            components=components,
        )

    assert not _paths(tmp_path).last_checkpoint.exists()
    assert not _paths(tmp_path).best_checkpoint.exists()


def test_later_all_skipped_epoch_preserves_previous_complete_checkpoint(
    tmp_path: Path,
) -> None:
    torch = _torch()
    paths = _paths(tmp_path)
    model = _tiny_model(torch)
    active_epoch = [0]
    scaler = _EpochAwareScaler(should_skip=lambda: active_epoch[0] > 0)

    def batches(epoch: int) -> Iterable[TrainingBatch]:
        active_epoch[0] = epoch
        yield _batch(torch)
        yield _batch(torch)

    components = TrainingComponents(
        model=model,
        train_batch_count=lambda _epoch: 2,
        iter_train_batches=batches,
        validation_recall_at_1=lambda _model, _epoch: 0.5,
        device="cpu",
        scaler_factory=lambda _enabled: scaler,
    )
    reporter = _Reporter()

    with pytest.raises(TrainingNumericalError, match="no successful optimizer update"):
        _ = run_training(
            _snapshot(_config(epochs=2, accumulation=2)),
            paths,
            reporter,
            threading.Event(),
            components=components,
        )

    last = torch.load(paths.last_checkpoint, map_location="cpu", weights_only=True)
    best = torch.load(paths.best_checkpoint, map_location="cpu", weights_only=True)
    assert _checkpoint_section(last, "training")["epoch"] == 1
    assert _checkpoint_section(best, "training")["best_epoch"] == 1
    assert [metric.epoch for metric in reporter.metrics] == [1]
    assert reporter.checkpoints == [paths.last_checkpoint]


def test_rng_optimizer_and_sampler_state_resume_equivalently(tmp_path: Path) -> None:
    torch = _torch()
    np.random.seed(41)  # noqa: NPY002 - test the process-global checkpoint stream.
    random.seed(41)
    torch.manual_seed(41)
    full_model = _tiny_model(torch)
    full_paths = _paths(tmp_path / "full")
    _ = run_training(
        _snapshot(_config(epochs=2, accumulation=2)),
        full_paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            full_model,
            4,
            lambda _epoch: _batch(torch, random_values=True),
        ),
    )

    np.random.seed(41)  # noqa: NPY002 - test restoration from the process-global stream.
    random.seed(41)
    torch.manual_seed(41)
    partial_model = _tiny_model(torch)
    partial_paths = _paths(tmp_path / "resumed")
    cancellation = threading.Event()
    reporter = _Reporter(on_metric=lambda _metric: cancellation.set())
    partial = run_training(
        _snapshot(_config(epochs=2, accumulation=2)),
        partial_paths,
        reporter,
        cancellation,
        components=_components(
            partial_model,
            4,
            lambda _epoch: _batch(torch, random_values=True),
        ),
    )
    assert partial.cancelled
    assert partial_paths.last_checkpoint.is_file()

    resumed_model = _tiny_model(torch)
    resumed = run_training(
        _snapshot(
            _config(epochs=2, accumulation=2),
            checkpoint_path=partial_paths.last_checkpoint,
        ),
        partial_paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            resumed_model,
            4,
            lambda _epoch: _batch(torch, random_values=True),
        ),
    )

    assert resumed.completed
    assert torch.equal(full_model.weight, resumed_model.weight)


def test_best_model_is_embedded_in_last_checkpoint_if_best_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _torch()
    engine_module = cast(
        "_EngineModuleForTests",
        cast("object", importlib.import_module("gods_watching.training.engine")),
    )
    real_save = engine_module.save_checkpoint_atomic
    paths = _paths(tmp_path)
    model = _tiny_model(torch)
    recalls = iter((0.8, 0.1))

    def fail_best_write(state: object, path: Path, **kwargs: object) -> None:
        if Path(path).name == "best.pt":
            message = "simulated best checkpoint failure"
            raise OSError(message)
        real_save(state, path, **kwargs)

    with monkeypatch.context() as local_patch:
        local_patch.setattr(engine_module, "save_checkpoint_atomic", fail_best_write)
        with pytest.raises(OSError, match="simulated best checkpoint failure"):
            _ = run_training(
                _snapshot(_config(epochs=2, accumulation=2)),
                paths,
                _Reporter(),
                threading.Event(),
                components=_components(
                    model,
                    2,
                    lambda _epoch: _batch(torch),
                    validation=lambda _model, _epoch: next(recalls),
                ),
            )

    last = torch.load(paths.last_checkpoint, map_location="cpu", weights_only=True)
    assert _checkpoint_section(last, "training")["best_epoch"] == 1
    assert "best_model" in last

    resumed_model = _tiny_model(torch)
    _ = run_training(
        _snapshot(
            _config(epochs=2, accumulation=2),
            checkpoint_path=paths.last_checkpoint,
        ),
        paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            resumed_model,
            2,
            lambda _epoch: _batch(torch),
            validation=lambda _model, _epoch: 0.1,
        ),
    )
    repaired_best = torch.load(paths.best_checkpoint, map_location="cpu", weights_only=True)
    assert _checkpoint_section(repaired_best, "training")["best_epoch"] == 1
    repaired_weight = cast("TrainingTensor", _checkpoint_section(repaired_best, "model")["weight"])
    best_weight = cast("TrainingTensor", _checkpoint_section(last, "best_model")["weight"])
    assert torch.equal(repaired_weight, best_weight)


def test_early_stopping_checkpoint_does_not_resume_extra_epochs(tmp_path: Path) -> None:
    torch = _torch()
    config = _config(epochs=5, accumulation=2).model_copy(
        update={"early_stopping_patience": 1}
    )
    paths = _paths(tmp_path)
    model = _tiny_model(torch)
    recalls = iter((0.8, 0.7, 0.6, 0.5, 0.4))
    first = run_training(
        _snapshot(config),
        paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            model,
            2,
            lambda _epoch: _batch(torch),
            validation=lambda _model, _epoch: next(recalls),
        ),
    )
    assert first.completed
    assert first.epochs_completed == 2

    resumed = run_training(
        _snapshot(config, checkpoint_path=paths.last_checkpoint),
        paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            _tiny_model(torch),
            2,
            lambda _epoch: _batch(torch),
            validation=lambda _model, _epoch: pytest.fail(
                "resumed an already early-stopped training run"
            ),
        ),
    )
    assert resumed.completed
    assert resumed.epochs_completed == 2


def test_cancellation_during_stopped_checkpoint_repair_wins_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _torch()
    engine_module = cast(
        "_EngineModuleForTests",
        cast("object", importlib.import_module("gods_watching.training.engine")),
    )
    config = _config(epochs=4, accumulation=2).model_copy(
        update={"early_stopping_patience": 1}
    )
    paths = _paths(tmp_path)
    recalls = iter((0.8, 0.7, 0.6, 0.5))
    _ = run_training(
        _snapshot(config),
        paths,
        _Reporter(),
        threading.Event(),
        components=_components(
            _tiny_model(torch),
            2,
            lambda _epoch: _batch(torch),
            validation=lambda _model, _epoch: next(recalls),
        ),
    )
    cancellation = threading.Event()
    # This test hooks the narrow crash-recovery boundary to inject cancellation.
    original_save_best = engine_module._save_best_checkpoint  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

    def cancel_during_repair(*args: object, **kwargs: object) -> None:
        original_save_best(*args, **kwargs)
        cancellation.set()

    monkeypatch.setattr(engine_module, "_save_best_checkpoint", cancel_during_repair)
    result = run_training(
        _snapshot(config, checkpoint_path=paths.last_checkpoint),
        paths,
        _Reporter(),
        cancellation,
        components=_components(_tiny_model(torch), 2, lambda _epoch: _batch(torch)),
    )

    assert result.cancelled
    assert not result.completed
    assert result.epochs_completed == 2
