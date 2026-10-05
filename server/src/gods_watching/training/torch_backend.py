"""Narrow dynamic adapter for the optional Torch installation in the worker image."""

from __future__ import annotations

import importlib
import random
from typing import TYPE_CHECKING, Protocol, cast, override

import numpy as np

from gods_watching.training.determinism import (
    configure_torch_determinism,
    prepare_deterministic_cuda_environment,
)
from gods_watching.training.engine_api import TrainingBackend

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from contextlib import AbstractContextManager

    from gods_watching.training.engine_api import (
        TrainingDevice,
        TrainingModel,
        TrainingOptimizer,
        TrainingScaler,
        TrainingScheduler,
        TrainingTensor,
    )


class _CudaApi(Protocol):
    """Torch CUDA calls used by this adapter."""

    OutOfMemoryError: type[BaseException]

    def is_available(self) -> bool: ...

    def empty_cache(self) -> None: ...

    def manual_seed_all(self, seed: int) -> None: ...

    def get_rng_state_all(self) -> list[TrainingTensor]: ...

    def set_rng_state_all(self, states: Sequence[TrainingTensor]) -> None: ...


class _AdamWFactory(Protocol):
    """Torch AdamW constructor narrowed to the recipe options."""

    def __call__(  # noqa: PLR0913
        self,
        parameters: Iterable[TrainingTensor],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        foreach: bool,
    ) -> TrainingOptimizer: ...


class _LambdaLRFactory(Protocol):
    """Torch LambdaLR constructor narrowed to one optimizer."""

    def __call__(
        self,
        optimizer: TrainingOptimizer,
        *,
        lr_lambda: Callable[[int], float],
    ) -> TrainingScheduler: ...


class _OptimizerModule(Protocol):
    """Torch optimizer constructors."""

    AdamW: _AdamWFactory
    lr_scheduler: _LRSchedulerModule


class _LRSchedulerModule(Protocol):
    """Torch learning-rate scheduler constructors."""

    LambdaLR: _LambdaLRFactory


class _GradScalerFactory(Protocol):
    """Torch AMP GradScaler constructor."""

    def __call__(self, device: str, *, enabled: bool) -> TrainingScaler: ...


class _AmpModule(Protocol):
    """Torch AMP constructors."""

    GradScaler: _GradScalerFactory


class _CudnnBackend(Protocol):
    """cuDNN reproducibility flags."""

    deterministic: bool
    benchmark: bool


class _TorchBackends(Protocol):
    """Torch backend controls required by the deterministic recipe."""

    cudnn: _CudnnBackend


class _FunctionalModule(Protocol):
    """Torch functional neural-network operations."""

    def normalize(self, tensor: TrainingTensor, *, dim: int) -> TrainingTensor: ...

    def cross_entropy(self, logits: TrainingTensor, labels: TrainingTensor) -> TrainingTensor:
        ...


class _ClipUtilities(Protocol):
    """Torch gradient-clipping operation."""

    def clip_grad_norm_(
        self,
        parameters: Sequence[TrainingTensor],
        max_norm: float,
    ) -> TrainingTensor: ...


class _NNModule(Protocol):
    """Torch neural-network operations used by the contrastive objective."""

    functional: _FunctionalModule
    utils: _ClipUtilities


class _TorchApi(Protocol):
    """Only the Torch APIs consumed by the worker."""

    cuda: _CudaApi
    optim: _OptimizerModule
    amp: _AmpModule
    backends: _TorchBackends
    nn: _NNModule
    float16: object

    def manual_seed(self, seed: int) -> None: ...

    def use_deterministic_algorithms(
        self,
        mode: bool,
        *,
        warn_only: bool = False,
    ) -> None: ...

    def device(self, name: str) -> TrainingDevice: ...

    def autocast(
        self,
        *,
        device_type: str,
        dtype: object,
        enabled: bool,
    ) -> AbstractContextManager[object]: ...

    def no_grad(self) -> AbstractContextManager[object]: ...

    def isfinite(self, tensor: TrainingTensor) -> TrainingTensor: ...

    def arange(self, size: int, *, device: TrainingDevice) -> TrainingTensor: ...

    def cat(self, tensors: Sequence[TrainingTensor]) -> TrainingTensor: ...

    def get_rng_state(self) -> TrainingTensor: ...

    def set_rng_state(self, state: TrainingTensor) -> None: ...


class TorchTrainingBackend(TrainingBackend):
    """Resolve Torch once, then expose a typed surface to the training controller."""

    _torch: _TorchApi

    def __init__(self) -> None:
        """Import Torch only when a worker is ready to enter training."""
        # Torch is intentionally optional in the API image and exists only in the worker image.
        prepare_deterministic_cuda_environment()
        self._torch = cast("_TorchApi", cast("object", importlib.import_module("torch")))
        configure_torch_determinism(self._torch)

    @override
    def seed_everything(self, seed: int) -> None:
        random.seed(seed)
        # Restore/reseed the process-global generator used by deterministic dataset sampling.
        np.random.seed(seed)  # noqa: NPY002
        self._torch.manual_seed(seed)
        if self.cuda_available():
            self._torch.cuda.manual_seed_all(seed)

    @override
    def create_device(self, name: str) -> TrainingDevice:
        return self._torch.device(name)

    @override
    def is_cuda_device(self, device: TrainingDevice) -> bool:
        return device.type == "cuda"

    @override
    def cuda_available(self) -> bool:
        return self._torch.cuda.is_available()

    @override
    def clear_cuda_cache(self) -> None:
        if self.cuda_available():
            self._torch.cuda.empty_cache()

    @override
    def is_out_of_memory(self, error: Exception) -> bool:
        return isinstance(error, self._torch.cuda.OutOfMemoryError)

    @override
    def create_optimizer_and_scheduler(
        self,
        model: TrainingModel,
        *,
        learning_rate: float,
        betas: tuple[float, float],
        epsilon: float,
        weight_decay: float,
        lr_multiplier: Callable[[int], float],
    ) -> tuple[TrainingOptimizer, TrainingScheduler]:
        optimizer = self._torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=betas,
            eps=epsilon,
            weight_decay=weight_decay,
            foreach=False,
        )
        scheduler = self._torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_multiplier,
        )
        return optimizer, scheduler

    @override
    def create_scaler(
        self,
        enabled: bool,
        factory: Callable[[bool], object] | None,
    ) -> TrainingScaler:
        if factory is not None:
            return cast("TrainingScaler", factory(enabled))
        return self._torch.amp.GradScaler("cuda", enabled=enabled)

    @override
    def autocast(
        self,
        device: TrainingDevice,
        *,
        enabled: bool,
    ) -> AbstractContextManager[object]:
        return self._torch.autocast(
            device_type=device.type,
            dtype=self._torch.float16,
            enabled=enabled,
        )

    @override
    def no_grad(self) -> AbstractContextManager[object]:
        return self._torch.no_grad()

    @override
    def contrastive_loss(
        self,
        model: TrainingModel,
        model_inputs: Mapping[str, object],
        device: TrainingDevice,
    ) -> TrainingTensor:
        inputs = {
            name: cast("TrainingTensor", value).to(device)
            for name, value in model_inputs.items()
        }
        outputs = model(**inputs)
        image_embeddings = self._torch.nn.functional.normalize(outputs.image_embeds, dim=-1)
        text_embeddings = self._torch.nn.functional.normalize(outputs.text_embeds, dim=-1)
        logits = model.logit_scale.exp() * image_embeddings @ text_embeddings.transpose(0, 1)
        labels = self._torch.arange(logits.shape[0], device=device)
        return (
            self._torch.nn.functional.cross_entropy(logits, labels)
            + self._torch.nn.functional.cross_entropy(logits.transpose(0, 1), labels)
        ) / 2

    @override
    def normalize(self, tensor: TrainingTensor) -> TrainingTensor:
        return self._torch.nn.functional.normalize(tensor, dim=-1)

    @override
    def concatenate(self, tensors: Sequence[TrainingTensor]) -> TrainingTensor:
        return self._torch.cat(tensors)

    @override
    def is_finite(self, tensor: TrainingTensor) -> bool:
        return bool(self._torch.isfinite(tensor).all().item())

    @override
    def scalar_value(self, tensor: TrainingTensor) -> float:
        return float(tensor.detach().float().item())

    @override
    def gradients_are_finite(self, parameters: Sequence[TrainingTensor]) -> bool:
        for parameter in parameters:
            if parameter.grad is not None and not self.is_finite(parameter.grad):
                return False
        return True

    @override
    def clip_grad_norm(
        self,
        parameters: Sequence[TrainingTensor],
        max_norm: float,
    ) -> float:
        return float(self._torch.nn.utils.clip_grad_norm_(parameters, max_norm).item())

    @override
    def optimizer_step_count(self, optimizer: TrainingOptimizer) -> int:
        total = 0
        for state in optimizer.state.values():
            step = state.get("step")
            if step is not None:
                total += int(cast("TrainingTensor", step).item())
        return total

    @override
    def learning_rate(self, optimizer: TrainingOptimizer) -> float:
        if not optimizer.param_groups:
            message = "AdamW optimizer has no parameter groups"
            raise ValueError(message)
        learning_rate = optimizer.param_groups[0]["lr"]
        if not isinstance(learning_rate, int | float):
            message = "AdamW learning rate is not numeric"
            raise TypeError(message)
        return float(learning_rate)

    @override
    def get_rng_state(self) -> TrainingTensor:
        return self._torch.get_rng_state().cpu()

    @override
    def set_rng_state(self, state: object) -> None:
        self._torch.set_rng_state(cast("TrainingTensor", state))

    @override
    def get_cuda_rng_state_all(self) -> list[TrainingTensor]:
        return self._torch.cuda.get_rng_state_all() if self.cuda_available() else []

    @override
    def set_cuda_rng_state_all(self, states: Sequence[object]) -> None:
        self._torch.cuda.set_rng_state_all(
            [cast("TrainingTensor", state) for state in states]
        )


def create_torch_training_backend() -> TorchTrainingBackend:
    """Construct the worker-only Torch adapter."""
    return TorchTrainingBackend()


__all__ = ["TorchTrainingBackend", "create_torch_training_backend"]
