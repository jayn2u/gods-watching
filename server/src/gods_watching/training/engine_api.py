"""Typed boundary between the CPU engine controller and CUDA training libraries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from contextlib import AbstractContextManager


class TrainingDevice(Protocol):
    """Minimal device identity exposed to the training controller."""

    @property
    def type(self) -> str:
        """Return the backend device kind, such as ``cpu`` or ``cuda``."""
        ...


class TrainingTensor(Protocol):
    """Tensor operations used by checkpointing and model state boundaries."""

    @property
    def grad(self) -> TrainingTensor | None:
        """Return the accumulated gradient, when present."""
        ...

    @property
    def shape(self) -> Sequence[int]:
        """Return tensor dimensions."""
        ...

    @property
    def device(self) -> TrainingDevice:
        """Return the device that currently holds the tensor."""
        ...

    def to(self, device: TrainingDevice | str) -> TrainingTensor:
        """Move the tensor to one selected training device."""
        ...

    def detach(self) -> TrainingTensor:
        """Return a graph-free tensor view."""
        ...

    def cpu(self) -> TrainingTensor:
        """Return a host copy of this tensor."""
        ...

    def clone(self) -> TrainingTensor:
        """Return a tensor copy."""
        ...

    def float(self) -> TrainingTensor:
        """Return the value in FP32."""
        ...

    def exp(self) -> TrainingTensor:
        """Return the elementwise exponential."""
        ...

    def transpose(self, first_dimension: int, second_dimension: int) -> TrainingTensor:
        """Return a tensor with two dimensions exchanged."""
        ...

    def all(self) -> TrainingTensor:
        """Return whether all tensor values are true."""
        ...

    def item(self) -> int | float:
        """Extract a scalar value."""
        ...

    def backward(self) -> None:
        """Accumulate gradients from this scalar loss."""
        ...

    def __matmul__(self, other: TrainingTensor) -> TrainingTensor:
        """Perform matrix multiplication."""
        ...

    def __truediv__(self, divisor: int) -> TrainingTensor:
        """Scale the tensor by an integer divisor."""
        ...

    def __mul__(self, other: TrainingTensor) -> TrainingTensor:
        """Multiply tensor values elementwise."""
        ...

    def __add__(self, other: TrainingTensor) -> TrainingTensor:
        """Add tensor values elementwise."""
        ...


class TrainingModelOutput(Protocol):
    """CLIP outputs consumed by the symmetric contrastive loss."""

    image_embeds: TrainingTensor
    text_embeds: TrainingTensor


class TrainingModel(Protocol):
    """CLIP model operations required by one run."""

    @property
    def logit_scale(self) -> TrainingTensor:
        """Return the learned image/text temperature parameter."""
        ...

    def to(self, device: TrainingDevice) -> TrainingModel:
        """Move model parameters to a selected device."""
        ...

    def train(self) -> None:
        """Enable training behavior."""
        ...

    def eval(self) -> None:
        """Enable evaluation behavior."""
        ...

    def parameters(self) -> Iterable[TrainingTensor]:
        """Return all trainable model tensors."""
        ...

    def state_dict(self) -> Mapping[str, TrainingTensor]:
        """Return model weights and buffers."""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore model weights and buffers."""
        ...

    def __call__(self, **inputs: TrainingTensor) -> TrainingModelOutput:
        """Run the CLIP forward pass."""
        ...


class TrainingOptimizer(Protocol):
    """Optimizer operations serialized in an epoch checkpoint."""

    @property
    def state(self) -> Mapping[object, Mapping[str, object]]:
        """Return per-parameter optimizer state."""
        ...

    @property
    def param_groups(self) -> Sequence[Mapping[str, object]]:
        """Return optimizer configuration groups."""
        ...

    def zero_grad(self, *, set_to_none: bool) -> None:
        """Clear parameter gradients."""
        ...

    def step(self) -> None:
        """Apply one optimizer update."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return a checkpoint-ready optimizer state."""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore optimizer state."""
        ...


class TrainingScheduler(Protocol):
    """Optimizer-step scheduler operations."""

    def step(self) -> None:
        """Advance one successful optimizer step."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return scheduler state."""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore scheduler state."""
        ...


class ScaledTrainingLoss(Protocol):
    """Scaled scalar loss that supports gradient calculation."""

    def backward(self) -> None:
        """Accumulate scaled gradients."""
        ...


class TrainingScaler(Protocol):
    """Mixed-precision gradient scaler operations."""

    def scale(self, loss: TrainingTensor) -> ScaledTrainingLoss:
        """Scale a loss before backward."""
        ...

    def unscale_(self, optimizer: TrainingOptimizer) -> None:
        """Unscale gradients before clipping."""
        ...

    def step(self, optimizer: TrainingOptimizer) -> None:
        """Apply the update unless overflow was detected."""
        ...

    def update(self) -> None:
        """Update the dynamic scale after an optimizer attempt."""
        ...

    def get_scale(self) -> float:
        """Return the current dynamic scale."""
        ...

    def state_dict(self) -> dict[str, object]:
        """Return scaler state."""
        ...

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore scaler state."""
        ...


class CheckpointRng(Protocol):
    """Random-state operations stored at complete epoch boundaries."""

    def get_rng_state(self) -> TrainingTensor:
        """Capture the CPU generator state."""
        ...

    def set_rng_state(self, state: object) -> None:
        """Restore the CPU generator state."""
        ...

    def cuda_available(self) -> bool:
        """Return whether CUDA is available."""
        ...

    def get_cuda_rng_state_all(self) -> list[TrainingTensor]:
        """Capture all CUDA generator states."""
        ...

    def set_cuda_rng_state_all(self, states: Sequence[object]) -> None:
        """Restore all CUDA generator states."""
        ...


class TrainingBackend(CheckpointRng, Protocol):
    """Typed training operations implemented at the isolated Torch boundary."""

    def seed_everything(self, seed: int) -> None:
        """Seed Python, NumPy, and Torch generators."""
        ...

    def create_device(self, name: str) -> TrainingDevice:
        """Resolve one configured training device."""
        ...

    def is_cuda_device(self, device: TrainingDevice) -> bool:
        """Return whether a resolved device uses CUDA."""
        ...

    def clear_cuda_cache(self) -> None:
        """Release cached CUDA allocations when CUDA is available."""
        ...

    def is_out_of_memory(self, error: Exception) -> bool:
        """Identify Torch CUDA out-of-memory failures."""
        ...

    def create_optimizer_and_scheduler(  # noqa: PLR0913
        self,
        model: TrainingModel,
        *,
        learning_rate: float,
        betas: tuple[float, float],
        epsilon: float,
        weight_decay: float,
        lr_multiplier: Callable[[int], float],
    ) -> tuple[TrainingOptimizer, TrainingScheduler]:
        """Build the fixed AdamW recipe and optimizer-step scheduler."""
        ...

    def create_scaler(
        self,
        enabled: bool,
        factory: Callable[[bool], object] | None,
    ) -> TrainingScaler:
        """Create the CUDA gradient scaler or an explicitly injected test scaler."""
        ...

    def autocast(
        self,
        device: TrainingDevice,
        *,
        enabled: bool,
    ) -> AbstractContextManager[object]:
        """Return the device autocast context."""
        ...

    def no_grad(self) -> AbstractContextManager[object]:
        """Return the no-gradient context."""
        ...

    def contrastive_loss(
        self,
        model: TrainingModel,
        model_inputs: Mapping[str, object],
        device: TrainingDevice,
    ) -> TrainingTensor:
        """Calculate symmetric image-to-text and text-to-image CLIP loss."""
        ...

    def normalize(self, tensor: TrainingTensor) -> TrainingTensor:
        """L2-normalize the final embedding dimension."""
        ...

    def concatenate(self, tensors: Sequence[TrainingTensor]) -> TrainingTensor:
        """Concatenate host embedding batches along the sample dimension."""
        ...

    def is_finite(self, tensor: TrainingTensor) -> bool:
        """Return whether every tensor element is finite."""
        ...

    def scalar_value(self, tensor: TrainingTensor) -> float:
        """Convert a one-element tensor to Python float."""
        ...

    def gradients_are_finite(self, parameters: Sequence[TrainingTensor]) -> bool:
        """Check every present gradient before an FP32 update."""
        ...

    def clip_grad_norm(
        self,
        parameters: Sequence[TrainingTensor],
        max_norm: float,
    ) -> float:
        """Clip gradients and return the total norm."""
        ...

    def optimizer_step_count(self, optimizer: TrainingOptimizer) -> int:
        """Count successful AdamW updates from its parameter state."""
        ...

    def learning_rate(self, optimizer: TrainingOptimizer) -> float:
        """Read the active learning rate for progress reporting."""
        ...


__all__ = [
    "CheckpointRng",
    "TrainingBackend",
    "TrainingDevice",
    "TrainingModel",
    "TrainingModelOutput",
    "TrainingOptimizer",
    "TrainingScaler",
    "TrainingScheduler",
    "TrainingTensor",
]
