"""Atomic, identity-bound training checkpoints with lazy Torch serialization."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import importlib
import math
import os
import random
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

MAX_CHECKPOINT_BYTES = 4 * 1024**3
_TEMPORARY_SPACE_FACTOR = 2
_IDENTITY_FIELDS = ("config_snapshot", "dataset_fingerprint", "source_fingerprint")

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import BinaryIO

    from gods_watching.training.engine_api import (
        CheckpointRng,
        TrainingModel,
        TrainingOptimizer,
        TrainingScaler,
        TrainingScheduler,
    )

    CheckpointSerializer = Callable[[Mapping[str, object], BinaryIO], None]
    CheckpointLoader = Callable[[Path], object]

_FINGERPRINT_LENGTH = 64


class _TorchStateTensor(Protocol):
    """Tensor conversion methods needed for a host-only state snapshot."""

    def detach(self) -> _TorchStateTensor: ...

    def cpu(self) -> _TorchStateTensor: ...


class _TorchSerializationApi(Protocol):
    """Torch serialization calls isolated from the CPU application package."""

    def save(self, state: Mapping[str, object], destination: BinaryIO) -> None: ...

    def load(
        self,
        path: Path,
        *,
        map_location: str,
        weights_only: bool,
    ) -> object: ...


class _NumpyIntegerArray(Protocol):
    """Integer-array conversion used to serialize legacy NumPy RNG state."""

    def tolist(self) -> list[int]: ...


class CheckpointError(RuntimeError):
    """Base class for incomplete or unsafe checkpoint operations."""


class CheckpointIdentityError(CheckpointError):
    """The checkpoint does not belong to the immutable current training inputs."""


class CheckpointSpaceError(CheckpointError):
    """Checkpoint serialization would exceed its bounded storage budget."""


@dataclass(frozen=True, slots=True)
class TrainingCheckpointProgress:
    """Resume counters and selected validation weights from a complete epoch."""

    next_epoch: int
    optimizer_steps: int
    best_metric: float | None
    bad_epochs: int
    best_epoch: int
    early_stopped: bool
    best_model: dict[str, object] | None


def save_checkpoint_atomic(
    state: Mapping[str, object],
    path: Path,
    *,
    serializer: CheckpointSerializer | None = None,
    max_checkpoint_bytes: int = MAX_CHECKPOINT_BYTES,
) -> None:
    """Serialize beside the destination and replace it only after file and directory fsync."""
    if max_checkpoint_bytes < 1:
        raise ValueError("checkpoint size limit must be positive")
    destination = Path(path)
    parent = destination.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_regular_destination(destination)
    required_temporary_space = max_checkpoint_bytes * _TEMPORARY_SPACE_FACTOR
    free_bytes = shutil.disk_usage(parent).free
    if free_bytes < required_temporary_space:
        message = (
            f"checkpoint needs {required_temporary_space} bytes of temporary space"
            f"; only {free_bytes} bytes are free"
        )
        raise CheckpointSpaceError(
            message
        )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            destination_stream = cast("BinaryIO", cast("object", temporary))
            (serializer or _torch_save)(state, destination_stream)
            temporary.flush()
            os.fsync(temporary.fileno())
            size = os.fstat(temporary.fileno()).st_size
        if size <= 0:
            raise CheckpointError("checkpoint serializer produced an empty file")
        if size > max_checkpoint_bytes:
            message = (
                f"checkpoint is {size} bytes, exceeding the "
                f"{max_checkpoint_bytes}-byte limit"
            )
            raise CheckpointSpaceError(message)
        _require_regular_destination(destination)
        _ = temporary_path.replace(destination)
        temporary_path = None
        directory_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _ = os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def load_checkpoint_verified(
    path: Path,
    snapshot: Mapping[str, object],
    *,
    loader: CheckpointLoader | None = None,
    max_checkpoint_bytes: int = MAX_CHECKPOINT_BYTES,
) -> dict[str, object]:
    """Load weights-only state on CPU and verify config, dataset, and source identity."""
    expected_identity = _checkpoint_identity(snapshot)
    checkpoint = Path(path)
    details = checkpoint.stat(follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode) or details.st_size <= 0:
        raise CheckpointError("checkpoint must be a non-empty regular file")
    if details.st_size > max_checkpoint_bytes:
        raise CheckpointSpaceError("checkpoint exceeds the configured size limit")
    payload = (loader or _torch_load)(checkpoint)
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint root must be a mapping")
    state = cast("dict[str, object]", payload)
    identity_value = state.get("identity")
    if not isinstance(identity_value, dict):
        raise CheckpointIdentityError("checkpoint identity metadata is missing")
    identity = cast("dict[str, object]", identity_value)
    actual_identity = {field: identity.get(field) for field in _IDENTITY_FIELDS}
    if actual_identity != expected_identity:
        raise CheckpointIdentityError(
            "checkpoint config, dataset fingerprint, or source fingerprint changed"
        )
    return state


def build_training_checkpoint(  # noqa: PLR0913
    identity: Mapping[str, object],
    *,
    model: TrainingModel,
    optimizer: TrainingOptimizer,
    scheduler: TrainingScheduler,
    scaler: TrainingScaler,
    torch_module: CheckpointRng,
    seed: int,
    next_epoch: int,
    optimizer_steps: int,
    best_metric: float | None,
    bad_epochs: int,
    best_epoch: int,
    best_model: dict[str, object] | None,
    early_stopped: bool,
) -> dict[str, object]:
    """Build a CPU-serializable epoch state including optimizer, sampler, and RNG state."""
    optimizer_state = _tree_to_cpu(optimizer.state_dict())
    return {
        "identity": dict(identity),
        "model": model_state_cpu(model),
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(torch_module),
        "sampler": {"seed": seed, "next_epoch": next_epoch},
        "best_model": _tree_to_cpu(best_model) if best_model is not None else None,
        "training": {
            "epoch": next_epoch,
            "optimizer_steps": optimizer_steps,
            "best_metric": best_metric,
            "bad_epochs": bad_epochs,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
        },
    }


def restore_training_checkpoint(  # noqa: PLR0913
    state: dict[str, object],
    *,
    model: TrainingModel,
    optimizer: TrainingOptimizer,
    scheduler: TrainingScheduler,
    scaler: TrainingScaler,
    torch_module: CheckpointRng,
) -> TrainingCheckpointProgress:
    """Restore model/optimizer/scheduler/scaler and all RNG state from one epoch boundary."""
    model.load_state_dict(cast("dict[str, object]", state["model"]))
    optimizer.load_state_dict(cast("dict[str, object]", state["optimizer"]))
    scheduler.load_state_dict(cast("dict[str, object]", state["scheduler"]))
    scaler.load_state_dict(cast("dict[str, object]", state["scaler"]))
    restore_rng_state(cast("dict[str, object]", state["rng"]), torch_module)
    training = cast("dict[str, object]", state["training"])
    next_epoch = training.get("epoch")
    optimizer_steps = training.get("optimizer_steps")
    bad_epochs = training.get("bad_epochs")
    best_metric = training.get("best_metric")
    best_epoch = training.get("best_epoch")
    early_stopped = training.get("early_stopped")
    best_model_value = state.get("best_model")
    if (
        type(next_epoch) is not int
        or next_epoch < 0
        or type(optimizer_steps) is not int
        or optimizer_steps < 0
        or type(bad_epochs) is not int
        or bad_epochs < 0
        or type(best_epoch) is not int
        or best_epoch < 0
        or best_epoch > next_epoch
        or type(early_stopped) is not bool
        or (
            best_metric is not None
            and (not isinstance(best_metric, int | float) or not math.isfinite(float(best_metric)))
        )
        or (best_metric is not None and not isinstance(best_model_value, dict))
    ):
        raise CheckpointError("checkpoint training progress is malformed")
    return TrainingCheckpointProgress(
        next_epoch=next_epoch,
        optimizer_steps=optimizer_steps,
        best_metric=float(best_metric) if best_metric is not None else None,
        bad_epochs=bad_epochs,
        best_epoch=best_epoch,
        early_stopped=early_stopped,
        best_model=cast("dict[str, object]", best_model_value)
        if best_model_value is not None
        else None,
    )


def model_state_cpu(model: TrainingModel) -> dict[str, object]:
    """Clone model state to host memory before serialization."""
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def capture_rng_state(torch_module: CheckpointRng) -> dict[str, object]:
    """Capture Python, NumPy primitive, Torch CPU, and optional CUDA RNG state."""
    np_state = np.random.get_state()  # noqa: NPY002 - preserve the global stream for resume.
    numpy_state: dict[str, object] = {
        "algorithm": str(np_state[0]),
        "keys": cast("_NumpyIntegerArray", cast("object", np_state[1])).tolist(),
        "position": int(np_state[2]),
        "has_gauss": int(np_state[3]),
        "cached_gaussian": float(np_state[4]),
    }
    cuda_state = torch_module.get_cuda_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch": torch_module.get_rng_state().cpu(),
        "cuda": [value.cpu() for value in cuda_state],
    }


def restore_rng_state(state: dict[str, object], torch_module: CheckpointRng) -> None:
    """Restore random generators while avoiding NumPy ndarray values in safe Torch load."""
    random.setstate(cast("tuple[object, ...]", state["python"]))
    numpy_state = cast("dict[str, object]", state["numpy"])
    keys = np.asarray(cast("list[int]", numpy_state["keys"]), dtype=np.uint32)
    np.random.set_state(  # noqa: NPY002 - restore the exact global stream from the checkpoint.
        (
            cast("str", numpy_state["algorithm"]),
            keys,
            cast("int", numpy_state["position"]),
            cast("int", numpy_state["has_gauss"]),
            cast("float", numpy_state["cached_gaussian"]),
        )
    )
    torch_module.set_rng_state(state["torch"])
    cuda_state = cast("list[object]", state["cuda"])
    if cuda_state and torch_module.cuda_available():
        torch_module.set_cuda_rng_state_all(cuda_state)


def _tree_to_cpu(value: object) -> object:
    if hasattr(value, "detach"):
        return cast("_TorchStateTensor", value).detach().cpu()
    if isinstance(value, dict):
        values = cast("dict[object, object]", value)
        return {key: _tree_to_cpu(item) for key, item in values.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in cast("list[object]", value)]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in cast("tuple[object, ...]", value))
    return value


def _checkpoint_identity(snapshot: Mapping[str, object]) -> dict[str, object]:
    identity = {field: snapshot.get(field) for field in _IDENTITY_FIELDS}
    if not isinstance(identity["config_snapshot"], dict):
        raise CheckpointIdentityError("training config snapshot is missing")
    for field in ("dataset_fingerprint", "source_fingerprint"):
        value = identity[field]
        if (
            not isinstance(value, str)
            or len(value) != _FINGERPRINT_LENGTH
            or any(character not in "0123456789abcdef" for character in value)
        ):
            message = f"training {field.replace('_', ' ')} is invalid"
            raise CheckpointIdentityError(message)
    return identity


def _require_regular_destination(path: Path) -> None:
    if path.is_symlink():
        raise CheckpointError("checkpoint destination must not be a symlink")
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(details.st_mode):
        raise CheckpointError("checkpoint destination must be a regular file")


def _torch_save(state: Mapping[str, object], destination: BinaryIO) -> None:
    torch = cast("_TorchSerializationApi", cast("object", importlib.import_module("torch")))
    torch.save(dict(state), destination)


def _torch_load(path: Path) -> object:
    torch = cast("_TorchSerializationApi", cast("object", importlib.import_module("torch")))
    return torch.load(path, map_location="cpu", weights_only=True)


__all__ = [
    "MAX_CHECKPOINT_BYTES",
    "CheckpointError",
    "CheckpointIdentityError",
    "CheckpointSpaceError",
    "TrainingCheckpointProgress",
    "build_training_checkpoint",
    "capture_rng_state",
    "load_checkpoint_verified",
    "model_state_cpu",
    "restore_rng_state",
    "restore_training_checkpoint",
    "save_checkpoint_atomic",
]
