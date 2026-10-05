"""Pure calibration bounds and profile construction used by the GPU child."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

from datetime import UTC, datetime
from math import ceil
from typing import TYPE_CHECKING, Annotated

from pydantic import Field, StrictInt

from gods_watching.contracts.base import ContractModel
from gods_watching.training.memory import (
    GIB,
    INFERENCE_RESERVE_FRACTION,
    MIN_INFERENCE_RESERVE_BYTES,
    OPTIMIZER_RECIPE_ID,
    GpuSnapshot,
    MemoryProfile,
    current_source_fingerprint,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from gods_watching.contracts.training import TrainingConfig


class CalibrationModelShape(ContractModel):
    """Architecture dimensions needed for a conservative pre-measurement bound."""

    vision_layers: Annotated[StrictInt, Field(gt=0)]
    vision_hidden_size: Annotated[StrictInt, Field(gt=0)]
    vision_heads: Annotated[StrictInt, Field(gt=0)]
    text_layers: Annotated[StrictInt, Field(gt=0)]
    text_hidden_size: Annotated[StrictInt, Field(gt=0)]
    text_heads: Annotated[StrictInt, Field(gt=0)]
    patch_size: Annotated[StrictInt, Field(gt=0)]


class CalibrationHeadroom(ContractModel):
    """Memory arithmetic required before a GPU calibration child can start."""

    training_upper_bound_bytes: Annotated[StrictInt, Field(gt=0)]
    inference_reserve_bytes: Annotated[StrictInt, Field(gt=0)]
    guard_bytes: Annotated[StrictInt, Field(gt=0)]
    required_free_bytes: Annotated[StrictInt, Field(gt=0)]
    current_free_bytes: Annotated[StrictInt, Field(ge=0)]
    admitted: bool


class CalibrationRefusedError(RuntimeError):
    """The measured GPU snapshot cannot safely support a calibration profile."""


MAX_CALIBRATION_OPTIMIZER_ATTEMPTS: int = 16


def optimizer_update_succeeded(
    scale_before: float,
    scale_after: float,
    optimizer_step_before: int,
    optimizer_step_after: int,
) -> bool:
    """Count AMP updates only when optimizer state advanced without overflow."""
    return optimizer_step_after > optimizer_step_before and scale_after >= scale_before


class CheckpointLoadOptions(ContractModel):
    """Weights-only loader flags selected for one approved CLIP checkpoint format."""

    use_safetensors: bool
    weights_only: bool = True


def builtin_base_load_options(model_root: Path) -> CheckpointLoadOptions:
    """Select the locked B/16 ``pytorch_model.bin`` with safe weights-only loading."""
    if not (model_root / "pytorch_model.bin").is_file():
        raise CalibrationRefusedError("locked B/16 pytorch_model.bin is missing")
    if (model_root / "model.safetensors").exists():
        raise CalibrationRefusedError("base package must use only its locked pytorch_model.bin")
    return CheckpointLoadOptions(use_safetensors=False)


def exported_candidate_load_options(model_root: Path) -> CheckpointLoadOptions:
    """Keep generated candidate packages on the existing safetensors importer path."""
    if not (model_root / "model.safetensors").is_file():
        raise CalibrationRefusedError("candidate model.safetensors is missing")
    if (model_root / "pytorch_model.bin").exists():
        raise CalibrationRefusedError("candidate package must not contain pickle weights")
    return CheckpointLoadOptions(use_safetensors=True)


def bind_cuda_device_uuid(
    selected_gpu_uuid: str,
    visible_gpu_uuid: str,
    nvidia_gpu_uuid: str,
) -> str:
    """Require CUDA visibility and NVML telemetry to identify the selected GPU."""
    if not selected_gpu_uuid or visible_gpu_uuid != selected_gpu_uuid:
        raise CalibrationRefusedError("CUDA_VISIBLE_DEVICES is not bound to the selected GPU UUID")
    if nvidia_gpu_uuid != selected_gpu_uuid:
        raise CalibrationRefusedError("NVIDIA-SMI is reporting a different GPU UUID")
    return selected_gpu_uuid


def capture_cuda_snapshot(
    gpu_uuid: str,
    nvidia_snapshot: GpuSnapshot,
    query_mem_get_info: Callable[[], tuple[int, int]],
) -> GpuSnapshot:
    """Combine NVML and CUDA ``(free_bytes, total_bytes)`` conservatively.

    NVIDIA-SMI reports physical MiB while CUDA may report a smaller usable
    capacity after driver reservations. The UUID must match; CUDA free memory
    and usable capacity both clamp the returned NVML free value. NVML physical
    capacity remains canonical for profile identity and reserve math.
    """
    cuda_free_bytes, cuda_total_bytes = query_mem_get_info()
    if gpu_uuid != nvidia_snapshot.uuid:
        raise CalibrationRefusedError("GPU UUID changed during calibration startup")
    if cuda_total_bytes <= 0 or cuda_free_bytes < 0 or cuda_free_bytes > cuda_total_bytes:
        raise CalibrationRefusedError("CUDA returned invalid free/usable GPU memory values")
    capacity_bound = min(cuda_total_bytes, nvidia_snapshot.total_bytes)
    free_bytes = min(cuda_free_bytes, nvidia_snapshot.free_bytes, capacity_bound)
    return GpuSnapshot(
        uuid=gpu_uuid,
        total_bytes=nvidia_snapshot.total_bytes,
        free_bytes=free_bytes,
        observed_at=datetime.now(UTC),
    )


def conservative_training_upper_bound(
    config: TrainingConfig,
    shape: CalibrationModelShape,
    *,
    parameter_bytes: int,
) -> int:
    """Bound model, gradients, AdamW state, activations, and workspace before CUDA use."""
    batch = config.micro_batch_size
    element_bytes = 2 if config.mixed_precision == "fp16" else 4
    image_tokens = (224 // shape.patch_size) ** 2 + 1
    text_tokens = 77
    checkpoint_multiplier = 1 if config.gradient_checkpointing else 8

    vision_activation_bytes = (
        batch
        * image_tokens
        * shape.vision_hidden_size
        * element_bytes
        * shape.vision_layers
        * checkpoint_multiplier
    )
    text_activation_bytes = (
        batch
        * text_tokens
        * shape.text_hidden_size
        * element_bytes
        * shape.text_layers
        * checkpoint_multiplier
    )
    vision_attention_bytes = batch * shape.vision_heads * image_tokens**2 * element_bytes * 2
    text_attention_bytes = batch * shape.text_heads * text_tokens**2 * element_bytes * 2
    fp16_weight_cache_bytes = parameter_bytes // 2 if config.mixed_precision == "fp16" else 0

    # FP32 parameters, gradients, and both FP32 Adam moments are included. The
    # optimizer recipe uses foreach=False to avoid a parameter-sized temp copy.
    persistent_bytes = parameter_bytes * 4 + fp16_weight_cache_bytes
    activation_bytes = (
        vision_activation_bytes
        + text_activation_bytes
        + vision_attention_bytes
        + text_attention_bytes
    )
    workspace_and_context = GIB
    return persistent_bytes + activation_bytes + workspace_and_context


def check_calibration_headroom(
    config: TrainingConfig,
    shape: CalibrationModelShape,
    gpu: GpuSnapshot,
    *,
    parameter_bytes: int,
    guard_bytes: int = GIB // 2,
) -> CalibrationHeadroom:
    """Refuse calibration before model transfer unless bound plus reserve fits."""
    training_bound = conservative_training_upper_bound(
        config,
        shape,
        parameter_bytes=parameter_bytes,
    )
    reserve = max(
        MIN_INFERENCE_RESERVE_BYTES,
        ceil(gpu.total_bytes * INFERENCE_RESERVE_FRACTION),
    )
    required = training_bound + reserve + guard_bytes
    return CalibrationHeadroom(
        training_upper_bound_bytes=training_bound,
        inference_reserve_bytes=reserve,
        guard_bytes=guard_bytes,
        required_free_bytes=required,
        current_free_bytes=gpu.free_bytes,
        admitted=gpu.free_bytes >= required,
    )


def build_memory_profile(  # noqa: PLR0913
    config: TrainingConfig,
    gpu: GpuSnapshot,
    *,
    torch_version: str,
    cuda_version: str,
    processor_identity: str,
    model_sha256: str,
    baseline_reserved_bytes: int,
    peak_reserved_bytes: int,
    peak_allocated_bytes: int,
    baseline_device_used_bytes: int,
    peak_device_used_bytes: int,
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_state_bytes: int,
    activation_workspace_bytes: int,
    context_bytes: int,
) -> MemoryProfile:
    """Bind measured two-step AdamW peaks to the exact runtime/profile key."""
    return MemoryProfile(
        gpu_uuid=gpu.uuid,
        gpu_total_bytes=gpu.total_bytes,
        source_fingerprint=current_source_fingerprint(),
        torch_version=torch_version,
        cuda_version=cuda_version,
        processor_identity=processor_identity,
        model_sha256=model_sha256,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing,
        batch_size=config.micro_batch_size,
        optimizer_recipe=OPTIMIZER_RECIPE_ID,
        image_size=224,
        text_max_length=77,
        baseline_reserved_bytes=baseline_reserved_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
        baseline_device_used_bytes=baseline_device_used_bytes,
        peak_device_used_bytes=peak_device_used_bytes,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_workspace_bytes=activation_workspace_bytes,
        context_bytes=context_bytes,
        calibrated_at=datetime.now(UTC),
    )


__all__ = [
    "MAX_CALIBRATION_OPTIMIZER_ATTEMPTS",
    "CalibrationHeadroom",
    "CalibrationModelShape",
    "CalibrationRefusedError",
    "CheckpointLoadOptions",
    "bind_cuda_device_uuid",
    "build_memory_profile",
    "builtin_base_load_options",
    "capture_cuda_snapshot",
    "check_calibration_headroom",
    "conservative_training_upper_bound",
    "exported_candidate_load_options",
    "optimizer_update_succeeded",
]
