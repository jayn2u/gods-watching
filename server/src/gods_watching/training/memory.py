"""CPU-safe memory profiles and fail-closed GPU admission decisions."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, StrictInt, model_validator

from gods_watching.contracts.base import ContractModel

if TYPE_CHECKING:
    from gods_watching.contracts.training import TrainingConfig

GIB: int = 1024**3
MIN_INFERENCE_RESERVE_BYTES: int = 2 * GIB
INFERENCE_RESERVE_FRACTION: float = 0.15
MAX_GPU_SNAPSHOT_AGE: timedelta = timedelta(seconds=30)
MAX_PROFILE_AGE: timedelta = timedelta(days=30)
MAX_FUTURE_CLOCK_SKEW: timedelta = timedelta(seconds=10)
OPTIMIZER_RECIPE_ID: str = "adamw-betas-0.9-0.999-eps-1e-8-foreach-false-v1"
SUPPORTED_TORCH_VERSION: str = "2.7.1+cu128"
SUPPORTED_CUDA_VERSION: str = "12.8"
SUPPORTED_MODEL_ID: str = "openai/clip-vit-base-patch16"
SUPPORTED_MODEL_REVISION: str = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
SUPPORTED_PROCESSOR_IDENTITY: str = f"{SUPPORTED_MODEL_ID}@{SUPPORTED_MODEL_REVISION}"
SUPPORTED_IMAGE_SIZE: int = 224
SUPPORTED_TEXT_MAX_LENGTH: int = 77
DEFAULT_MODEL_ROOT: Path = Path("/models/clip")
_HASH_CHUNK_SIZE = 1024 * 1024

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
PositiveBytes = Annotated[StrictInt, Field(gt=0)]
NonNegativeBytes = Annotated[StrictInt, Field(ge=0)]


class MemoryAdmissionError(RuntimeError):
    """Base class for a stale or unsupported memory admission input."""


class UnsupportedMemoryProfileError(MemoryAdmissionError):
    """The exact GPU, source, model, recipe, or config has no valid profile."""


class StaleGpuSnapshotError(MemoryAdmissionError):
    """GPU memory telemetry is too old or belongs to another device."""


class GpuSnapshot(ContractModel):
    """Fresh free/total device memory telemetry tied to one physical GPU UUID."""

    uuid: str = Field(min_length=1, max_length=128)
    total_bytes: PositiveBytes
    free_bytes: NonNegativeBytes
    observed_at: datetime

    @model_validator(mode="after")
    def _valid_snapshot(self) -> GpuSnapshot:
        if self.free_bytes > self.total_bytes:
            raise ValueError("GPU free memory cannot exceed total memory")
        if self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise ValueError("GPU observation time must include a timezone")
        return self


class MemoryProfile(ContractModel):
    """Immutable measured peak and exact inputs for one supported config profile."""

    gpu_uuid: str = Field(min_length=1, max_length=128)
    gpu_total_bytes: PositiveBytes
    source_fingerprint: Sha256
    torch_version: str = Field(min_length=1, max_length=64)
    cuda_version: str = Field(min_length=1, max_length=32)
    processor_identity: str = Field(min_length=1, max_length=512)
    model_sha256: Sha256
    mixed_precision: Literal["fp16", "fp32"]
    gradient_checkpointing: bool
    batch_size: Annotated[StrictInt, Field(ge=2, le=128)]
    optimizer_recipe: str = Field(min_length=1, max_length=128)
    image_size: Annotated[StrictInt, Field(gt=0)]
    text_max_length: Annotated[StrictInt, Field(gt=0)]
    baseline_reserved_bytes: NonNegativeBytes
    peak_reserved_bytes: PositiveBytes
    peak_allocated_bytes: PositiveBytes
    baseline_device_used_bytes: NonNegativeBytes
    peak_device_used_bytes: PositiveBytes
    parameter_bytes: PositiveBytes
    gradient_bytes: PositiveBytes
    optimizer_state_bytes: PositiveBytes
    activation_workspace_bytes: NonNegativeBytes
    context_bytes: NonNegativeBytes
    calibrated_at: datetime

    @model_validator(mode="after")
    def _valid_measurements(self) -> MemoryProfile:
        if self.peak_reserved_bytes < self.baseline_reserved_bytes:
            raise ValueError("peak reserved memory cannot be below baseline")
        if self.peak_allocated_bytes > self.peak_reserved_bytes:
            raise ValueError("peak allocated memory cannot exceed peak reserved memory")
        if self.peak_device_used_bytes < self.baseline_device_used_bytes:
            raise ValueError("peak device use cannot be below baseline")
        if self.calibrated_at.tzinfo is None or self.calibrated_at.utcoffset() is None:
            raise ValueError("memory profile timestamp must include a timezone")
        return self

    @property
    def profile_identity(self) -> str:
        """Hash only immutable profile-key fields, excluding measured values and age."""
        identity = {
            "gpu_uuid": self.gpu_uuid,
            "gpu_total_bytes": self.gpu_total_bytes,
            "source_fingerprint": self.source_fingerprint,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "processor_identity": self.processor_identity,
            "model_sha256": self.model_sha256,
            "mixed_precision": self.mixed_precision,
            "gradient_checkpointing": self.gradient_checkpointing,
            "batch_size": self.batch_size,
            "optimizer_recipe": self.optimizer_recipe,
            "image_size": self.image_size,
            "text_max_length": self.text_max_length,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def training_peak_bytes(self) -> int:
        """Use reserved, device-wide, and tensor component peaks conservatively."""
        allocator_delta = self.peak_reserved_bytes - self.baseline_reserved_bytes
        device_delta = self.peak_device_used_bytes - self.baseline_device_used_bytes
        component_bound = (
            self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
            + self.activation_workspace_bytes
            + self.context_bytes
        )
        return max(allocator_delta, device_delta, component_bound)


class MemoryEstimate(ContractModel):
    """Conservative additional memory required by one exact profile."""

    gpu_uuid: str
    gpu_total_bytes: PositiveBytes
    gpu_free_bytes: NonNegativeBytes
    gpu_observed_at: datetime
    training_peak_bytes: PositiveBytes
    reserve_bytes: PositiveBytes
    required_bytes: PositiveBytes
    profile_identity: Sha256


class AdmissionResult(ContractModel):
    """Explain whether a current GPU snapshot can safely admit one profile."""

    admitted: bool
    required_bytes: PositiveBytes
    free_bytes: NonNegativeBytes
    reserve_bytes: PositiveBytes
    reason: Literal["admitted", "insufficient_free_memory"]


@lru_cache(maxsize=1)
def current_source_fingerprint() -> str:
    """Hash the installed Python package so source changes invalidate profiles."""
    package_root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=2)
def model_directory_sha256(model_root: Path) -> str:
    """Hash all regular files in one immutable local model/processor package."""
    root = model_root.expanduser()
    if root.is_symlink() or not root.is_dir():
        raise UnsupportedMemoryProfileError("pinned local B/16 model package is unavailable")
    root = root.resolve(strict=True)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file() or path.is_symlink())
    if not files:
        raise UnsupportedMemoryProfileError("pinned local B/16 model package is empty")
    for path in files:
        if path.is_symlink() or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode):
            raise UnsupportedMemoryProfileError("pinned model package contains an unsafe file")
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_HASH_CHUNK_SIZE), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def current_model_sha256() -> str:
    """Hash the operator-mounted model package used by the training worker."""
    model_root = Path(os.environ.get("GW_TRAINING_MODEL_ROOT", str(DEFAULT_MODEL_ROOT)))
    try:
        return model_directory_sha256(model_root)
    except OSError as exc:
        raise UnsupportedMemoryProfileError(
            "pinned local B/16 model package cannot be verified"
        ) from exc


def _validate_snapshot(gpu: GpuSnapshot) -> datetime:
    now = datetime.now(UTC)
    observed_at = gpu.observed_at.astimezone(UTC)
    age = now - observed_at
    if age > MAX_GPU_SNAPSHOT_AGE or age < -MAX_FUTURE_CLOCK_SKEW:
        raise StaleGpuSnapshotError("GPU memory snapshot is stale")
    return observed_at


def _validate_profile(  # noqa: C901
    config: TrainingConfig,
    profile: MemoryProfile | None,
    gpu: GpuSnapshot,
    *,
    now: datetime,
) -> MemoryProfile:
    if profile is None:
        raise UnsupportedMemoryProfileError("no calibrated profile for this config")
    calibrated_at = profile.calibrated_at.astimezone(UTC)
    age = now - calibrated_at
    if age > MAX_PROFILE_AGE or age < -MAX_FUTURE_CLOCK_SKEW:
        raise UnsupportedMemoryProfileError("memory profile is stale")
    if profile.gpu_uuid != gpu.uuid:
        raise UnsupportedMemoryProfileError("memory profile GPU UUID does not match")
    if profile.gpu_total_bytes != gpu.total_bytes:
        raise UnsupportedMemoryProfileError("memory profile GPU capacity does not match")
    if profile.source_fingerprint != current_source_fingerprint():
        raise UnsupportedMemoryProfileError("memory profile source fingerprint does not match")
    if profile.torch_version != SUPPORTED_TORCH_VERSION:
        raise UnsupportedMemoryProfileError("memory profile Torch version does not match")
    if profile.cuda_version != SUPPORTED_CUDA_VERSION:
        raise UnsupportedMemoryProfileError("memory profile CUDA version does not match")
    if profile.processor_identity != SUPPORTED_PROCESSOR_IDENTITY:
        raise UnsupportedMemoryProfileError("memory profile processor identity does not match")
    if profile.model_sha256 != current_model_sha256():
        raise UnsupportedMemoryProfileError("memory profile local model hash does not match")
    if (
        profile.image_size != SUPPORTED_IMAGE_SIZE
        or profile.text_max_length != SUPPORTED_TEXT_MAX_LENGTH
    ):
        raise UnsupportedMemoryProfileError("memory profile input shape does not match")
    if profile.optimizer_recipe != OPTIMIZER_RECIPE_ID:
        raise UnsupportedMemoryProfileError("memory profile optimizer recipe does not match")
    if (
        profile.batch_size != config.micro_batch_size
        or profile.mixed_precision != config.mixed_precision
        or profile.gradient_checkpointing != config.gradient_checkpointing
    ):
        raise UnsupportedMemoryProfileError("no calibrated profile for this config")
    return profile


def estimate_memory(
    config: TrainingConfig,
    profile: MemoryProfile | None,
    gpu: GpuSnapshot,
) -> MemoryEstimate:
    """Estimate training peak plus a non-configurable inference safety reserve."""
    observed_at = _validate_snapshot(gpu)
    valid_profile = _validate_profile(config, profile, gpu, now=datetime.now(UTC))
    reserve_bytes = max(
        MIN_INFERENCE_RESERVE_BYTES,
        math.ceil(gpu.total_bytes * INFERENCE_RESERVE_FRACTION),
    )
    training_peak_bytes = valid_profile.training_peak_bytes
    return MemoryEstimate(
        gpu_uuid=gpu.uuid,
        gpu_total_bytes=gpu.total_bytes,
        gpu_free_bytes=gpu.free_bytes,
        gpu_observed_at=observed_at,
        training_peak_bytes=training_peak_bytes,
        reserve_bytes=reserve_bytes,
        required_bytes=training_peak_bytes + reserve_bytes,
        profile_identity=valid_profile.profile_identity,
    )


def assess_admission(estimate: MemoryEstimate, gpu: GpuSnapshot) -> AdmissionResult:
    """Apply a fresh free-memory check to an earlier exact-profile estimate."""
    _ = _validate_snapshot(gpu)
    if estimate.gpu_uuid != gpu.uuid or estimate.gpu_total_bytes != gpu.total_bytes:
        raise StaleGpuSnapshotError("GPU snapshot does not match the estimated profile device")
    admitted = estimate.required_bytes <= gpu.free_bytes
    return AdmissionResult(
        admitted=admitted,
        required_bytes=estimate.required_bytes,
        free_bytes=gpu.free_bytes,
        reserve_bytes=estimate.reserve_bytes,
        reason="admitted" if admitted else "insufficient_free_memory",
    )


__all__ = [
    "OPTIMIZER_RECIPE_ID",
    "SUPPORTED_CUDA_VERSION",
    "SUPPORTED_IMAGE_SIZE",
    "SUPPORTED_MODEL_ID",
    "SUPPORTED_MODEL_REVISION",
    "SUPPORTED_PROCESSOR_IDENTITY",
    "SUPPORTED_TEXT_MAX_LENGTH",
    "SUPPORTED_TORCH_VERSION",
    "AdmissionResult",
    "GpuSnapshot",
    "MemoryAdmissionError",
    "MemoryEstimate",
    "MemoryProfile",
    "StaleGpuSnapshotError",
    "UnsupportedMemoryProfileError",
    "assess_admission",
    "current_model_sha256",
    "current_source_fingerprint",
    "estimate_memory",
    "model_directory_sha256",
]
