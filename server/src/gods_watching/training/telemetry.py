"""Training-worker-only GPU telemetry; importing this module does not load torch."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from importlib import import_module
from typing import Protocol, cast

from gods_watching.training.calibration import CalibrationRefusedError, capture_cuda_snapshot
from gods_watching.training.memory import GpuSnapshot

_NVIDIA_TELEMETRY_FIELDS = 3


class _CudaDeviceProperties(Protocol):
    uuid: object


class _CudaApi(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def init(self) -> None: ...

    def get_device_properties(self, device: object) -> _CudaDeviceProperties: ...

    def mem_get_info(self, device: object) -> tuple[int, int]: ...


class _TorchApi(Protocol):
    cuda: _CudaApi

    def device(self, device: str) -> object: ...


def _nvidia_snapshot(device: int | str) -> GpuSnapshot:
    command = [
        "nvidia-smi",
        f"--id={device}",
        "--query-gpu=uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)  # noqa: S603
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise CalibrationRefusedError("GPU telemetry returned an ambiguous device list")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != _NVIDIA_TELEMETRY_FIELDS:
        raise CalibrationRefusedError("GPU telemetry returned a malformed memory row")
    uuid, total_mib, free_mib = fields
    mib = 1024**2
    return GpuSnapshot(
        uuid=uuid,
        total_bytes=int(total_mib) * mib,
        free_bytes=int(free_mib) * mib,
        observed_at=datetime.now(UTC),
    )


def _canonical_gpu_uuid(value: object) -> str:
    if value is None:
        raise CalibrationRefusedError("could not verify CUDA device UUID")
    try:
        normalized = str(value).strip()
    except Exception as error:
        raise CalibrationRefusedError("could not verify CUDA device UUID") from error
    if not normalized:
        raise CalibrationRefusedError("could not verify CUDA device UUID")
    if normalized.upper().startswith("GPU-"):
        normalized = normalized[4:]
    return normalized.casefold()


def current_gpu_snapshot(*, device_uuid: str | None = None) -> GpuSnapshot:
    """Combine NVML and initialized CUDA free memory for the selected GPU UUID."""
    selected_uuid = device_uuid or os.environ.get("GW_TRAINING_GPU_UUID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if selected_uuid is None and visible_devices.startswith("GPU-"):
        selected_uuid = visible_devices.split(",", maxsplit=1)[0]
    nvidia = _nvidia_snapshot(selected_uuid or os.environ.get("GW_TRAINING_GPU_INDEX", "0"))
    torch = cast("_TorchApi", cast("object", import_module("torch")))

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise CalibrationRefusedError("training worker must see exactly one CUDA device")
    torch.cuda.init()
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch_uuid = _canonical_gpu_uuid(getattr(properties, "uuid", None))
    nvidia_uuid = _canonical_gpu_uuid(nvidia.uuid)
    if torch_uuid != nvidia_uuid:
        raise CalibrationRefusedError("CUDA and NVIDIA-SMI are reporting different GPU UUIDs")
    if selected_uuid is not None and _canonical_gpu_uuid(selected_uuid) != nvidia_uuid:
        raise CalibrationRefusedError("NVIDIA-SMI is reporting a different selected GPU UUID")
    if visible_devices.startswith("GPU-") and _canonical_gpu_uuid(visible_devices) != nvidia_uuid:
        raise CalibrationRefusedError("CUDA_VISIBLE_DEVICES is not bound to the selected GPU UUID")
    return capture_cuda_snapshot(
        nvidia.uuid,
        nvidia,
        lambda: torch.cuda.mem_get_info(device),
    )


__all__ = ["current_gpu_snapshot"]
