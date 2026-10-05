"""One fixed deterministic CUDA policy shared by training and calibration."""

from __future__ import annotations

import os
from typing import Protocol, cast

_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class _CudnnBackend(Protocol):
    deterministic: bool
    benchmark: bool


class _TorchBackends(Protocol):
    cudnn: _CudnnBackend


class _DeterministicTorch(Protocol):
    backends: _TorchBackends

    def use_deterministic_algorithms(self, mode: bool) -> None: ...


def prepare_deterministic_cuda_environment() -> None:
    """Set cuBLAS determinism before importing Torch or initializing CUDA."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIG


def configure_torch_determinism(torch_module: object) -> None:
    """Apply the same algorithm and cuDNN controls in both CUDA code paths."""
    prepare_deterministic_cuda_environment()
    torch_api = cast("_DeterministicTorch", torch_module)
    torch_api.use_deterministic_algorithms(mode=True)
    cudnn = torch_api.backends.cudnn
    cudnn.deterministic = True
    cudnn.benchmark = False


__all__ = [
    "configure_torch_determinism",
    "prepare_deterministic_cuda_environment",
]
