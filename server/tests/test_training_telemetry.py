from __future__ import annotations

import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import override

import pytest

from gods_watching.training.calibration import CalibrationRefusedError
from gods_watching.training.telemetry import current_gpu_snapshot


class _FakeCuda:
    _uuid: object | None

    def __init__(self, uuid: object | None) -> None:
        self._uuid = uuid

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1

    def init(self) -> None:
        return None

    def get_device_properties(self, _index: object) -> SimpleNamespace:
        return SimpleNamespace(uuid=self._uuid)

    def mem_get_info(self, _device: object) -> tuple[int, int]:
        return 8 * 1024**3, 16 * 1024**3


class _FakeTorch(ModuleType):
    cuda: _FakeCuda

    def __init__(self, uuid: object | None) -> None:
        super().__init__("torch")
        self.cuda = _FakeCuda(uuid)

    def device(self, value: str) -> str:
        return value


def _torch_module(*, uuid: object | None) -> ModuleType:
    return _FakeTorch(uuid)


def _nvidia_result(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
    result: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="GPU-physical-one, 16384, 8192\n",
        stderr="",
    )
    return result


class _CudaUuid:
    _value: str

    def __init__(self, value: str) -> None:
        self._value = value

    @override
    def __str__(self) -> str:
        return self._value


def test_numeric_cuda_visibility_is_checked_against_torch_physical_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GW_TRAINING_GPU_INDEX", "1")
    monkeypatch.delenv("GW_TRAINING_GPU_UUID", raising=False)
    monkeypatch.setattr("gods_watching.training.telemetry.subprocess.run", _nvidia_result)
    monkeypatch.setitem(sys.modules, "torch", _torch_module(uuid="physical-zero"))

    with pytest.raises(CalibrationRefusedError, match="different GPU UUID"):
        _ = current_gpu_snapshot()


def test_torch_cuda_uuid_object_is_canonicalized_through_its_string_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GW_TRAINING_GPU_INDEX", "1")
    monkeypatch.delenv("GW_TRAINING_GPU_UUID", raising=False)
    monkeypatch.setattr("gods_watching.training.telemetry.subprocess.run", _nvidia_result)
    torch = _torch_module(uuid=_CudaUuid("GPU-physical-one"))
    monkeypatch.setitem(sys.modules, "torch", torch)

    snapshot = current_gpu_snapshot()

    assert snapshot.uuid == "GPU-physical-one"
    assert snapshot.free_bytes == 8 * 1024**3


def test_telemetry_fails_closed_when_torch_cannot_report_device_uuid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GW_TRAINING_GPU_INDEX", "1")
    monkeypatch.delenv("GW_TRAINING_GPU_UUID", raising=False)
    monkeypatch.setattr("gods_watching.training.telemetry.subprocess.run", _nvidia_result)
    monkeypatch.setitem(sys.modules, "torch", _torch_module(uuid=None))

    with pytest.raises(CalibrationRefusedError, match="could not verify CUDA device UUID"):
        _ = current_gpu_snapshot()
