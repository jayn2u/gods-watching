from __future__ import annotations

import builtins
import importlib.util
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

import pytest

from gods_watching.contracts.training import TrainingConfig
from gods_watching.training.calibration import CalibrationRefusedError
from gods_watching.training.memory import GpuSnapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class _Cudnn:
    deterministic: bool
    benchmark: bool

    def __init__(self) -> None:
        self.deterministic = False
        self.benchmark = True


class _Backends:
    cudnn: _Cudnn

    def __init__(self, cudnn: _Cudnn) -> None:
        self.cudnn = cudnn


class _Cuda:
    _before_probe: Callable[[], None]

    def __init__(self, before_probe: Callable[[], None]) -> None:
        self._before_probe = before_probe

    def is_available(self) -> bool:
        self._before_probe()
        return False


class _FakeTorch(ModuleType):
    backends: _Backends
    cuda: _Cuda
    __version__: str
    deterministic_algorithms: bool

    def __init__(self, cudnn: _Cudnn, before_probe: Callable[[], None]) -> None:
        super().__init__("torch")
        self.backends = _Backends(cudnn)
        self.cuda = _Cuda(before_probe)
        self.__version__ = "2.7.1+cu128"
        self.deterministic_algorithms = False

    def use_deterministic_algorithms(self, *, mode: bool) -> None:
        self.deterministic_algorithms = mode


class _TransformerStub(ModuleType):
    CLIPModel: type
    CLIPProcessor: type

    def __init__(self) -> None:
        super().__init__("transformers")
        self.CLIPModel = object
        self.CLIPProcessor = object


def test_memory_calibration_uses_the_training_policy_before_first_cuda_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path(__file__).parents[2] / "qa" / "training" / "calibrate_memory.py"
    spec = importlib.util.spec_from_file_location("gw_calibrate_memory_test", source)
    assert spec is not None
    assert spec.loader is not None
    calibration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibration)

    cudnn = _Cudnn()

    fake_torch: _FakeTorch

    def check_cuda_policy() -> None:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        assert fake_torch.deterministic_algorithms
        assert cudnn.deterministic
        assert not cudnn.benchmark

    fake_torch = _FakeTorch(cudnn, check_cuda_policy)
    transformer_stub = _TransformerStub()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", transformer_stub)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", "wrong")
    original_import = builtins.__import__

    def check_before_import(
        name: str,
        _globals: dict[str, object] | None = None,
        _locals: dict[str, object] | None = None,
        _fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "torch":
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        return cast(
            "ModuleType",
            original_import(name, _globals, _locals, _fromlist, level),
        )

    monkeypatch.setattr(builtins, "__import__", check_before_import)
    gpu = GpuSnapshot(
        uuid="GPU-17913b0a-8144-5f39-7062-15265e5dca33",
        total_bytes=16 * 1024**3,
        free_bytes=12 * 1024**3,
        observed_at=datetime.now(UTC),
    )
    child_attribute = "_run_calibration_child"
    calibrate_child = cast(
        "Callable[[TrainingConfig, Path, Path, GpuSnapshot], object]",
        cast("object", getattr(calibration, child_attribute)),
    )

    with pytest.raises(CalibrationRefusedError, match="CUDA is unavailable"):
        _ = calibrate_child(
            TrainingConfig(),
            tmp_path / "models",
            tmp_path / "lock.json",
            gpu,
        )
