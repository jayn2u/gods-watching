from __future__ import annotations

import os
from typing import TYPE_CHECKING

from gods_watching.training import torch_backend

if TYPE_CHECKING:
    import pytest


class _FakeCuda:
    OutOfMemoryError: type[Exception] = MemoryError

    def is_available(self) -> bool:
        return False

    def manual_seed_all(self, _seed: int) -> None:
        return None


class _FakeCudnn:
    deterministic: bool = False
    benchmark: bool = True


class _FakeBackends:
    cudnn: _FakeCudnn = _FakeCudnn()


class _FakeTorch:
    cuda: _FakeCuda = _FakeCuda()
    backends: _FakeBackends = _FakeBackends()

    def __init__(self) -> None:
        self.deterministic_algorithms: bool = False
        self.seed: int | None = None

    def use_deterministic_algorithms(self, mode: bool) -> None:
        self.deterministic_algorithms = mode

    def manual_seed(self, seed: int) -> None:
        self.seed = seed


def test_training_backend_sets_determinism_before_loading_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fake_torch = _FakeTorch()

    def import_module(name: str) -> _FakeTorch:
        assert name == "torch"
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        return fake_torch

    monkeypatch.setattr(
        "gods_watching.training.torch_backend.importlib.import_module",
        import_module,
    )
    backend = torch_backend.create_torch_training_backend()
    backend.seed_everything(41)

    assert fake_torch.deterministic_algorithms
    assert fake_torch.backends.cudnn.deterministic
    assert not fake_torch.backends.cudnn.benchmark
    assert fake_torch.seed == 41
