from __future__ import annotations

import importlib
import sys


def test_engine_data_module_imports_without_optional_training_sdk() -> None:
    module_name = "gods_watching.training.engine_data"
    _ = importlib.import_module(module_name)

    assert module_name in sys.modules
