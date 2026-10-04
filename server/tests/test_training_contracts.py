import math

import pytest
from pydantic import ValidationError

from gods_watching.contracts.training import TrainingConfig


def test_training_config_defaults_and_boundaries() -> None:
    config = TrainingConfig()

    assert config.epochs == 30
    assert config.learning_rate == 1e-5
    assert config.micro_batch_size == 16
    assert config.weight_decay == 0.01
    assert config.gradient_accumulation == 4
    assert config.warmup_ratio == 0.05
    assert config.seed == 42
    assert config.early_stopping_patience == 5
    assert config.gradient_clipping_norm == 1.0
    assert config.mixed_precision == "fp16"
    assert config.gradient_checkpointing is True

    assert (
        TrainingConfig(
            epochs=1,
            learning_rate=1e-7,
            micro_batch_size=2,
            weight_decay=0,
            gradient_accumulation=1,
            warmup_ratio=0,
            seed=0,
            early_stopping_patience=None,
            gradient_clipping_norm=0.1,
            mixed_precision="fp32",
            gradient_checkpointing=False,
        ).epochs
        == 1
    )
    assert (
        TrainingConfig(
            epochs=100,
            learning_rate=0.001,
            micro_batch_size=128,
            weight_decay=0.2,
            gradient_accumulation=32,
            warmup_ratio=0.3,
            seed=2_147_483_647,
            early_stopping_patience=20,
            gradient_clipping_norm=10,
        ).epochs
        == 100
    )

    with pytest.raises(ValidationError):
        _ = TrainingConfig(micro_batch_size=1)
    with pytest.raises(ValidationError):
        _ = TrainingConfig(learning_rate=math.nan)
    with pytest.raises(ValidationError):
        _ = TrainingConfig.model_validate({"unsupported_option": True})
    with pytest.raises(ValidationError):
        config.epochs = 31
