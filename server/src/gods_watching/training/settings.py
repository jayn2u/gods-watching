"""Operator-managed allowlist for the local training dataset root."""

# ruff: noqa: TRY003, EM101

from pathlib import Path
from typing import ClassVar

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainingDatasetNotConfiguredError(RuntimeError):
    """Raised when no operator-approved training dataset root is configured."""


class TrainingSettings(BaseSettings):
    """Read the sole allowed dataset root from ``GW_TRAINING_DATASET_ROOT``."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="GW_TRAINING_",
        extra="ignore",
        frozen=True,
    )

    dataset_root: Path | None = None
    memory_profiles_path: Path = Path("/var/lib/gods-watching/training/memory-profiles.json")
    training_root: Path = Path("/var/lib/gods-watching/training")
    model_assets_root: Path = Field(
        default=Path("/models"),
        validation_alias=AliasChoices(
            "GW_TRAINING_MODEL_ASSETS_ROOT",
            "GW_MODEL_ASSETS_ROOT",
        ),
    )
    model_lock_path: Path = Field(
        default=Path("/opt/gods-watching/assets/models.lock.json"),
        validation_alias=AliasChoices(
            "GW_TRAINING_MODEL_LOCK_PATH",
            "GW_MODEL_LOCK_PATH",
        ),
    )
    request_timeout_seconds: float = Field(default=20.0, gt=0, le=60)
    request_poll_interval_seconds: float = Field(default=0.2, gt=0, le=5)
    request_lease_seconds: float = Field(default=10.0, gt=0, le=60)

    @field_validator("dataset_root")
    @classmethod
    def _require_absolute_dataset_root(cls, value: Path | None) -> Path | None:
        if value is not None and not value.is_absolute():
            raise ValueError("configured training dataset root must be an absolute path")
        return value

    @field_validator("model_assets_root")
    @classmethod
    def _require_absolute_model_assets_root(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("configured model assets root must be an absolute path")
        return value

    @field_validator("model_lock_path")
    @classmethod
    def _require_absolute_model_lock_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("configured model lock path must be an absolute path")
        return value

    def require_dataset_root(self) -> Path:
        """Return the configured root without accepting request-provided paths."""
        if self.dataset_root is None:
            raise TrainingDatasetNotConfiguredError(
                "training is unavailable until an operator configures a dataset root"
            )
        return self.dataset_root


__all__ = ["TrainingDatasetNotConfiguredError", "TrainingSettings"]
