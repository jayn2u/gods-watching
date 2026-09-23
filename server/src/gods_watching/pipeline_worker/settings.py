"""Environment settings and pinned model identity for the pipeline worker process."""

from pathlib import Path
from typing import ClassVar, Final

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from gods_watching.setup.models import load_models_lock

_CLIP_MODEL_PREFIX: Final = "openai/clip-"


class PipelineWorkerSettings(BaseSettings):
    """Resolve database, inference, crop storage, and credential targets for the worker."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="GW_",
        extra="ignore",
        frozen=True,
    )

    database_url: str = Field(repr=False)
    triton_grpc_url: str
    crops_root: Path
    camera_cipher_key: str = Field(repr=False)
    worker_poll_seconds: float = Field(default=1.0, gt=0.0)
    model_lock_path: Path = Path("/opt/gods-watching/assets/models.lock.json")
    model_assets_root: Path = Path("/models")


class LockedClipModelError(RuntimeError):
    """Report a committed model lock that does not pin a CLIP snapshot."""


def locked_clip_model(lock_path: Path) -> tuple[str, str]:
    """Return the CLIP model id and revision pinned by the committed model lock."""
    lock = load_models_lock(lock_path)
    for model in lock.models:
        if model.model_id.startswith(_CLIP_MODEL_PREFIX):
            return model.model_id, model.revision
    detail = f"model lock does not pin a CLIP model: {lock_path}"
    raise LockedClipModelError(detail)


__all__ = ["LockedClipModelError", "PipelineWorkerSettings", "locked_clip_model"]
