from pathlib import Path

import pytest
from pydantic import ValidationError

from gods_watching.pipeline_worker.settings import PipelineWorkerSettings, locked_clip_model

_REQUIRED = {
    "GW_DATABASE_URL": "postgresql+asyncpg://gw:secret@db:5432/gods_watching",
    "GW_TRITON_GRPC_URL": "triton:8001",
    "GW_CROPS_ROOT": "/var/lib/gods-watching/crops",
    "GW_CAMERA_CIPHER_KEY": "fernet-key",
}


def test_worker_settings_load_required_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: the worker process environment
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GW_WORKER_POLL_SECONDS", raising=False)

    # When: settings are loaded
    settings = PipelineWorkerSettings()  # pyright: ignore[reportCallIssue]

    # Then: every connection target resolves and polling defaults to one second
    assert settings.database_url == _REQUIRED["GW_DATABASE_URL"]
    assert settings.triton_grpc_url == "triton:8001"
    assert settings.crops_root == Path("/var/lib/gods-watching/crops")
    assert settings.camera_cipher_key == "fernet-key"
    assert settings.worker_poll_seconds == 1.0


def test_worker_settings_reject_missing_database(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given: a worker environment without a database URL
    for name, value in _REQUIRED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("GW_DATABASE_URL")

    # When / Then: startup fails before any service is constructed
    with pytest.raises(ValidationError):
        _ = PipelineWorkerSettings()  # pyright: ignore[reportCallIssue]


def test_locked_clip_model_comes_from_the_committed_lock() -> None:
    # Given: the repository's committed model lock
    lock_path = Path(__file__).resolve().parents[2] / "assets/models.lock.json"

    # When: the worker resolves the CLIP identity it stamps on appearances
    model_id, revision = locked_clip_model(lock_path)

    # Then: it matches the pinned snapshot Triton serves
    assert model_id == "openai/clip-vit-base-patch16"
    assert revision == "57c216476eefef5ab752ec549e440a49ae4ae5f3"
