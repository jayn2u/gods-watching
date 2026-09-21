from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from gods_watching.api.production import ProductionSettings

_COMMAND = TypeAdapter(list[str])


def test_production_uvicorn_preserves_gateway_peer_for_auth_policy() -> None:
    # Given: the packaged application image command
    dockerfile = Path(__file__).parents[2] / "deploy" / "Dockerfile.app"
    command_line = next(
        line
        for line in dockerfile.read_text(encoding="utf-8").splitlines()
        if line.startswith("CMD ")
    )

    # When: the executable command is parsed
    command = _COMMAND.validate_json(command_line.removeprefix("CMD "))

    # Then: Uvicorn leaves forwarding identity to the configured gateway policy
    assert "--no-proxy-headers" in command
    assert "--proxy-headers" not in command


def test_production_settings_parse_complete_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    web_root = tmp_path / "web"
    web_root.mkdir()
    values = {
        "GW_CAMERA_CIPHER_KEY": "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=",
        "GW_CROPS_ROOT": str(tmp_path / "crops"),
        "GW_DATABASE_URL": "postgresql+asyncpg://gods_watching:secret@postgres/gods_watching",
        "GW_MEDIA_CONTROL_HOST": "media-gateway",
        "GW_MEDIA_CONTROL_PASSWORD": "control-secret",
        "GW_MEDIA_CONTROL_USER": "control",
        "GW_MEDIA_WHEP_HOST": "media-gateway",
        "GW_MEDIA_READER_PASSWORD": "reader-secret",
        "GW_MEDIA_READER_USER": "reader",
        "GW_OPERATOR_PASSWORD": "correct horse battery staple",
        "GW_PUBLIC_ORIGIN": "http://localhost:8080",
        "GW_TRITON_GRPC_URL": "triton:8001",
        "GW_WEB_ROOT": str(web_root),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    settings = ProductionSettings()

    assert settings.media_control_host == "media-gateway"
    assert settings.media_control_port == 9997
    assert settings.media_whep_port == 8889
    assert settings.web_root == web_root


def test_production_settings_reject_missing_required_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required = (
        "GW_CAMERA_CIPHER_KEY",
        "GW_CROPS_ROOT",
        "GW_DATABASE_URL",
        "GW_MEDIA_CONTROL_HOST",
        "GW_MEDIA_CONTROL_PASSWORD",
        "GW_MEDIA_CONTROL_USER",
        "GW_MEDIA_READER_PASSWORD",
        "GW_MEDIA_READER_USER",
        "GW_OPERATOR_PASSWORD",
        "GW_PUBLIC_ORIGIN",
        "GW_TRITON_GRPC_URL",
        "GW_WEB_ROOT",
    )
    for name in required:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError):
        _ = ProductionSettings()
