from pathlib import Path

import pytest
from pydantic import ValidationError

import gods_watching.settings as settings_package
from gods_watching.settings import AppSettings


def test_app_settings_public_import_preserves_defaults_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: no process settings overrides in the environment
    for name in (
        "GW_RUNTIME_ROOT",
        "GW_PUBLIC_HOST",
        "GW_RETENTION_DAYS",
        "GW_QUOTA_BYTES",
    ):
        monkeypatch.delenv(name, raising=False)

    # When: settings are loaded through the public package boundary
    defaults = AppSettings()

    # Then: the original defaults and validation semantics remain intact
    assert defaults.runtime_root == Path("runtime")
    assert defaults.public_host == "localhost"
    assert defaults.retention_days == 7
    assert defaults.quota_bytes == 100_000_000_000
    with pytest.raises(ValidationError):
        _ = AppSettings(retention_days=0)

    # When: supported GW_ environment variables are supplied
    monkeypatch.setenv("GW_RUNTIME_ROOT", "var/runtime")
    monkeypatch.setenv("GW_PUBLIC_HOST", "lan.example")
    monkeypatch.setenv("GW_RETENTION_DAYS", "14")
    monkeypatch.setenv("GW_QUOTA_BYTES", "250000000000")
    configured = AppSettings()

    # Then: the public import resolves the same environment-backed settings
    assert configured.runtime_root == Path("var/runtime")
    assert configured.public_host == "lan.example"
    assert configured.retention_days == 14
    assert configured.quota_bytes == 250_000_000_000


def test_settings_is_a_package_without_a_shadowing_module() -> None:
    # Given / When: the package and process settings module are imported
    package_path = Path(settings_package.__file__ or "")

    # Then: there is one unambiguous public module for AppSettings
    assert package_path.name == "__init__.py"
    assert AppSettings.__module__ == "gods_watching.settings.process"
    assert not package_path.parent.parent.joinpath("settings.py").exists()
