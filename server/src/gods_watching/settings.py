"""Typed process settings loaded from the environment."""

from pathlib import Path
from typing import ClassVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Resolve runtime paths and the configured LAN public host."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="GW_",
        extra="forbid",
        frozen=True,
    )

    runtime_root: Path = Path("runtime")
    public_host: str = "localhost"
    retention_days: int = Field(default=7, ge=1)
    quota_bytes: int = Field(default=100_000_000_000, gt=0)
