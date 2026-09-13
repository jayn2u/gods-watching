"""Typed JSON artifacts emitted by the Task 12 integration driver."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]


class _FrozenModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class HttpResult(_FrozenModel):
    """One parsed HTTP response crossing the driver boundary."""

    status: int
    headers: dict[str, str]
    body: JsonValue


class SessionResponse(_FrozenModel):
    """The session fields read by the driver."""

    authenticated: bool = False
    idle_expires_at: str | None = None


class CameraResponse(_FrozenModel):
    """The camera identity and redacted source metadata read by the driver."""

    camera_id: str
    source_port: int | None = None


class SettingsResponse(_FrozenModel):
    """The persisted settings fields read by the driver."""

    retention_days: int | None = None


class BrowserCookie(_FrozenModel):
    """Redacted cookie metadata captured by Chromium."""

    secure: bool
    http_only: bool
    same_site: str
    value_length: int


class BrowserStream(_FrozenModel):
    """Video playback metadata captured by Chromium."""

    post_status: int
    location: str | None = None
    frames_decoded: int = 0
    first_frames_decoded: int = 0
    codec: str | None = None
    connection_state: str | None = None


class BrowserArtifact(_FrozenModel):
    """The validated browser scenario artifact."""

    mode: str
    anonymous_list: int
    anonymous_whep: int
    login: int
    cookie: BrowserCookie | None = None
    camera_response: int | None = None
    source_hidden: bool | None = None
    stream: BrowserStream
    readers_during: int | None = None
    passive_idle_unchanged: bool
    logout: int
    readers_after_logout: int | None = None
    after_logout_authenticated: bool | None = None
    wrong_login_statuses: list[int] = Field(default_factory=list)
    last_retry_after: str | None = None
    cross_origin_login: int | None = None
    cross_origin_logout: int | None = None
    survives_cross_origin: bool | None = None
    malformed_source: int | None = None
    browser_errors: list[str] = Field(default_factory=list)


class DriverArtifact(_FrozenModel):
    """The validated top-level driver artifact."""

    mode: str
    checks: dict[str, bool]
    observations: JsonObject
