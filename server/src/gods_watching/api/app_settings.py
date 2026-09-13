"""Prepared configuration required by the HTTP application boundary."""

from dataclasses import dataclass
from typing import Final

SESSION_COOKIE_NAME: Final = "gw_session"


@dataclass(frozen=True, slots=True)
class ApiSettings:
    """Hold explicitly prepared public origin and cookie policy."""

    public_origin: str
    trusted_gateway: str | None = None
    session_cookie_name: str = SESSION_COOKIE_NAME
    secure_cookie: bool = True
    session_cookie_path: str = "/"
    activity_refresh_seconds: float = 60.0
    max_body_bytes: int = 1_048_576
