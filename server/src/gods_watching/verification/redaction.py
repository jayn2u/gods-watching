"""Redact credentials before text crosses into evidence."""

import re
from typing import Final

_URL_USERINFO: Final = re.compile(
    r"(?P<scheme>[a-z][a-z0-9+.-]*://)[^/@\s]+@", re.IGNORECASE
)
_NAMED_SECRET: Final = re.compile(
    r"(?i)\b(password|passwd|token|secret|authorization)(\s*[:=]\s*)([^\s,;]+)"
)


def redact(text: str) -> str:
    """Remove URL userinfo and common named secret values."""
    without_userinfo = _URL_USERINFO.sub(r"\g<scheme>[REDACTED]@", text)
    return _NAMED_SECRET.sub(r"\1\2[REDACTED]", without_userinfo)
