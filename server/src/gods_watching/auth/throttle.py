"""Canonical client-IP login failure throttle."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Final

_FAILURE_WINDOW: Final = timedelta(seconds=60)
_LOCKOUT_DURATION: Final = timedelta(seconds=300)
_MAX_FAILURES: Final = 5


@dataclass(slots=True)  # noqa: RUF100  # noqa: MUTABLE_OK
class IpLoginThrottle:
    """Keep mutable bounded failure state because each request updates lockout data."""

    _failures: dict[str, deque[datetime]] = field(default_factory=dict)
    _blocked_until: dict[str, datetime] = field(default_factory=dict)

    def _prune(self, client_ip: str, now: datetime) -> deque[datetime]:
        failures = self._failures.setdefault(client_ip, deque())
        threshold = now - _FAILURE_WINDOW
        while failures and failures[0] <= threshold:
            _ = failures.popleft()
        blocked_until = self._blocked_until.get(client_ip)
        if blocked_until is not None and blocked_until <= now:
            del self._blocked_until[client_ip]
        return failures

    def is_blocked(self, client_ip: str, now: datetime) -> bool:
        """Return whether this address is currently in the 300-second lockout."""
        _ = self._prune(client_ip, now)
        blocked_until = self._blocked_until.get(client_ip)
        return blocked_until is not None and blocked_until > now

    def retry_after(self, client_ip: str, now: datetime) -> int:
        """Return whole seconds remaining in a lockout, or zero when clear."""
        _ = self._prune(client_ip, now)
        blocked_until = self._blocked_until.get(client_ip)
        if blocked_until is None or blocked_until <= now:
            return 0
        return max(1, ceil((blocked_until - now).total_seconds()))

    def record_failure(self, client_ip: str, now: datetime) -> bool:
        """Record one failed attempt and return whether the address is blocked."""
        failures = self._prune(client_ip, now)
        if self.is_blocked(client_ip, now):
            return True
        if len(failures) >= _MAX_FAILURES:
            self._blocked_until[client_ip] = now + _LOCKOUT_DURATION
            return True
        failures.append(now)
        return False

    def record_success(self, client_ip: str) -> None:
        """Clear failure history after a successful authenticated login."""
        _ = self._failures.pop(client_ip, None)
        _ = self._blocked_until.pop(client_ip, None)
