"""Authentication boundary policies for client identity and same-origin writes."""

from __future__ import annotations

from dataclasses import dataclass
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import override
from urllib.parse import SplitResult, urlsplit

from .types import ClientIp

_IpAddress = IPv4Address | IPv6Address


@dataclass(frozen=True, slots=True)
class ClientIpError(Exception):
    """Report a malformed direct client address."""

    @override
    def __str__(self) -> str:
        """Return a stable error without echoing untrusted address text."""
        return "client address is invalid"


def _parse_ip(value: str) -> _IpAddress | None:
    try:
        return ip_address(value.strip())
    except ValueError:
        return None


def canonical_client_ip(
    peer_ip: str,
    forwarded_for: str | None,
    trusted_gateway: str | None,
) -> ClientIp:
    """Use forwarded identity only when the direct peer is the configured gateway."""
    peer = _parse_ip(peer_ip)
    if peer is None:
        raise ClientIpError
    if trusted_gateway is None or forwarded_for is None:
        return ClientIp(peer.compressed)
    gateway = _parse_ip(trusted_gateway)
    if gateway is None or peer != gateway:
        return ClientIp(peer.compressed)
    forwarded = next((entry.strip() for entry in forwarded_for.split(",") if entry.strip()), "")
    forwarded_ip = _parse_ip(forwarded)
    if forwarded_ip is None:
        return ClientIp(peer.compressed)
    return ClientIp(forwarded_ip.compressed)


def _normalized_origin(value: str) -> tuple[str, str, int | None] | None:
    parsed: SplitResult = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.path not in {"", "/"}:
        return None
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if hostname is None:
        return None
    default_port = 443 if parsed.scheme == "https" else 80
    return parsed.scheme, hostname.lower(), port or default_port


def is_same_origin(origin: str | None, expected_origin: str) -> bool:
    """Return whether a mutation Origin exactly matches the configured origin."""
    if origin is None:
        return False
    actual = _normalized_origin(origin)
    expected = _normalized_origin(expected_origin)
    return actual is not None and actual == expected


@dataclass(frozen=True, slots=True)
class MutationOriginError(Exception):
    """Report a cross-origin or missing Origin mutation header."""

    @override
    def __str__(self) -> str:
        """Return a stable CSRF failure without echoing request headers."""
        return "mutation origin is not allowed"


def require_same_origin(origin: str | None, expected_origin: str) -> None:
    """Raise when a state-changing request does not satisfy same-origin policy."""
    if not is_same_origin(origin, expected_origin):
        raise MutationOriginError
