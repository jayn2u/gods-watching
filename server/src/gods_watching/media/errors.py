"""Typed media transport boundary errors."""

from dataclasses import dataclass
from typing import override


@dataclass(frozen=True, slots=True)
class InvalidGatewayLocationError(Exception):
    """Reject an upstream resource location outside the expected WHEP path."""

    location: str

    @override
    def __str__(self) -> str:
        """Return a credential-free boundary message."""
        return "Media gateway returned an invalid resource location"
