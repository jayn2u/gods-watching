"""Typed Task 11 scenario and driver failure."""

from typing import override


class Task11ExecutionError(RuntimeError):
    """Report one failed publication prerequisite or observable."""

    __slots__: tuple[str, ...] = ("detail",)

    detail: str

    def __init__(self, detail: str) -> None:
        """Retain the machine-readable failure detail."""
        super().__init__(detail)
        self.detail = detail

    @override
    def __str__(self) -> str:
        """Return the stable failure detail."""
        return self.detail


__all__ = ["Task11ExecutionError"]
