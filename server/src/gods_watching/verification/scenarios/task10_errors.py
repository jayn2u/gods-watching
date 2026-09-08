"""Typed errors shared by the installed Task 10 scenario and driver."""

from typing import override


class Task10ExecutionError(RuntimeError):
    """Describe a failed Task 10 prerequisite or driver boundary."""

    __slots__: tuple[str, ...] = ("detail",)

    detail: str

    def __init__(self, detail: str) -> None:
        """Initialize the error with its machine-readable detail."""
        super().__init__(detail)
        self.detail = detail

    @override
    def __str__(self) -> str:
        return self.detail
