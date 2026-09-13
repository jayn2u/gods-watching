"""Typed failures raised while driving Task 14 search scenarios."""

from typing import final, override


@final
class Task14ExecutionError(RuntimeError):
    """Report a Task 14 runtime step that could not produce evidence."""

    def __init__(self, *, detail: str) -> None:
        """Retain the operator-safe failure detail."""
        self.detail = detail
        super().__init__(detail)

    @override
    def __str__(self) -> str:
        return self.detail


__all__ = ["Task14ExecutionError"]
