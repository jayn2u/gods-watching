"""Typed fixture preparation failures."""

from typing import final, override


@final
class FixturePreparationError(RuntimeError):
    """Describe one rejected fixture input without hiding its failure class."""

    __slots__: tuple[str, str] = ("code", "detail")
    code: str
    detail: str

    def __init__(self, *, code: str, detail: str) -> None:
        """Store a stable machine code and sanitized detail."""
        super().__init__(code, detail)
        self.code = code
        self.detail = detail

    @override
    def __str__(self) -> str:
        """Render the machine code without losing its structured fields."""
        return f"{self.code}: {self.detail}"
