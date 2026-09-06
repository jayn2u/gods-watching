"""Shared Pydantic contract configuration."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ContractModel(BaseModel):
    """Reject unknown input and keep parsed boundary values immutable."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)
