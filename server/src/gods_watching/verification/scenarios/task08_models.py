"""Typed schemas for Task 8 GPU evidence."""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ModalityProof(BaseModel):
    """Parse one modality's vector and reference evidence."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    shape: tuple[int, int]
    finite: bool
    norms: tuple[float, ...]
    reference_cosines: tuple[float, ...]


class ClipProof(BaseModel):
    """Parse the container-side reference comparison."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True, extra="forbid")
    cuda_device: str
    model_revision: str
    text: ModalityProof
    image: ModalityProof
    batches_match: bool
