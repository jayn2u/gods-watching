"""Verification harness input and output contracts."""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from .base import ContractModel

ScenarioName = Annotated[str, Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")]


class VerificationRequest(ContractModel):
    """Select an isolated scenario and its evidence destination."""

    scenario: ScenarioName
    evidence_dir: Path


class VerificationResult(ContractModel):
    """Record a binary verification result for machine consumption."""

    scenario: ScenarioName
    outcome: Literal["passed", "failed"]
    exit_code: int
    artifact_paths: tuple[Path, ...]
