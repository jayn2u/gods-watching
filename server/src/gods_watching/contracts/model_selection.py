"""Authenticated settings contracts for the global CLIP model selector."""

from enum import StrEnum
from typing import Annotated
from uuid import UUID

from pydantic import Field

from .base import ContractModel


class ModelTransitionPhase(StrEnum):
    """Public durable transition phases."""

    QUEUED = "queued"
    PREPARING = "preparing"
    REINDEXING = "reindexing"
    ACTIVATING = "activating"
    ROLLING_BACK = "rolling_back"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelCatalogEntry(ContractModel):
    """One immutable registry package and local preparation status."""

    model_id: str
    display_name: str
    dimension: Annotated[int, Field(gt=0)]
    prepared: bool
    reason: str | None = None


class ModelTransitionResponse(ContractModel):
    """Bounded progress and safe outcome details for one switch."""

    id: UUID
    source_model_id: str
    target_model_id: str
    phase: ModelTransitionPhase
    processed: Annotated[int, Field(ge=0)]
    total: Annotated[int, Field(ge=0)]
    skipped: Annotated[int, Field(ge=0)]
    skip_reasons: dict[str, Annotated[int, Field(ge=0)]]
    error: str | None = None


class ModelSettingsResponse(ContractModel):
    """Catalog, active identity, maintenance state, and latest transition."""

    active_model_id: str
    maintenance: bool
    models: tuple[ModelCatalogEntry, ...]
    transition: ModelTransitionResponse | None


class ModelApplyRequest(ContractModel):
    """Select one prepared immutable registry package."""

    model_id: str = Field(min_length=1, max_length=255)


__all__ = [
    "ModelApplyRequest",
    "ModelCatalogEntry",
    "ModelSettingsResponse",
    "ModelTransitionPhase",
    "ModelTransitionResponse",
]
