"""Typed camera runtime handoff for the later media and worker integration."""

from dataclasses import dataclass
from enum import StrEnum
from typing import NewType, Protocol
from uuid import UUID

from gods_watching.contracts.identifiers import CameraId, CameraSessionId

from .source_probe import ParsedRtspSource

CameraGenerationId = NewType("CameraGenerationId", UUID)


class CameraActivationReason(StrEnum):
    """Explain why a new detector generation should start."""

    CREATED = "created"
    SOURCE_EDIT = "source_edit"
    DETECTION_ENABLED = "detection_enabled"


class CameraCancellationReason(StrEnum):
    """Explain why the current detector generation should stop."""

    SOURCE_EDIT = "source_edit"
    DETECTION_DISABLED = "detection_disabled"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True)
class CameraActivationRequest:
    """Carry one version-fenced source activation to task 12c/18."""

    camera_id: CameraId
    version: int
    session_id: CameraSessionId
    generation_id: CameraGenerationId
    source: ParsedRtspSource
    detection_threshold: float
    reason: CameraActivationReason


@dataclass(frozen=True, slots=True)
class CameraCancellationRequest:
    """Carry one version-fenced cancellation to task 12c/18."""

    camera_id: CameraId
    version: int
    session_id: CameraSessionId
    generation_id: CameraGenerationId
    reason: CameraCancellationReason


@dataclass(frozen=True, slots=True)
class CameraLifecyclePlan:
    """Describe ordered cancellation then activation work without running it."""

    activation: CameraActivationRequest | None
    cancellation: CameraCancellationRequest | None


class CameraLifecyclePort(Protocol):
    """Capability supplied by the later authenticated media/worker layer."""

    async def activate(self, request: CameraActivationRequest) -> None:
        """Activate exactly the supplied camera source generation."""
        ...

    async def cancel(self, request: CameraCancellationRequest) -> None:
        """Cancel exactly the supplied camera source generation."""
        ...
