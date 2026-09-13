"""Typed evidence models for Task 7 runtime probes."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class EvidenceModel(BaseModel):
    """Parse immutable Task 7 evidence at each process boundary."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="ignore", frozen=True)


class DetectionEvidence(EvidenceModel):
    """Describe one class-filtered original-coordinate box."""

    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int


class ProbeEvidence(EvidenceModel):
    """Parse the real gRPC probe's binary observables."""

    source_dimensions: tuple[int, int]
    low_count: int
    low_min_confidence: float | None
    high_count: int
    empty_count: int
    malformed_error: str
    post_error_count: int
    concurrent_counts: tuple[int, int, int, int]
    coordinates_within_original: bool
    detections: tuple[DetectionEvidence, ...]


class TensorConfig(EvidenceModel):
    """Parse one Triton model tensor declaration."""

    name: str
    data_type: str
    dims: tuple[int, ...]


class DynamicBatching(EvidenceModel):
    """Parse the model's effective queue delay."""

    max_queue_delay_microseconds: int


class ModelConfig(EvidenceModel):
    """Parse the effective detector contract returned by Triton."""

    max_batch_size: int
    input: tuple[TensorConfig, ...]
    output: tuple[TensorConfig, ...]
    dynamic_batching: DynamicBatching


class BatchStat(EvidenceModel):
    """Parse one observed Triton execution batch size."""

    batch_size: int


class ModelStat(EvidenceModel):
    """Parse detector batch statistics from the running server."""

    batch_stats: tuple[BatchStat, ...]


class ModelStats(EvidenceModel):
    """Parse the running model statistics envelope."""

    model_stats: tuple[ModelStat, ...]


@dataclass(frozen=True, slots=True)
class HealthyRun:
    """Bind healthy runtime facts to their captured artifacts."""

    ready: bool
    probe: ProbeEvidence
    config: ModelConfig
    stats: ModelStats
    image_id: str
    paths: tuple[Path, ...]
