"""Immutable operator settings for a CUHK-PEDES CLIP training run."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from .base import ContractModel

EpochCount = Annotated[int, Field(strict=True, ge=1, le=100)]
LearningRate = Annotated[float, Field(ge=1e-7, le=1e-3, allow_inf_nan=False)]
MicroBatchSize = Annotated[int, Field(strict=True, ge=2, le=128)]
WeightDecay = Annotated[float, Field(ge=0, le=0.2, allow_inf_nan=False)]
GradientAccumulation = Annotated[int, Field(strict=True, ge=1, le=32)]
WarmupRatio = Annotated[float, Field(ge=0, le=0.3, allow_inf_nan=False)]
RandomSeed = Annotated[int, Field(strict=True, ge=0, le=2_147_483_647)]
EarlyStoppingPatience = Annotated[int, Field(strict=True, ge=1, le=20)]
GradientClippingNorm = Annotated[float, Field(ge=0.1, le=10, allow_inf_nan=False)]


class TrainingConfig(ContractModel):
    """Validate and freeze every user-controlled training hyperparameter."""

    epochs: EpochCount = 30
    learning_rate: LearningRate = 1e-5
    micro_batch_size: MicroBatchSize = 16
    weight_decay: WeightDecay = 0.01
    gradient_accumulation: GradientAccumulation = 4
    warmup_ratio: WarmupRatio = 0.05
    seed: RandomSeed = 42
    early_stopping_patience: EarlyStoppingPatience | None = 5
    gradient_clipping_norm: GradientClippingNorm = 1.0
    mixed_precision: Literal["fp16", "fp32"] = "fp16"
    gradient_checkpointing: bool = True


NonNegativeCount = Annotated[int, Field(strict=True, ge=0)]
RecallScore = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
BoundedTrainingError = Annotated[str, Field(max_length=1000)]


class TrainingSplitCounts(ContractModel):
    """Aggregate counts for one original CUHK-PEDES split."""

    images: NonNegativeCount
    captions: NonNegativeCount
    identities: NonNegativeCount


class TrainingDatasetSnapshot(ContractModel):
    """Safe dataset identity and counts persisted with a job."""

    dataset_id: Literal["cuhk-pedes"]
    fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    protocol: str = Field(min_length=1, max_length=96)
    split_counts: dict[Literal["train", "val", "test"], TrainingSplitCounts]
    image_count: NonNegativeCount
    caption_count: NonNegativeCount
    identity_count: NonNegativeCount


class TrainingDatasetStatus(ContractModel):
    """Describe whether the operator-configured dataset is usable."""

    registered: bool
    valid: bool
    reason: str | None = None
    snapshot: TrainingDatasetSnapshot | None = None


class TrainingPreflightRequest(ContractModel):
    """Evaluate one immutable config against the latest supervisor GPU snapshot."""

    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingPreflightResponse(ContractModel):
    """Expose a current exact-profile memory estimate and admission decision."""

    admitted: bool
    training_peak_bytes: NonNegativeCount
    reserve_bytes: NonNegativeCount
    required_bytes: NonNegativeCount
    free_bytes: NonNegativeCount
    profile_identity: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    observed_at: datetime
    reason: Literal["admitted", "insufficient_free_memory"]


class TrainingJobSubmitRequest(ContractModel):
    """Submit a job using the sole registered dataset and a caller request identity."""

    request_id: UUID
    dataset_id: Literal["cuhk-pedes"] = "cuhk-pedes"
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingActionRequest(ContractModel):
    """Identify a cancel or resume request for one durable job."""

    request_id: UUID


class TrainingRetrievalScores(ContractModel):
    """Macro text-to-image Recall@K on the original held-out test split."""

    recall_at_1: RecallScore
    recall_at_5: RecallScore
    recall_at_10: RecallScore


class TrainingEvaluationSummary(ContractModel):
    """Safe baseline/candidate test metrics and immutable provenance for job detail."""

    dataset_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    dataset_split: Literal["test"]
    protocol: str = Field(min_length=1, max_length=96)
    baseline_model_id: str = Field(min_length=1, max_length=128)
    baseline_revision: str = Field(min_length=1, max_length=128)
    baseline_package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    training_source_fingerprint: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    evaluation_code_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    metric_definition: str = Field(min_length=1, max_length=128)
    best_validation_epoch: Annotated[int, Field(strict=True, ge=1)]
    baseline: TrainingRetrievalScores
    candidate: TrainingRetrievalScores
    candidate_weights_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    package_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class TrainingJobResponse(ContractModel):
    """Safe durable job view; no run paths, captions, or process credentials."""

    id: UUID
    request_id: UUID
    phase: Literal[
        "starting",
        "training",
        "evaluating",
        "publishing",
        "succeeded",
        "cancelling",
        "cancelled",
        "failed",
        "interrupted",
    ]
    config: TrainingConfig
    dataset: TrainingDatasetSnapshot
    current_epoch: NonNegativeCount
    current_step: NonNegativeCount
    owner_generation: NonNegativeCount
    cancel_requested: bool
    attempts: NonNegativeCount
    best_metric: RecallScore | None = None
    candidate_model_id: str | None = None
    candidate_revision: str | None = None
    evaluation: TrainingEvaluationSummary | None = None
    error: BoundedTrainingError | None = None
    created_at: datetime
    updated_at: datetime
    finished_at: datetime | None = None


class TrainingJobPage(ContractModel):
    """One bounded cursor page of durable job history."""

    items: tuple[TrainingJobResponse, ...]
    next_cursor: str | None = None


class TrainingMetric(ContractModel):
    """One durable epoch/step metric without raw dataset text."""

    epoch: NonNegativeCount
    step: NonNegativeCount
    training_loss: float | None = None
    validation_recall_at_1: RecallScore | None = None
    allocated_bytes: NonNegativeCount | None = None
    reserved_bytes: NonNegativeCount | None = None
    observed_at: datetime


class TrainingMetricPage(ContractModel):
    """One bounded cursor page of job metrics."""

    items: tuple[TrainingMetric, ...]
    next_cursor: str | None = None


class TrainingLogEntry(ContractModel):
    """One bounded safe training log line."""

    cursor: str = Field(min_length=1, max_length=160)
    level: Literal["debug", "info", "warning", "error"]
    message: str = Field(max_length=500)
    observed_at: datetime


class TrainingLogPage(ContractModel):
    """One bounded cursor page of safe job logs."""

    items: tuple[TrainingLogEntry, ...]
    next_cursor: str | None = None


__all__ = [
    "TrainingActionRequest",
    "TrainingConfig",
    "TrainingDatasetSnapshot",
    "TrainingDatasetStatus",
    "TrainingEvaluationSummary",
    "TrainingJobPage",
    "TrainingJobResponse",
    "TrainingJobSubmitRequest",
    "TrainingLogEntry",
    "TrainingLogPage",
    "TrainingMetric",
    "TrainingMetricPage",
    "TrainingPreflightRequest",
    "TrainingPreflightResponse",
    "TrainingRetrievalScores",
    "TrainingSplitCounts",
]
