"""Durable training jobs and their singleton GPU execution slot."""

from datetime import datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from gods_watching.storage.models import Base


class TrainingPhase(StrEnum):
    """Durable phases of one training job."""

    STARTING = "starting"
    TRAINING = "training"
    EVALUATING = "evaluating"
    PUBLISHING = "publishing"
    SUCCEEDED = "succeeded"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"

    @classmethod
    def active(cls) -> tuple["TrainingPhase", ...]:
        """Return phases that retain the single GPU execution slot."""
        return (
            cls.STARTING,
            cls.TRAINING,
            cls.EVALUATING,
            cls.PUBLISHING,
            cls.CANCELLING,
        )

    @property
    def terminal(self) -> bool:
        """Whether this phase releases the GPU execution slot."""
        return self in {self.SUCCEEDED, self.CANCELLED, self.FAILED, self.INTERRUPTED}


class TrainingRequestKind(StrEnum):
    """Kinds of bounded API-to-supervisor handshakes."""

    PREFLIGHT = "preflight"
    SUBMIT = "submit"
    RESUME = "resume"


class TrainingRequestPhase(StrEnum):
    """Durable request outcomes; refusal never creates a training job."""

    PENDING = "pending"
    ACCEPTED = "accepted"
    REFUSED = "refused"
    EXPIRED = "expired"
    FAILED = "failed"


class TrainingJob(Base):
    """Persist immutable inputs and bounded progress for a single training run."""

    __tablename__: str = "training_jobs"
    __table_args__: tuple[CheckConstraint | UniqueConstraint | Index, ...] = (
        CheckConstraint(
            (
                "phase IN ('starting', 'training', 'evaluating', 'publishing', 'succeeded', "
                "'cancelling', 'cancelled', 'failed', 'interrupted')"
            ),
            name="ck_training_jobs_phase",
        ),
        CheckConstraint("current_epoch >= 0", name="ck_training_jobs_current_epoch"),
        CheckConstraint("current_step >= 0", name="ck_training_jobs_current_step"),
        CheckConstraint("owner_generation >= 0", name="ck_training_jobs_owner_generation"),
        CheckConstraint("attempts >= 0", name="ck_training_jobs_attempts"),
        CheckConstraint(
            "best_metric IS NULL OR best_metric BETWEEN 0 AND 1",
            name="ck_training_jobs_best_metric",
        ),
        CheckConstraint(
            (
                "(child_pid IS NULL AND child_start_time IS NULL) OR "
                "(child_pid IS NOT NULL AND child_start_time IS NOT NULL "
                "AND child_pid > 0 AND child_start_time >= 0)"
            ),
            name="ck_training_jobs_child_identity",
        ),
        CheckConstraint(
            "(candidate_model_id IS NULL) = (candidate_revision IS NULL)",
            name="ck_training_jobs_candidate_identity",
        ),
        CheckConstraint(
            "error IS NULL OR char_length(error) <= 1000",
            name="ck_training_jobs_error_length",
        ),
        CheckConstraint(
            "dataset_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_training_jobs_dataset_fingerprint",
        ),
        CheckConstraint(
            "source_fingerprint ~ '^[0-9a-f]{64}$'",
            name="ck_training_jobs_source_fingerprint",
        ),
        UniqueConstraint("request_id", name="uq_training_jobs_request_id"),
        Index(
            "uq_training_jobs_one_active",
            text("(true)"),
            unique=True,
            postgresql_where=text(
                # Keep this constraint aligned with TrainingPhase.active().
                "phase IN ('starting', 'training', 'evaluating', 'publishing', 'cancelling')"
            ),
        ),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), nullable=False)
    config_snapshot: Mapped[dict[str, object]] = mapped_column(postgresql.JSONB, nullable=False)
    dataset_snapshot: Mapped[dict[str, object]] = mapped_column(postgresql.JSONB, nullable=False)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(
        String(32), nullable=False, default=TrainingPhase.STARTING.value
    )
    current_epoch: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_step: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    owner_generation: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    child_pid: Mapped[int | None] = mapped_column(BigInteger)
    child_start_time: Mapped[int | None] = mapped_column(BigInteger)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_metric: Mapped[float | None] = mapped_column(Float)
    checkpoint_path: Mapped[str | None] = mapped_column(Text)
    candidate_model_id: Mapped[str | None] = mapped_column(String(255))
    candidate_revision: Mapped[str | None] = mapped_column(String(255))
    error: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class TrainingExecutionSlot(Base):
    """Serialize admission to the one GPU training process."""

    __tablename__: str = "training_execution_slots"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("singleton", name="ck_training_execution_slots_singleton"),
    )

    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    active_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="RESTRICT"), unique=True
    )


class TrainingRequest(Base):
    """Persist one expiring API/supervisor request and its durable response."""

    __tablename__: str = "training_requests"
    __table_args__: tuple[CheckConstraint | UniqueConstraint | Index, ...] = (
        CheckConstraint(
            "kind IN ('preflight', 'submit', 'resume')",
            name="ck_training_requests_kind",
        ),
        CheckConstraint(
            "phase IN ('pending', 'accepted', 'refused', 'expired', 'failed')",
            name="ck_training_requests_phase",
        ),
        CheckConstraint("lease_generation >= 0", name="ck_training_requests_lease_generation"),
        CheckConstraint(
            "lease_owner IS NULL OR char_length(lease_owner) BETWEEN 1 AND 128",
            name="ck_training_requests_lease_owner",
        ),
        CheckConstraint(
            "error IS NULL OR char_length(error) <= 1000",
            name="ck_training_requests_error_length",
        ),
        CheckConstraint("expires_at > created_at", name="ck_training_requests_expiry"),
        UniqueConstraint("request_id", name="uq_training_requests_request_id"),
        Index(
            "ix_training_requests_pending_order",
            "phase",
            "expires_at",
            "created_at",
            postgresql_where=text("phase = 'pending'"),
        ),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    request_id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    parent_job_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("training_jobs.id", ondelete="RESTRICT")
    )
    job_id: Mapped[UUID | None] = mapped_column(ForeignKey("training_jobs.id", ondelete="RESTRICT"))
    config_snapshot: Mapped[dict[str, object]] = mapped_column(postgresql.JSONB, nullable=False)
    dataset_snapshot: Mapped[dict[str, object] | None] = mapped_column(postgresql.JSONB)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64))
    source_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    phase: Mapped[str] = mapped_column(
        String(16), nullable=False, default=TrainingRequestPhase.PENDING.value
    )
    response_snapshot: Mapped[dict[str, object] | None] = mapped_column(postgresql.JSONB)
    lease_generation: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    error: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


__all__ = [
    "TrainingExecutionSlot",
    "TrainingJob",
    "TrainingPhase",
    "TrainingRequest",
    "TrainingRequestKind",
    "TrainingRequestPhase",
]
