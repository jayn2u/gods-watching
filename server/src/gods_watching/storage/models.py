"""Transactional PostgreSQL schema mappings."""

from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .vector import Vector


class Base(DeclarativeBase):
    """Own metadata for application relations."""


class Camera(Base):
    """Persist a configured RTSP source with encrypted source material."""

    __tablename__: str = "cameras"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("char_length(name) BETWEEN 1 AND 80", name="ck_cameras_name_length"),
        CheckConstraint(
            "detection_threshold BETWEEN 0.1 AND 0.95",
            name="ck_cameras_detection_threshold",
        ),
        CheckConstraint("version >= 1", name="ck_cameras_version"),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(80), unique=True)
    source_ciphertext: Mapped[bytes] = mapped_column(LargeBinary)
    source_host: Mapped[str] = mapped_column(String(255))
    source_port: Mapped[int | None] = mapped_column(Integer)
    detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    detection_threshold: Mapped[float] = mapped_column(Float, default=0.5)
    version: Mapped[int] = mapped_column(Integer, default=1)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class CameraSession(Base):
    """Separate reconnect and source-generation track namespaces."""

    __tablename__: str = "camera_sessions"

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[UUID] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    generation_id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), default=uuid4)
    cause: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class WorkerRuntimeStatus(Base):
    """Persist the worker heartbeat and process-wide readiness snapshot."""

    __tablename__: str = "worker_runtime_status"

    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    inference_ready: Mapped[bool] = mapped_column(Boolean)
    persistence_paused: Mapped[bool] = mapped_column(Boolean)
    storage_managed_bytes: Mapped[int] = mapped_column(BigInteger)
    storage_quota_bytes: Mapped[int] = mapped_column(BigInteger)
    indexing_queue_depth: Mapped[int] = mapped_column(Integer)
    last_searchable_latency_seconds: Mapped[float | None] = mapped_column(Float)


class CameraRuntimeStatus(Base):
    """Persist the latest bounded ingest snapshot for one configured camera."""

    __tablename__: str = "camera_runtime_status"

    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True
    )
    camera_session_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("camera_sessions.id", ondelete="CASCADE")
    )
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_ingest_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actual_framerate: Mapped[float] = mapped_column(Float)
    detector_framerate: Mapped[float] = mapped_column(Float)
    dropped_frames: Mapped[int] = mapped_column(BigInteger)
    detector_requests: Mapped[int] = mapped_column(BigInteger)
    detector_results: Mapped[int] = mapped_column(BigInteger)
    last_error: Mapped[str | None] = mapped_column(String(200))


class CameraDetectionLatest(Base):
    """Retain one replaceable source-frame detector result per camera."""

    __tablename__: str = "camera_detection_latest"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("width > 0", name="ck_camera_detection_latest_width"),
        CheckConstraint("height > 0", name="ck_camera_detection_latest_height"),
    )

    camera_id: Mapped[UUID] = mapped_column(
        ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True
    )
    camera_session_id: Mapped[UUID] = mapped_column(
        ForeignKey("camera_sessions.id", ondelete="CASCADE")
    )
    db_generation_id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True))
    source_generation_id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True))
    frame_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    boxes: Mapped[list[dict[str, float]]] = mapped_column(postgresql.JSONB)


class Appearance(Base):
    """Store one query-visible representative per camera session track."""

    __tablename__: str = "appearances"
    __table_args__: tuple[CheckConstraint | UniqueConstraint | Index, ...] = (
        UniqueConstraint("camera_id", "session_id", "track_id", name="uq_appearance_track"),
        CheckConstraint("track_id >= 0", name="ck_appearances_track_id"),
        CheckConstraint("last_seen >= first_seen", name="ck_appearances_seen_order"),
        CheckConstraint(
            "ended_at IS NULL OR ended_at >= last_seen", name="ck_appearances_end_order"
        ),
        CheckConstraint("representative_version >= 1", name="ck_appearances_version"),
        CheckConstraint("x_min >= 0 AND y_min >= 0", name="ck_appearances_bbox_origin"),
        CheckConstraint("x_max > x_min AND y_max > y_min", name="ck_appearances_bbox_extent"),
        CheckConstraint(
            "x_max <= source_width AND y_max <= source_height",
            name="ck_appearances_bbox_bounds",
        ),
        CheckConstraint(
            "detector_confidence BETWEEN 0 AND 1",
            name="ck_appearances_confidence",
        ),
        CheckConstraint("crop_quality >= 0", name="ck_appearances_crop_quality"),
        CheckConstraint("byte_size > 0", name="ck_appearances_byte_size"),
        CheckConstraint(
            "embedding IS NULL OR embedding_dimension = vector_dims(embedding)",
            name="ck_appearances_embedding_dimension",
        ),
        Index("ix_appearances_camera_time", "camera_id", "first_seen", "last_seen"),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True)
    camera_id: Mapped[UUID] = mapped_column(ForeignKey("cameras.id", ondelete="RESTRICT"))
    session_id: Mapped[UUID] = mapped_column(ForeignKey("camera_sessions.id", ondelete="RESTRICT"))
    track_id: Mapped[int] = mapped_column(BigInteger)
    first_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ended_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    representative_version: Mapped[int] = mapped_column(Integer)
    crop_object_key: Mapped[str] = mapped_column(String(80), unique=True)
    x_min: Mapped[int] = mapped_column(Integer)
    y_min: Mapped[int] = mapped_column(Integer)
    x_max: Mapped[int] = mapped_column(Integer)
    y_max: Mapped[int] = mapped_column(Integer)
    source_width: Mapped[int] = mapped_column(Integer)
    source_height: Mapped[int] = mapped_column(Integer)
    detector_confidence: Mapped[float] = mapped_column(Float)
    crop_quality: Mapped[float] = mapped_column(Float)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    embedded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    model_id: Mapped[str] = mapped_column(String(255))
    model_revision: Mapped[str] = mapped_column(String(255))
    # Unconstrained at the storage boundary: 512 and 768 dimensional model
    # spaces coexist while a transition stages its replacement vectors.
    embedding_dimension: Mapped[int] = mapped_column(
        Integer,
        server_default="512",
    )
    embedding: Mapped[str | None] = mapped_column(Vector())
    tombstoned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class CropGarbage(Base):
    """Record retryable crop deletion work transactionally."""

    __tablename__: str = "crop_gc"
    __table_args__: tuple[CheckConstraint | Index, ...] = (
        CheckConstraint("attempts >= 0", name="ck_crop_gc_attempts"),
        Index("ix_crop_gc_appearance_order", "appearance_id", "enqueued_at", "id"),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    appearance_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("appearances.id", ondelete="RESTRICT")
    )
    object_key: Mapped[str] = mapped_column(String(80), unique=True)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    enqueued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)


class LoginSession(Base):
    """Persist hashed opaque login tokens and expiry state only."""

    __tablename__: str = "sessions"

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_activity_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    idle_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    absolute_expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ApplicationSettings(Base):
    """Persist the single operator retention, quota, and wall settings row."""

    __tablename__: str = "settings"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("singleton", name="ck_settings_singleton"),
        CheckConstraint("retention_days >= 1", name="ck_settings_retention_days"),
        CheckConstraint("quota_bytes > 0", name="ck_settings_quota_bytes"),
    )

    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    retention_days: Mapped[int] = mapped_column(Integer, default=7)
    quota_bytes: Mapped[int] = mapped_column(BigInteger, default=100_000_000_000)
    wall_slot_ids: Mapped[list[UUID]] = mapped_column(
        ARRAY(postgresql.UUID(as_uuid=True)), default=list
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ActiveModelIdentity(Base):
    """The one model identity whose vectors are currently searchable."""

    __tablename__: str = "active_model_identity"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("singleton", name="ck_active_model_identity_singleton"),
        CheckConstraint("embedding_dimension > 0", name="ck_active_model_identity_dimension"),
    )

    singleton: Mapped[bool] = mapped_column(Boolean, primary_key=True, default=True)
    model_id: Mapped[str] = mapped_column(String(255))
    model_revision: Mapped[str] = mapped_column(String(255))
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class ModelTransitionJob(Base):
    """Durable state and bounded counters for one global model switch."""

    __tablename__: str = "model_transition_jobs"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint("total >= 0", name="ck_model_transition_total"),
        CheckConstraint(
            "processed >= 0 AND processed <= total",
            name="ck_model_transition_processed",
        ),
        CheckConstraint(
            "skipped >= 0 AND skipped <= processed",
            name="ck_model_transition_skipped",
        ),
        CheckConstraint("source_dimension > 0", name="ck_model_transition_source_dimension"),
        CheckConstraint("target_dimension > 0", name="ck_model_transition_target_dimension"),
    )

    id: Mapped[UUID] = mapped_column(postgresql.UUID(as_uuid=True), primary_key=True, default=uuid4)
    source_model_id: Mapped[str] = mapped_column(String(255))
    source_model_revision: Mapped[str] = mapped_column(String(255))
    source_dimension: Mapped[int] = mapped_column(Integer)
    target_model_id: Mapped[str] = mapped_column(String(255))
    target_model_revision: Mapped[str] = mapped_column(String(255))
    target_dimension: Mapped[int] = mapped_column(Integer)
    phase: Mapped[str] = mapped_column(String(32))
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    skip_reasons: Mapped[dict[str, int]] = mapped_column(
        postgresql.JSONB, default=dict, server_default="{}"
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ModelTransitionStage(Base):
    """One staged crop vector keyed by transition and appearance identity."""

    __tablename__: str = "model_transition_stages"
    __table_args__: tuple[CheckConstraint, ...] = (
        CheckConstraint(
            "processed_at IS NULL OR embedding IS NOT NULL OR skip_reason IS NOT NULL",
            name="ck_model_transition_stage_result",
        ),
        CheckConstraint(
            "embedding_dimension > 0",
            name="ck_model_transition_stage_dimension",
        ),
        CheckConstraint(
            "embedding IS NULL OR embedding_dimension = vector_dims(embedding)",
            name="ck_model_transition_stage_embedding_dimension",
        ),
        CheckConstraint(
            "source_representative_version >= 1",
            name="ck_model_transition_stage_version",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("model_transition_jobs.id", ondelete="CASCADE"), primary_key=True
    )
    appearance_id: Mapped[UUID] = mapped_column(
        ForeignKey("appearances.id", ondelete="CASCADE"), primary_key=True
    )
    source_crop_object_key: Mapped[str] = mapped_column(String(80))
    source_representative_version: Mapped[int] = mapped_column(Integer)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[str | None] = mapped_column(Vector())
    skip_reason: Mapped[str | None] = mapped_column(String(64))
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
