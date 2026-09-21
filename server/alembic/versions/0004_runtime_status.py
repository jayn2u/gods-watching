"""Add cross-process worker and camera runtime snapshots."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004_runtime_status"
down_revision = "0003_crop_gc_appearance_owner"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create worker heartbeat and per-camera status relations."""
    op.create_table(
        "worker_runtime_status",
        sa.Column("singleton", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("inference_ready", sa.Boolean(), nullable=False),
        sa.Column("persistence_paused", sa.Boolean(), nullable=False),
        sa.Column("storage_managed_bytes", sa.BigInteger(), nullable=False),
        sa.Column("storage_quota_bytes", sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint("singleton"),
        sa.CheckConstraint("singleton", name="ck_worker_runtime_singleton"),
    )
    op.create_table(
        "camera_runtime_status",
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_ingest_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_framerate", sa.Float(), nullable=False),
        sa.Column("dropped_frames", sa.BigInteger(), nullable=False),
        sa.Column("detector_requests", sa.BigInteger(), nullable=False),
        sa.Column("detector_results", sa.BigInteger(), nullable=False),
        sa.Column("last_error", sa.String(length=200), nullable=True),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("camera_id"),
    )


def downgrade() -> None:
    """Remove worker heartbeat and per-camera status relations."""
    op.drop_table("camera_runtime_status")
    op.drop_table("worker_runtime_status")
