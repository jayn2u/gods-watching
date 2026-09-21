"""Add indexing backlog and searchable-latency status telemetry."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0005_indexing_status"
down_revision = "0004_runtime_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add bounded indexing observations to the worker heartbeat."""
    op.add_column(
        "worker_runtime_status",
        sa.Column("indexing_queue_depth", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "camera_runtime_status",
        sa.Column("camera_session_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_camera_runtime_status_session_id",
        "camera_runtime_status",
        "camera_sessions",
        ["camera_session_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.add_column(
        "camera_runtime_status",
        sa.Column("detector_framerate", sa.Float(), nullable=False, server_default="0"),
    )
    op.alter_column("camera_runtime_status", "detector_framerate", server_default=None)
    op.add_column(
        "worker_runtime_status",
        sa.Column("last_searchable_latency_seconds", sa.Float(), nullable=True),
    )
    op.alter_column("worker_runtime_status", "indexing_queue_depth", server_default=None)


def downgrade() -> None:
    """Remove indexing observations from the worker heartbeat."""
    op.drop_column("worker_runtime_status", "last_searchable_latency_seconds")
    op.drop_column("worker_runtime_status", "indexing_queue_depth")
    op.drop_column("camera_runtime_status", "detector_framerate")
    op.drop_constraint(
        "fk_camera_runtime_status_session_id",
        "camera_runtime_status",
        type_="foreignkey",
    )
    op.drop_column("camera_runtime_status", "camera_session_id")
