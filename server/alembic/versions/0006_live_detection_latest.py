"""Add the replaceable latest detector overlay snapshot."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0006_live_detection_latest"
down_revision = "0005_indexing_status"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create one bounded detector result row per configured camera."""
    op.create_table(
        "camera_detection_latest",
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("camera_session_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("db_generation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("source_generation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("frame_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("boxes", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint("width > 0", name="ck_camera_detection_latest_width"),
        sa.CheckConstraint("height > 0", name="ck_camera_detection_latest_height"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["camera_session_id"], ["camera_sessions.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("camera_id"),
    )


def downgrade() -> None:
    """Remove the latest detector overlay snapshot relation."""
    op.drop_table("camera_detection_latest")
