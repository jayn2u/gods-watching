"""Add immutable camera-management event history."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0008_camera_events"
down_revision = "0007_model_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the minimal durable event projection and export indexes."""
    op.create_table(
        "camera_events",
        sa.Column("id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("camera_name", sa.String(length=80), nullable=False),
        sa.CheckConstraint(
            "event_type IN ('camera.created', 'camera.updated', 'camera.deleted')",
            name="ck_camera_events_event_type",
        ),
        sa.CheckConstraint(
            "char_length(camera_name) BETWEEN 1 AND 80",
            name="ck_camera_events_camera_name_length",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_camera_events_occurred_at_id",
        "camera_events",
        ["occurred_at", "id"],
    )
    op.create_index(
        "ix_camera_events_camera_id_id",
        "camera_events",
        ["camera_id", "id"],
    )


def downgrade() -> None:
    """Drop only synthetic event history owned by this migration."""
    op.drop_index("ix_camera_events_camera_id_id", table_name="camera_events")
    op.drop_index("ix_camera_events_occurred_at_id", table_name="camera_events")
    op.drop_table("camera_events")
