"""Record successful engine completion separately from final evaluation/publication."""

import sqlalchemy as sa

from alembic import op

revision = "0010_training_engine_completion"
down_revision = "0009_training_requests"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add a durable timestamp for training-engine completion."""
    op.add_column(
        "training_jobs",
        sa.Column("engine_completed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove the engine-completion timestamp."""
    op.drop_column("training_jobs", "engine_completed_at")
