"""Persist the safe held-out evaluation that binds candidate publication."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0011_training_evaluation_report"
down_revision = "0010_training_engine_completion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Store only the public evaluation summary, never test examples or paths."""
    op.add_column(
        "training_jobs",
        sa.Column("evaluation_report", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Remove the public evaluation summary."""
    op.drop_column("training_jobs", "evaluation_report")
