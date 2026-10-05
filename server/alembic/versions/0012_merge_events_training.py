"""Merge the camera-event and training migration branches."""

revision = "0012_merge_events_training"
down_revision = ("0008_camera_events", "0011_training_evaluation_report")
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Join two already-applied schema branches without changing either schema."""


def downgrade() -> None:
    """Return control to the two independent branch heads."""
