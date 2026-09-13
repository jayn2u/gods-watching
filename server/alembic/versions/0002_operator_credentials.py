"""Add the singleton operator credential relation."""

from alembic import op

revision = "0002_operator_credentials"
down_revision = "0001_storage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create the migration-owned operator credential row."""
    op.execute(
        """
        CREATE TABLE operator_credentials (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            password_hash text NOT NULL,
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )


def downgrade() -> None:
    """Remove the operator credential relation."""
    op.execute("DROP TABLE operator_credentials")
