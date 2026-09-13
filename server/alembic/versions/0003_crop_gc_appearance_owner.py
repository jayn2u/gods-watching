"""Link crop garbage obligations to their owning appearance."""

from sqlalchemy import Column
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0003_crop_gc_appearance_owner"
down_revision = "0002_operator_credentials"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add ownership after proving existing crop garbage is matchable."""
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM crop_gc AS garbage
                LEFT JOIN appearances AS appearance
                    ON appearance.crop_object_key = garbage.object_key
                WHERE appearance.id IS NULL
            ) THEN
                RAISE EXCEPTION
                    '0003 requires every existing crop_gc key to match an appearance; '
                    'drain unmatched crop_gc rows before upgrading';
            END IF;
        END $$
        """
    )
    op.add_column(
        "crop_gc",
        Column("appearance_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.execute(
        """
        UPDATE crop_gc AS garbage
        SET appearance_id = appearance.id
        FROM appearances AS appearance
        WHERE garbage.object_key = appearance.crop_object_key
          AND garbage.appearance_id IS NULL
        """
    )
    op.create_foreign_key(
        "fk_crop_gc_appearance_id",
        "crop_gc",
        "appearances",
        ["appearance_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_crop_gc_appearance_order",
        "crop_gc",
        ["appearance_id", "enqueued_at", "id"],
    )


def downgrade() -> None:
    """Remove ownership metadata while retaining crop garbage rows."""
    op.drop_index("ix_crop_gc_appearance_order", table_name="crop_gc")
    op.drop_constraint("fk_crop_gc_appearance_id", "crop_gc", type_="foreignkey")
    op.drop_column("crop_gc", "appearance_id")
