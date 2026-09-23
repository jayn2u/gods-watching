"""Add durable CLIP model identity, transition staging, and dynamic vectors."""

from alembic import op

revision = "0007_model_selection"
down_revision = "0006_live_detection_latest"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Widen appearance vectors without dropping existing embeddings."""
    # The old fixed-512 index depends on the column typmod.  Drop it before
    # changing the column to an unconstrained vector and rebuild dimension-
    # specific expression indexes below so pgvector still uses HNSW retrieval.
    op.execute("DROP INDEX IF EXISTS ix_appearances_embedding_hnsw")
    op.execute(
        """
        ALTER TABLE appearances
            ADD COLUMN embedding_dimension integer NOT NULL DEFAULT 512
                CHECK (embedding_dimension > 0)
        """
    )
    op.execute(
        """
        ALTER TABLE appearances
            ALTER COLUMN embedding TYPE vector
            USING embedding::vector
        """
    )
    op.execute(
        """
        ALTER TABLE appearances
            ADD CONSTRAINT ck_appearances_embedding_dimension
            CHECK (embedding IS NULL OR embedding_dimension = vector_dims(embedding))
        """
    )
    op.execute(
        """
        CREATE INDEX ix_appearances_embedding_hnsw_512
            ON appearances USING hnsw ((embedding::vector(512)) vector_cosine_ops)
            WHERE tombstoned_at IS NULL AND embedding IS NOT NULL
                AND embedding_dimension = 512
        """
    )
    op.execute(
        """
        CREATE INDEX ix_appearances_embedding_hnsw_768
            ON appearances USING hnsw ((embedding::vector(768)) vector_cosine_ops)
            WHERE tombstoned_at IS NULL AND embedding IS NOT NULL
                AND embedding_dimension = 768
        """
    )
    op.execute(
        """
        CREATE TABLE active_model_identity (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            model_id varchar(255) NOT NULL,
            model_revision varchar(255) NOT NULL,
            embedding_dimension integer NOT NULL CHECK (embedding_dimension > 0),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        INSERT INTO active_model_identity
            (singleton, model_id, model_revision, embedding_dimension)
        SELECT true, 'openai/clip-vit-base-patch16',
            '57c216476eefef5ab752ec549e440a49ae4ae5f3', 512
        WHERE NOT EXISTS (SELECT 1 FROM active_model_identity)
        """
    )
    op.execute(
        """
        CREATE TABLE model_transition_jobs (
            id uuid PRIMARY KEY,
            source_model_id varchar(255) NOT NULL,
            source_model_revision varchar(255) NOT NULL,
            source_dimension integer NOT NULL CHECK (source_dimension > 0),
            target_model_id varchar(255) NOT NULL,
            target_model_revision varchar(255) NOT NULL,
            target_dimension integer NOT NULL CHECK (target_dimension > 0),
            phase varchar(32) NOT NULL,
            total integer NOT NULL DEFAULT 0 CHECK (total >= 0),
            processed integer NOT NULL DEFAULT 0 CHECK (processed >= 0 AND processed <= total),
            skipped integer NOT NULL DEFAULT 0 CHECK (skipped >= 0 AND skipped <= processed),
            skip_reasons jsonb NOT NULL DEFAULT '{}'::jsonb,
            error text,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            finished_at timestamptz
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_model_transition_active
            ON model_transition_jobs ((true))
            WHERE phase IN ('queued', 'preparing', 'reindexing', 'activating', 'rolling_back')
        """
    )
    op.execute(
        """
        CREATE TABLE model_transition_stages (
            job_id uuid NOT NULL REFERENCES model_transition_jobs(id) ON DELETE CASCADE,
            appearance_id uuid NOT NULL REFERENCES appearances(id) ON DELETE CASCADE,
            source_crop_object_key varchar(80) NOT NULL,
            source_representative_version integer NOT NULL
                CHECK (source_representative_version >= 1),
            embedding_dimension integer NOT NULL CHECK (embedding_dimension > 0),
            embedding vector,
            skip_reason varchar(64),
            processed_at timestamptz,
            PRIMARY KEY (job_id, appearance_id),
            CHECK (
                processed_at IS NULL OR embedding IS NOT NULL OR skip_reason IS NOT NULL
            ),
            CHECK (
                embedding IS NULL OR embedding_dimension = vector_dims(embedding)
            )
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_model_transition_stages_job_status
            ON model_transition_stages (job_id, processed_at, appearance_id)
        """
    )


def downgrade() -> None:
    """Restore the original fixed 512-dimensional storage shape."""
    op.execute("DROP INDEX IF EXISTS ix_model_transition_stages_job_status")
    op.execute("DROP TABLE IF EXISTS model_transition_stages")
    op.execute("DROP INDEX IF EXISTS uq_model_transition_active")
    op.execute("DROP TABLE IF EXISTS model_transition_jobs")
    op.execute("DROP TABLE IF EXISTS active_model_identity")
    op.execute("DROP INDEX IF EXISTS ix_appearances_embedding_hnsw_512")
    op.execute("DROP INDEX IF EXISTS ix_appearances_embedding_hnsw_768")
    op.execute(
        """
        ALTER TABLE appearances
            ALTER COLUMN embedding TYPE vector(512)
            USING embedding::vector(512)
        """
    )
    op.execute(
        "ALTER TABLE appearances DROP CONSTRAINT IF EXISTS ck_appearances_embedding_dimension"
    )
    op.execute("ALTER TABLE appearances DROP COLUMN embedding_dimension")
    op.execute(
        """
        CREATE INDEX ix_appearances_embedding_hnsw
            ON appearances USING hnsw (embedding vector_cosine_ops)
            WHERE tombstoned_at IS NULL
        """
    )
