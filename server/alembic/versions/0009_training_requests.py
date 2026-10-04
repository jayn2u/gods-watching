"""Add expiring API/supervisor requests with lease fencing."""

from alembic import op

revision = "0009_training_requests"
down_revision = "0008_training_jobs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Persist request identity, bounded response, expiry, and worker lease state."""
    op.execute(
        """
        CREATE TABLE training_requests (
            id uuid PRIMARY KEY,
            request_id uuid NOT NULL,
            kind varchar(16) NOT NULL CHECK (kind IN ('preflight', 'submit', 'resume')),
            parent_job_id uuid REFERENCES training_jobs(id) ON DELETE RESTRICT,
            job_id uuid REFERENCES training_jobs(id) ON DELETE RESTRICT,
            config_snapshot jsonb NOT NULL,
            dataset_snapshot jsonb,
            dataset_fingerprint varchar(64)
                CHECK (dataset_fingerprint IS NULL OR dataset_fingerprint ~ '^[0-9a-f]{64}$'),
            source_fingerprint varchar(64) NOT NULL
                CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
            phase varchar(16) NOT NULL DEFAULT 'pending'
                CHECK (phase IN ('pending', 'accepted', 'refused', 'expired', 'failed')),
            response_snapshot jsonb,
            lease_generation bigint NOT NULL DEFAULT 0 CHECK (lease_generation >= 0),
            lease_owner varchar(128)
                CHECK (lease_owner IS NULL OR char_length(lease_owner) BETWEEN 1 AND 128),
            lease_expires_at timestamptz,
            expires_at timestamptz NOT NULL,
            error varchar(1000) CHECK (error IS NULL OR char_length(error) <= 1000),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            resolved_at timestamptz,
            CONSTRAINT uq_training_requests_request_id UNIQUE (request_id),
            CONSTRAINT ck_training_requests_expiry CHECK (expires_at > created_at)
        )
        """
    )
    op.execute(
        """
        CREATE INDEX ix_training_requests_pending_order
            ON training_requests (phase, expires_at, created_at)
            WHERE phase = 'pending'
        """
    )
    op.execute(
        """
        CREATE FUNCTION prevent_training_request_input_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.request_id IS DISTINCT FROM OLD.request_id
                OR NEW.kind IS DISTINCT FROM OLD.kind
                OR NEW.parent_job_id IS DISTINCT FROM OLD.parent_job_id
                OR NEW.config_snapshot IS DISTINCT FROM OLD.config_snapshot
                OR NEW.dataset_snapshot IS DISTINCT FROM OLD.dataset_snapshot
                OR NEW.dataset_fingerprint IS DISTINCT FROM OLD.dataset_fingerprint
                OR NEW.source_fingerprint IS DISTINCT FROM OLD.source_fingerprint
                OR NEW.expires_at IS DISTINCT FROM OLD.expires_at THEN
                RAISE EXCEPTION 'training request inputs are immutable'
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_training_requests_immutable_inputs
            BEFORE UPDATE ON training_requests
            FOR EACH ROW EXECUTE FUNCTION prevent_training_request_input_mutation()
        """
    )


def downgrade() -> None:
    """Remove the request handshake table and immutable-input trigger."""
    op.execute("DROP TRIGGER IF EXISTS trg_training_requests_immutable_inputs ON training_requests")
    op.execute("DROP FUNCTION IF EXISTS prevent_training_request_input_mutation()")
    op.execute("DROP INDEX IF EXISTS ix_training_requests_pending_order")
    op.execute("DROP TABLE IF EXISTS training_requests")
