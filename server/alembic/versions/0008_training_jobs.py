"""Add durable training jobs and the singleton GPU execution slot."""

from alembic import op

revision = "0008_training_jobs"
down_revision = "0007_model_selection"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create immutable job inputs, progress state, and execution ownership."""
    op.execute(
        """
        CREATE TABLE training_jobs (
            id uuid PRIMARY KEY,
            request_id uuid NOT NULL,
            config_snapshot jsonb NOT NULL,
            dataset_snapshot jsonb NOT NULL,
            dataset_fingerprint varchar(64) NOT NULL
                CHECK (dataset_fingerprint ~ '^[0-9a-f]{64}$'),
            source_fingerprint varchar(64) NOT NULL
                CHECK (source_fingerprint ~ '^[0-9a-f]{64}$'),
            phase varchar(32) NOT NULL CHECK (
                phase IN ('starting', 'training', 'evaluating', 'publishing', 'succeeded',
                    'cancelling', 'cancelled', 'failed', 'interrupted')
            ),
            current_epoch integer NOT NULL DEFAULT 0 CHECK (current_epoch >= 0),
            current_step bigint NOT NULL DEFAULT 0 CHECK (current_step >= 0),
            owner_generation bigint NOT NULL DEFAULT 0 CHECK (owner_generation >= 0),
            child_pid bigint,
            child_start_time bigint,
            cancel_requested boolean NOT NULL DEFAULT false,
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            best_metric double precision CHECK (best_metric IS NULL OR best_metric BETWEEN 0 AND 1),
            checkpoint_path text,
            candidate_model_id varchar(255),
            candidate_revision varchar(255),
            error varchar(1000) CHECK (error IS NULL OR char_length(error) <= 1000),
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now(),
            started_at timestamptz,
            finished_at timestamptz,
            CONSTRAINT uq_training_jobs_request_id UNIQUE (request_id),
            CONSTRAINT ck_training_jobs_child_identity CHECK (
                child_pid IS NULL AND child_start_time IS NULL OR
                child_pid IS NOT NULL AND child_start_time IS NOT NULL AND
                child_pid > 0 AND child_start_time >= 0
            ),
            CONSTRAINT ck_training_jobs_candidate_identity CHECK (
                (candidate_model_id IS NULL) = (candidate_revision IS NULL)
            )
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX uq_training_jobs_one_active
            ON training_jobs ((true))
            WHERE phase IN ('starting', 'training', 'evaluating', 'publishing', 'cancelling')
        """
    )
    op.execute(
        """
        CREATE TABLE training_execution_slots (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            active_job_id uuid UNIQUE
                REFERENCES training_jobs(id) ON DELETE RESTRICT
        )
        """
    )
    op.execute(
        "INSERT INTO training_execution_slots (singleton, active_job_id) VALUES (true, NULL)"
    )
    op.execute(
        """
        CREATE FUNCTION prevent_training_job_input_mutation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.request_id IS DISTINCT FROM OLD.request_id
                OR NEW.config_snapshot IS DISTINCT FROM OLD.config_snapshot
                OR NEW.dataset_snapshot IS DISTINCT FROM OLD.dataset_snapshot
                OR NEW.dataset_fingerprint IS DISTINCT FROM OLD.dataset_fingerprint
                OR NEW.source_fingerprint IS DISTINCT FROM OLD.source_fingerprint THEN
                RAISE EXCEPTION 'training job inputs are immutable'
                    USING ERRCODE = 'check_violation';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_training_jobs_immutable_inputs
            BEFORE UPDATE ON training_jobs
            FOR EACH ROW EXECUTE FUNCTION prevent_training_job_input_mutation()
        """
    )


def downgrade() -> None:
    """Remove the training job schema."""
    op.execute("DROP TRIGGER IF EXISTS trg_training_jobs_immutable_inputs ON training_jobs")
    op.execute("DROP FUNCTION IF EXISTS prevent_training_job_input_mutation()")
    op.execute("DROP TABLE IF EXISTS training_execution_slots")
    op.execute("DROP INDEX IF EXISTS uq_training_jobs_one_active")
    op.execute("DROP TABLE IF EXISTS training_jobs")
