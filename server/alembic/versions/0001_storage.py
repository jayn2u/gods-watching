"""Create the transactional storage foundation."""

from alembic import op

revision = "0001_storage"
down_revision = None
branch_labels = None
depends_on = None


def _execute_script(script: str) -> None:
    for statement in script.split(";"):
        if statement.strip():
            op.execute(statement)


def upgrade() -> None:
    """Create pgvector and all authoritative application relations."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _execute_script("""
        CREATE TABLE cameras (
            id uuid PRIMARY KEY,
            name varchar(80) NOT NULL UNIQUE CHECK (char_length(name) BETWEEN 1 AND 80),
            source_ciphertext bytea NOT NULL,
            source_host varchar(255) NOT NULL,
            source_port integer,
            detection_enabled boolean NOT NULL DEFAULT true,
            detection_threshold double precision NOT NULL DEFAULT 0.5
                CHECK (detection_threshold BETWEEN 0.1 AND 0.95),
            version integer NOT NULL DEFAULT 1 CHECK (version >= 1),
            deleted_at timestamptz,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        CREATE TABLE camera_sessions (
            id uuid PRIMARY KEY,
            camera_id uuid NOT NULL REFERENCES cameras(id) ON DELETE RESTRICT,
            generation_id uuid NOT NULL,
            cause varchar(32) NOT NULL,
            started_at timestamptz NOT NULL DEFAULT now(),
            ended_at timestamptz
        );
        CREATE TABLE appearances (
            id uuid PRIMARY KEY,
            camera_id uuid NOT NULL REFERENCES cameras(id) ON DELETE RESTRICT,
            session_id uuid NOT NULL REFERENCES camera_sessions(id) ON DELETE RESTRICT,
            track_id bigint NOT NULL CHECK (track_id >= 0),
            first_seen timestamptz NOT NULL,
            last_seen timestamptz NOT NULL CHECK (last_seen >= first_seen),
            ended_at timestamptz CHECK (ended_at IS NULL OR ended_at >= last_seen),
            representative_version integer NOT NULL CHECK (representative_version >= 1),
            crop_object_key varchar(80) NOT NULL UNIQUE,
            x_min integer NOT NULL,
            y_min integer NOT NULL,
            x_max integer NOT NULL,
            y_max integer NOT NULL,
            source_width integer NOT NULL,
            source_height integer NOT NULL,
            detector_confidence double precision NOT NULL
                CHECK (detector_confidence BETWEEN 0 AND 1),
            crop_quality double precision NOT NULL CHECK (crop_quality >= 0),
            byte_size bigint NOT NULL CHECK (byte_size > 0),
            embedded_at timestamptz NOT NULL,
            model_id varchar(255) NOT NULL,
            model_revision varchar(255) NOT NULL,
            embedding vector(512),
            tombstoned_at timestamptz,
            CONSTRAINT uq_appearance_track UNIQUE (camera_id, session_id, track_id),
            CONSTRAINT ck_appearances_bbox_origin CHECK (x_min >= 0 AND y_min >= 0),
            CONSTRAINT ck_appearances_bbox_extent CHECK (x_max > x_min AND y_max > y_min),
            CONSTRAINT ck_appearances_bbox_bounds
                CHECK (x_max <= source_width AND y_max <= source_height)
        );
        CREATE INDEX ix_appearances_camera_time
            ON appearances (camera_id, first_seen, last_seen);
        CREATE INDEX ix_appearances_embedding_hnsw
            ON appearances USING hnsw (embedding vector_cosine_ops)
            WHERE tombstoned_at IS NULL;
        CREATE TABLE crop_gc (
            id uuid PRIMARY KEY,
            object_key varchar(80) NOT NULL UNIQUE,
            byte_size bigint NOT NULL CHECK (byte_size > 0),
            enqueued_at timestamptz NOT NULL DEFAULT now(),
            attempts integer NOT NULL DEFAULT 0 CHECK (attempts >= 0),
            last_error text
        );
        CREATE TABLE sessions (
            id uuid PRIMARY KEY,
            token_hash bytea NOT NULL UNIQUE,
            created_at timestamptz NOT NULL DEFAULT now(),
            last_activity_at timestamptz NOT NULL,
            idle_expires_at timestamptz NOT NULL,
            absolute_expires_at timestamptz NOT NULL,
            revoked_at timestamptz
        );
        CREATE TABLE settings (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            retention_days integer NOT NULL DEFAULT 7 CHECK (retention_days >= 1),
            quota_bytes bigint NOT NULL DEFAULT 100000000000 CHECK (quota_bytes > 0),
            wall_slot_ids uuid[] NOT NULL DEFAULT '{}',
            updated_at timestamptz NOT NULL DEFAULT now()
        );
        INSERT INTO settings (singleton) VALUES (true);
        """)


def downgrade() -> None:
    """Remove the application schema and its pgvector extension."""
    _execute_script("""
        DROP TABLE settings;
        DROP TABLE sessions;
        DROP TABLE crop_gc;
        DROP TABLE appearances;
        DROP TABLE camera_sessions;
        DROP TABLE cameras;
        DROP EXTENSION vector;
        """)
