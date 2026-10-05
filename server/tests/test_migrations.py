from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_camera_events_and_training_migrations_share_one_head() -> None:
    repository_root = Path(__file__).parents[2]
    config = Config(str(repository_root / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)

    heads = scripts.get_heads()

    assert len(heads) == 1
    merge = scripts.get_revision(heads[0])
    assert merge is not None
    assert merge.is_merge_point
    assert isinstance(merge.down_revision, tuple)
    assert set(merge.down_revision) == {
        "0008_camera_events",
        "0011_training_evaluation_report",
    }


def test_merge_revision_fits_alembic_version_column() -> None:
    repository_root = Path(__file__).parents[2]
    config = Config(str(repository_root / "alembic.ini"))
    scripts = ScriptDirectory.from_config(config)

    head = scripts.get_revision(scripts.get_heads()[0])

    assert head is not None
    assert len(head.revision) <= 32
