from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_MARKER = _REPOSITORY_ROOT / "runtime" / "assets" / "models" / "clip" / "gods-watching-model.json"


def test_training_image_capability_allows_only_approved_bind_writes(tmp_path: Path) -> None:
    image = os.environ.get("GW_TRAINING_IMAGE")
    if not image:
        pytest.skip("set GW_TRAINING_IMAGE to the final built image for the DAC/mount probe")
    if not _MARKER.is_file():
        pytest.fail("the locked read-only B/16 identity marker is not prepared")
    marker_mode = _MARKER.stat().st_mode & 0o777
    if marker_mode != 0o600 or _MARKER.stat().st_uid != 0:
        pytest.fail("this probe requires the preserved root-owned 0600 baseline marker")

    run_root = tmp_path / "runs"
    publication_root = tmp_path / "imported"
    dataset_root = tmp_path / "dataset"
    for directory in (run_root, publication_root, dataset_root):
        _ = directory.mkdir(mode=0o700)
    _ = (dataset_root / "reid_raw.json").write_text("{}", encoding="utf-8")

    trainer_script = """from pathlib import Path
import os
import stat
assert os.geteuid() == 0
assert Path('/models/clip/gods-watching-model.json').read_text()
Path('/runs/smoke-write.txt').write_text('run')
Path('/models/imported/smoke-write.txt').write_text('publication')
job_dir = Path('/runs/jobs/smoke-job')
job_dir.parent.mkdir(mode=0o700, parents=True)
job_dir.mkdir(mode=0o700)
status_fd = os.open(job_dir / 'status.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(status_fd, 'w') as status:
    status.write('ready')
assert job_dir.parent.stat().st_uid == 0
assert stat.S_IMODE(job_dir.parent.stat().st_mode) == 0o700
assert job_dir.stat().st_uid == 0
assert stat.S_IMODE(job_dir.stat().st_mode) == 0o700
status_path = job_dir / 'status.json'
assert status_path.stat().st_uid == 0
assert stat.S_IMODE(status_path.stat().st_mode) == 0o600
for path in (Path('/dataset/reid_raw.json'), Path('/models/clip/gods-watching-model.json')):
    try:
        path.write_text('changed')
    except OSError:
        continue
    raise SystemExit(f'read-only input accepted a write: {path}')
    """
    trainer = subprocess.run(  # noqa: S603
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            "0:0",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "DAC_OVERRIDE",
            "--tmpfs",
            "/tmp",  # noqa: S108 - the isolated container needs writable temporary space
            "--mount",
            f"type=bind,src={run_root},dst=/runs",
            "--mount",
            f"type=bind,src={_MARKER.parents[1]},dst=/models,readonly",
            "--mount",
            f"type=bind,src={dataset_root},dst=/dataset,readonly",
            "--mount",
            f"type=bind,src={publication_root},dst=/models/imported",
            image,
            "python",
            "-c",
            trainer_script,
        ),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert trainer.returncode == 0, trainer.stderr
    assert (run_root / "smoke-write.txt").read_text(encoding="utf-8") == "run"
    assert (publication_root / "smoke-write.txt").read_text(encoding="utf-8") == "publication"

    api_script = """from pathlib import Path
assert Path('/runs/jobs/smoke-job/status.json').read_text() == 'ready'
try:
    Path('/runs/api-write.txt').write_text('forbidden')
except OSError:
    pass
else:
    raise SystemExit('read-only API run mount accepted a write')
"""
    api_reader = subprocess.run(  # noqa: S603
        (
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--user",
            "0:0",
            "--mount",
            f"type=bind,src={run_root},dst=/runs,readonly",
            image,
            "python",
            "-c",
            api_script,
        ),
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert api_reader.returncode == 0, api_reader.stderr
    assert not (run_root / "api-write.txt").exists()
