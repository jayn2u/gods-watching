from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

import gods_watching.training.supervisor as supervisor_module
from gods_watching.training.supervisor import (
    ProcessIdentityStatus,
    process_identity_matches,
    process_identity_status,
    process_start_time,
)


def test_live_orphan_identity_is_detected_and_pid_reuse_is_rejected() -> None:
    pid = os.getpid()
    start_time = process_start_time(pid)

    assert process_identity_matches(pid, start_time)
    assert not process_identity_matches(pid, start_time + 1)


def test_missing_child_identity_fails_closed() -> None:
    assert not process_identity_matches(2_147_483_647, 1)
    assert process_identity_status(2_147_483_647, 1) == ProcessIdentityStatus.GONE


def test_process_identity_read_permission_error_is_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def denied(_pid: int, *, proc_root: Path | None = None) -> int:
        del proc_root
        raise PermissionError

    monkeypatch.setattr(supervisor_module, "process_start_time", denied)

    assert process_identity_status(123, 987654) == ProcessIdentityStatus.UNVERIFIABLE


def test_proc_stat_reader_handles_spaces_and_parentheses_in_comm(tmp_path: Path) -> None:
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    # /proc/<pid>/stat field 22 is index 19 after the closing parenthesis.
    fields = ["S", *["0"] * 18, "987654", "0"]
    _ = (process_dir / "stat").write_text(
        f"123 (name with ) chars) {' '.join(fields)}", encoding="utf-8"
    )

    assert process_start_time(123, proc_root=tmp_path) == 987654
