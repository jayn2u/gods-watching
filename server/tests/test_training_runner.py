from __future__ import annotations

import os
import subprocess
import sys
import textwrap

from gods_watching.training.runner import _safe_error_message  # pyright: ignore[reportPrivateUsage]


def test_runner_gate_requires_the_fixed_record_and_does_not_import_engine() -> None:
    script = textwrap.dedent(
        """
        import io
        import sys
        from gods_watching.training.runner import ownership_gate_is_open

        assert not ownership_gate_is_open(io.BytesIO(b""))
        assert not ownership_gate_is_open(io.BytesIO(b"ready\\n"))
        assert ownership_gate_is_open(io.BytesIO(b"owned\\n"))
        assert "gods_watching.training.engine" not in sys.modules
        assert "torch" not in sys.modules
        """
    )
    environment = os.environ.copy()
    completed = subprocess.run(  # noqa: S603 - fixed local Python and literal test script.
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


def test_safe_error_message_never_returns_exception_details() -> None:
    poison = "secret-token /srv/private/captions/train.txt subject wore a red coat"
    error = RuntimeError(poison)

    assert _safe_error_message(error) == "RuntimeError"
    assert poison not in _safe_error_message(error)

    class TrainingOutOfMemoryError(RuntimeError):
        pass

    assert (
        _safe_error_message(TrainingOutOfMemoryError(poison))
        == "training ran out of GPU memory"
    )
