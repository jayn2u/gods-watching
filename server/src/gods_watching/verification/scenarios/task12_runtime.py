"""Own the command boundary for Task 12's real integration driver."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gods_watching.verification.models import ScenarioContextProtocol


def driver_command(
    context: ScenarioContextProtocol,
    *,
    mode: str,
    output: Path,
) -> tuple[str, ...]:
    """Build a secret-free command for one isolated Task 12 run."""
    repository_root = Path(__file__).resolve().parents[5]
    return (
        "/usr/bin/env",
        f"GW_TASK12_MODE={mode}",
        f"GW_TASK12_RUN_ROOT={context.run_root}",
        f"GW_TASK12_OUTPUT={output}",
        f"GW_TASK12_PROJECT={context.compose_project}",
        f"GW_TASK12_APP_PORT={context.allocated_port}",
        "uv",
        "run",
        "--project",
        str(repository_root),
        "python",
        "-m",
        "gods_watching.verification.scenarios.task12_driver",
    )
