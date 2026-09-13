"""Register the Task 12 live authorization boundary scenario."""

from __future__ import annotations

from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task12_live_runtime import driver_command
from .task12_models import DriverArtifact


async def _run(context: ScenarioContextProtocol) -> ScenarioReport:
    output = context.run_root / "task12-live-boundaries.json"
    async with context.process(
        name="task12-live-boundaries-driver",
        command=driver_command(context, output=output),
    ) as process:
        return_code = await process.wait()
    if return_code != 0:
        message = "Task 12 live-boundaries driver failed"
        raise RuntimeError(message)
    payload = DriverArtifact.model_validate_json(output.read_text(encoding="utf-8"))
    checks = tuple(
        Check(name=name, passed=value, detail="observed by isolated real-runtime phase")
        for name, value in sorted(payload.checks.items())
    )
    return ScenarioReport(
        checks=checks,
        artifact_paths=(
            output,
            context.run_root / "live-boundaries",
            context.run_root / "live-boundaries" / "task12-resource-manifest.json",
        ),
        evidence_kind=EvidenceKind.REAL,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("camera-auth-live-boundaries"),
        runner=_run,
        required_commands=("bwrap", "docker", "ffmpeg", "ffprobe", "node", "openssl", "uv"),
        timeout_seconds=900.0,
    )
)
