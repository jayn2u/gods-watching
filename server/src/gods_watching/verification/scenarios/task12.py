"""Register real authenticated camera and authorization scenarios."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task12_runtime import driver_command


class _DriverArtifact(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)

    mode: str
    checks: dict[str, bool]
    observations: dict[str, object]


async def _run(
    context: ScenarioContextProtocol,
    *,
    mode: str,
) -> ScenarioReport:
    output = context.run_root / f"task12-{mode}.json"
    async with context.process(
        name=f"task12-{mode}-driver",
        command=driver_command(context, mode=mode, output=output),
    ) as process:
        return_code = await process.wait()
    if return_code != 0:
        failure = f"Task 12 {mode} driver failed"
        raise RuntimeError(failure)
    artifact = _DriverArtifact.model_validate_json(output.read_text(encoding="utf-8"))
    checks = tuple(
        Check(name=name, passed=passed, detail="observed by isolated integration driver")
        for name, passed in sorted(artifact.checks.items())
    )
    browser_path = context.run_root / "task12-browser.json"
    return ScenarioReport(
        checks=checks,
        artifact_paths=(
            output,
            browser_path,
            context.run_root / "task12-resource-manifest.json",
        ),
        evidence_kind=EvidenceKind.REAL,
    )


async def run_camera_auth(context: ScenarioContextProtocol) -> ScenarioReport:
    """Exercise the authenticated camera, settings, CLI, and WHEP surfaces."""
    return await _run(context, mode="happy")


async def run_camera_auth_denied(context: ScenarioContextProtocol) -> ScenarioReport:
    """Exercise denial, throttling, origin, malformed-source, and passive-session behavior."""
    return await _run(context, mode="denied")


_REQUIRED_COMMANDS = ("bwrap", "docker", "ffmpeg", "ffprobe", "node", "openssl", "uv")
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("camera-auth"),
        runner=run_camera_auth,
        required_commands=_REQUIRED_COMMANDS,
        timeout_seconds=300.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("camera-auth-denied"),
        runner=run_camera_auth_denied,
        required_commands=_REQUIRED_COMMANDS,
        timeout_seconds=300.0,
    )
)
