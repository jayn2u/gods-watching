"""Task-six scenario proving the harness against a real disposable process."""

import sys

from .models import Check, EvidenceKind, ScenarioContextProtocol, ScenarioReport


async def run_harness_self_check(context: ScenarioContextProtocol) -> ScenarioReport:
    """Start and clean up a real child process without claiming product health."""
    manifest = context.run_root / "resource-manifest.json"
    program = (
        "import pathlib,sys; text=pathlib.Path(sys.argv[1]).read_text(); "
        "sys.exit(0 if 'self-check-process' in text and 'registered' in text else 1)"
    )
    command = (sys.executable, "-c", program, str(manifest))
    async with context.process(name="self-check-process", command=command) as process:
        return_code = await process.wait()
    return ScenarioReport(
        checks=(
            Check(
                name="manifest-precedes-process-spawn",
                passed=return_code == 0,
                detail="child process observed its preregistered resource manifest",
            ),
        ),
        evidence_kind=EvidenceKind.SYNTHETIC,
    )
