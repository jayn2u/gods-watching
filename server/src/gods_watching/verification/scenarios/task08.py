"""Exercise both pinned CLIP Triton models on the real GPU."""

import json
from typing import Final

from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario
from gods_watching.verification.scenarios.task08_models import ClipProof
from gods_watching.verification.scenarios.task08_runtime import (
    ASSET_ROOT,
    FIXTURE,
    cleanup,
    docker_run,
    failure_container,
    run_command,
    wait_ready,
)

_NORM_TOLERANCE: Final = 1e-5
_REFERENCE_COSINE: Final = 0.999


async def _run_clip(context: ScenarioContextProtocol) -> ScenarioReport:
    container = f"{context.compose_project}-task8-clip"
    image_path = context.run_root / "clip-input.jpg"
    proof_path = context.run_root / "task-8-clip.json"
    extract_code = await run_command(
        context,
        name="task8-fixture-frame",
        command=(
            "ffmpeg",
            "-v",
            "error",
            "-ss",
            "2",
            "-i",
            str(FIXTURE),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(image_path),
        ),
    )
    started = await run_command(
        context,
        name="task8-clip-container",
        command=docker_run(
            context,
            container=container,
            models=ASSET_ROOT,
            selected=("clip_image", "clip_text"),
        ),
    )
    try:
        ready = await wait_ready(context, container) if started == 0 else 1
        probe = (
            await run_command(
                context,
                name="task8-real-gpu-probe",
                command=(
                    "docker",
                    "exec",
                    container,
                    "python",
                    "/qa/triton_probe.py",
                    "/evidence/clip-input.jpg",
                    "/evidence/task-8-clip.json",
                ),
            )
            if ready == 0
            else 1
        )
        adapter_probe = (
            await run_command(
                context,
                name="task8-typed-adapter-probe",
                command=(
                    "docker",
                    "exec",
                    "--env",
                    "PYTHONPATH=/client-site-packages:/app",
                    container,
                    "python",
                    "/qa/adapter_probe.py",
                    "/evidence/clip-input.jpg",
                    "/evidence/adapter.json",
                ),
            )
            if probe == 0
            else 1
        )
    finally:
        await cleanup(context, container)
    proof = ClipProof.model_validate_json(proof_path.read_text(encoding="utf-8"))
    vector_contract = (
        proof.text.shape == proof.image.shape == (2, 512)
        and proof.text.finite
        and proof.image.finite
        and all(
            abs(norm - 1.0) <= _NORM_TOLERANCE for norm in (*proof.text.norms, *proof.image.norms)
        )
    )
    return ScenarioReport(
        checks=(
            Check(
                name="clip-models-ready",
                passed=extract_code == started == ready == 0,
                detail="both explicitly loaded models became ready",
            ),
            Check(
                name="real-gpu-unit-vectors",
                passed=probe == 0 and vector_contract,
                detail=f"{proof.cuda_device}; text/image batches are finite FP32[512] unit vectors",
            ),
            Check(
                name="direct-reference-match",
                passed=min(*proof.text.reference_cosines, *proof.image.reference_cosines)
                >= _REFERENCE_COSINE,
                detail="both modalities meet cosine >= 0.999 against direct pinned CLIP",
            ),
            Check(
                name="batch-behavior",
                passed=proof.batches_match,
                detail="batch size two preserves deterministic same-image output",
            ),
            Check(
                name="typed-adapter-real-grpc",
                passed=adapter_probe == 0,
                detail="production adapter returned unit image/text vectors over real gRPC",
            ),
        ),
        artifact_paths=(proof_path, image_path, context.run_root / "adapter.json"),
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_clip_errors(context: ScenarioContextProtocol) -> ScenarioReport:
    container = f"{context.compose_project}-task8-errors"
    request_path = context.run_root / "request-errors.json"
    _ = await run_command(
        context,
        name="task8-errors-container",
        command=docker_run(
            context, container=container, models=ASSET_ROOT, selected=("clip_image", "clip_text")
        ),
    )
    try:
        ready = await wait_ready(context, container)
        request_code = (
            await run_command(
                context,
                name="task8-invalid-requests",
                command=(
                    "docker",
                    "exec",
                    container,
                    "python",
                    "/qa/error_probe.py",
                    "/evidence/request-errors.json",
                ),
            )
            if ready == 0
            else 1
        )
    finally:
        await cleanup(context, container)
    empty_models = context.runtime_root / "empty-models"
    empty_models.mkdir()
    missing_rejected = await failure_container(
        context, suffix="missing-snapshot", models=empty_models
    )
    revision_rejected = await failure_container(
        context, suffix="revision-mismatch", models=ASSET_ROOT, revision="wrong-revision"
    )
    summary_path = context.run_root / "task-8-errors.json"
    summary = {
        "invalid_requests_rejected": request_code == 0,
        "missing_snapshot_not_ready": missing_rejected,
        "revision_mismatch_not_ready": revision_rejected,
    }
    _ = summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ScenarioReport(
        checks=(
            Check(
                name="malformed-inputs-rejected",
                passed=request_code == 0,
                detail="bad image, empty/unsupported text and >77 tokens returned explicit errors",
            ),
            Check(
                name="missing-snapshot-not-ready",
                passed=missing_rejected,
                detail="missing local snapshot prevents model readiness",
            ),
            Check(
                name="revision-mismatch-not-ready",
                passed=revision_rejected,
                detail="unapproved revision prevents model readiness",
            ),
        ),
        artifact_paths=(
            summary_path,
            request_path,
            context.run_root / "missing-snapshot.log",
            context.run_root / "revision-mismatch.log",
        ),
        evidence_kind=EvidenceKind.REAL,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("clip"),
        runner=_run_clip,
        required_commands=("docker", "ffmpeg"),
        timeout_seconds=600.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("clip-errors"),
        runner=_run_clip_errors,
        required_commands=("docker",),
        timeout_seconds=600.0,
    )
)
