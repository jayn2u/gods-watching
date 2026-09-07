"""Verify real YOLO11s detector inference through Triton gRPC."""

import hashlib
import json
from typing import Final

import anyio

from gods_watching.verification.models import (
    Check,
    EvidenceKind,
    ImplementedScenario,
    ScenarioContextProtocol,
    ScenarioReport,
)
from gods_watching.verification.registry import parse_scenario_name, register_scenario

from .task07_models import ModelConfig
from .task07_probe import (
    ASSETS_ROOT,
    IMAGE_ID,
    REPOSITORY_ROOT,
    WEIGHTS_SHA256,
    healthy_run,
)
from .task07_runtime import (
    capture_logs,
    remove_detector,
    resource_path,
    start_detector,
    wait_for_endpoint,
)

_MAX_BATCH_SIZE: Final = 4
_QUEUE_DELAY_MICROSECONDS: Final = 10_000
_LOW_CONFIDENCE: Final = 0.1
_IMPLICIT_DEFAULT_CONFIDENCE: Final = 0.25


def _contract_matches(config: ModelConfig) -> bool:
    inputs = {(tensor.name, tensor.data_type, tensor.dims) for tensor in config.input}
    outputs = {(tensor.name, tensor.data_type, tensor.dims) for tensor in config.output}
    return (
        config.max_batch_size == _MAX_BATCH_SIZE
        and config.dynamic_batching.max_queue_delay_microseconds == _QUEUE_DELAY_MICROSECONDS
        and inputs == {("IMAGE", "TYPE_STRING", (1,)), ("CONFIDENCE", "TYPE_FP32", (1,))}
        and outputs == {("BOXES", "TYPE_FP32", (300, 6)), ("COUNT", "TYPE_INT32", (1,))}
    )


async def _run_detector(context: ScenarioContextProtocol) -> ScenarioReport:
    run = await healthy_run(context, suffix="detector", probe_name="task-7-detections.json")
    observed_batches = tuple(
        stat.batch_size for model in run.stats.model_stats for stat in model.batch_stats
    )
    weights_hash = hashlib.sha256((ASSETS_ROOT / "yolo/yolo11s.pt").read_bytes()).hexdigest()
    checks = (
        Check(
            name="real-detector-ready",
            passed=run.ready,
            detail="Triton and detector model reported ready",
        ),
        Check(
            name="exact-tensor-contract",
            passed=_contract_matches(run.config),
            detail="effective config has approved shapes, batching, and queue delay",
        ),
        Check(
            name="low-threshold-preserved",
            passed=run.probe.low_count >= 1
            and run.probe.low_min_confidence is not None
            and _LOW_CONFIDENCE <= run.probe.low_min_confidence < _IMPLICIT_DEFAULT_CONFIDENCE,
            detail=(
                f"0.1 request returned {run.probe.low_count} boxes; "
                f"minimum confidence {run.probe.low_min_confidence}"
            ),
        ),
        Check(
            name="original-pixel-person-boxes",
            passed=run.probe.coordinates_within_original
            and all(item.class_id == 0 for item in run.probe.detections),
            detail=(
                f"validated {run.probe.low_count} class-0 boxes against "
                f"{run.probe.source_dimensions}"
            ),
        ),
        Check(
            name="empty-frame-has-no-detections",
            passed=run.probe.empty_count == 0,
            detail=f"empty frame count={run.probe.empty_count}",
        ),
        Check(
            name="request-errors-are-isolated",
            passed="detector request failed" in run.probe.malformed_error
            and run.probe.post_error_count > 0,
            detail="bad bytes failed and a following valid request succeeded",
        ),
        Check(
            name="dynamic-batching-observed",
            passed=any(size > 1 for size in observed_batches),
            detail=f"observed execution batch sizes {observed_batches}",
        ),
        Check(
            name="pinned-provenance",
            passed=run.image_id == IMAGE_ID and weights_hash == WEIGHTS_SHA256,
            detail=f"image={run.image_id}; weights={weights_hash}",
        ),
    )
    return ScenarioReport(
        checks=checks,
        artifact_paths=(*run.paths, resource_path(context)),
        evidence_kind=EvidenceKind.REAL,
    )


async def _run_detector_errors(context: ScenarioContextProtocol) -> ScenarioReport:
    container = f"{context.compose_project}-detector-missing"
    empty_models = context.runtime_root / "missing-models"
    empty_models.mkdir()
    endpoint = f"127.0.0.1:{context.allocated_port}"
    log_path = context.run_root / "missing-weights-container.log"
    started = await start_detector(
        context,
        container=container,
        repository_root=REPOSITORY_ROOT,
        models_root=empty_models,
    )
    try:
        unavailable = started.return_code == 0 and await wait_for_endpoint(
            endpoint, model_ready=False
        )
        _ = await capture_logs(context, container=container, destination=log_path)
    finally:
        with anyio.CancelScope(shield=True):
            _ = await remove_detector(context, container=container)
    logs = log_path.read_text(encoding="utf-8")
    recovered = await healthy_run(context, suffix="errors", probe_name="task-7-error-probe.json")
    error_path = context.run_root / "task-7-errors.json"
    _ = error_path.write_text(
        json.dumps(
            {
                "missing_weights_unavailable": unavailable,
                "missing_weights_error": "detector weights are missing" in logs,
                "malformed_bytes_error": recovered.probe.malformed_error,
                "post_error_count": recovered.probe.post_error_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ScenarioReport(
        checks=(
            Check(
                name="missing-weights-not-ready",
                passed=unavailable and "detector weights are missing" in logs,
                detail=(
                    "missing read-only model mount left detector unavailable with actionable error"
                ),
            ),
            Check(
                name="malformed-bytes-return-error",
                passed="detector request failed" in recovered.probe.malformed_error,
                detail="malformed encoded bytes returned a gRPC inference error",
            ),
            Check(
                name="valid-request-after-error",
                passed=recovered.probe.post_error_count > 0,
                detail=f"post-error valid count={recovered.probe.post_error_count}",
            ),
        ),
        artifact_paths=(
            error_path,
            log_path,
            *recovered.paths,
            resource_path(context),
        ),
        evidence_kind=EvidenceKind.REAL,
    )


register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("detector"),
        runner=_run_detector,
        required_commands=("docker", "ffmpeg"),
        timeout_seconds=180.0,
    )
)
register_scenario(
    ImplementedScenario(
        name=parse_scenario_name("detector-errors"),
        runner=_run_detector_errors,
        required_commands=("docker", "ffmpeg"),
        timeout_seconds=240.0,
    )
)
