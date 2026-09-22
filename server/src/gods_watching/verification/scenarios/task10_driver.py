"""Drive real PyAV, Triton YOLO, and ByteTrack work for Task 10."""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from time import monotonic
from typing import Final, override
from uuid import uuid4

import anyio

from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import GenerationBinding, PipelineHandoff
from gods_watching.inference.detector import (
    DetectorClient,
    DetectorRequest,
    DetectorResult,
    TritonGrpcDetectorTransport,
)
from gods_watching.ingest.worker import IngestCoordinator, IngestWorker, IngestWorkerConfiguration
from gods_watching.media.models import SourceGenerationId

from .task10_artifacts import load_identity_observation
from .task10_driver_evidence import (
    DetectorObservation,
    DriverEvidenceInputs,
    build_driver_evidence,
)
from .task10_errors import Task10ExecutionError
from .task10_models import IdentityObservation, ReconnectEvidence, ToggleEvidence

_OUTAGE_MODE: Final = "ingest-outage"
_SLOW_CAMERA_INDEX: Final = 2
_SLOW_DELAY_SECONDS: Final = 2.8
_STALE_FENCE_DELAY_SECONDS: Final = 0.9
_IDENTITY_PROXY_MAX_GAP_SECONDS: Final = 2.0
_IDENTITY_PROXY_MIN_IOU: Final = 0.5


class _RealDetector:
    def __init__(
        self,
        client: DetectorClient,
        *,
        camera_index: int,
        mode: str,
        slow_signal: Path,
    ) -> None:
        self._client: DetectorClient = client
        self._camera_index: int = camera_index
        self._mode: str = mode
        self._slow_signal: Path = slow_signal
        self.observation: DetectorObservation = DetectorObservation()

    async def detect(self, request: DetectorRequest) -> DetectorResult:
        started = monotonic()
        self.observation.calls += 1
        result = await self._client.detect(request)
        self.observation.detections += result.count
        if result.count:
            self.observation.positive_results += 1
        delay = self._delay_seconds()
        if delay == 0.0:
            self.observation.latencies.append(monotonic() - started)
            return result
        self.observation.delayed_calls += 1
        if self._camera_index == _SLOW_CAMERA_INDEX:
            _ = self._slow_signal.touch(exist_ok=True)
        slow_started = monotonic()
        try:
            await anyio.sleep(delay)
        except anyio.get_cancelled_exc_class():
            self.observation.timed_out_requests += 1
            self.observation.slow_intervals.append((slow_started, monotonic()))
            raise
        self.observation.slow_intervals.append((slow_started, monotonic()))
        self.observation.latencies.append(monotonic() - started)
        return result

    def _delay_seconds(self) -> float:
        if self._mode != _OUTAGE_MODE:
            return 0.0
        call_remainder = self.observation.calls % 8
        if call_remainder == 0:
            return _STALE_FENCE_DELAY_SECONDS
        if call_remainder != 1:
            return 0.0
        return _SLOW_DELAY_SECONDS


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise Task10ExecutionError(detail=f"required Task 10 environment is missing: {name}")
    return value


def _binding(camera_id: CameraId, *, version: int = 1) -> GenerationBinding:
    return GenerationBinding(
        camera_id=camera_id,
        camera_session_id=CameraSessionId(uuid4()),
        db_generation_id=CameraGenerationId(uuid4()),
        source_generation_id=SourceGenerationId(uuid4()),
        camera_version=version,
    )


def _bbox_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    left_area = max(0.0, left_x2 - left_x1) * max(0.0, left_y2 - left_y1)
    right_area = max(0.0, right_x2 - right_x1) * max(0.0, right_y2 - right_y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def build_toggle_evidence(  # noqa: D103, PLR0913
    *,
    disabled_at: float,
    reenabled_at: float,
    calls_before: int,
    calls_at_disable: int,
    calls_at_reenable: int,
    calls_after: int,
    frames_at_disable: int,
    frames_at_reenable: int,
    end_events: tuple[str, ...],
    handoffs_after_reenable: int,
) -> ToggleEvidence:
    return ToggleEvidence(
        disabled_for_seconds=reenabled_at - disabled_at,
        detector_calls_before=calls_before,
        detector_calls_during_disabled=calls_at_reenable - calls_at_disable,
        detector_calls_after_reenable=calls_after - calls_at_reenable,
        end_events_on_disable=end_events,
        decoded_frames_while_disabled=frames_at_reenable - frames_at_disable,
        handoffs_after_reenable=handoffs_after_reenable,
    )


async def _main() -> None:  # noqa: C901, PLR0915
    mode = _env("GW_TASK10_MODE")
    if mode not in ("ingest", _OUTAGE_MODE):
        raise Task10ExecutionError(detail=f"unsupported Task 10 mode: {mode}")
    rtsp_host = _env("GW_TASK10_RTSP_HOST")
    rtsp_port = int(_env("GW_TASK10_RTSP_PORT"))
    triton_url = _env("GW_TASK10_TRITON_URL")
    output = Path(_env("GW_TASK10_OUTPUT"))
    slow_signal = Path(_env("GW_TASK10_SLOW_SIGNAL"))
    duration = float(os.environ.get("GW_TASK10_DURATION", "20"))
    repository_root = Path(_env("GW_TASK10_REPOSITORY_ROOT"))
    transport = TritonGrpcDetectorTransport(url=triton_url)
    client = DetectorClient(transport=transport, timeout_seconds=2.0)
    workers: list[IngestWorker] = []
    initial_bindings: list[GenerationBinding] = []
    detectors: list[_RealDetector] = []
    handoff_counts: list[Counter[str]] = []
    crop_dimensions: list[list[tuple[int, int]]] = []
    first_seen_values: list[list[str]] = []
    identity_evaluated = [0, 0, 0, 0]
    identity_ambiguous = [0, 0, 0, 0]
    identity_transitions = [0, 0, 0, 0]
    identity_last_ended: list[
        tuple[str, tuple[float, float, float, float], float, str | None] | None
    ] = [
        None,
        None,
        None,
        None,
    ]
    dispatch_turns = [0, 0, 0, 0]
    eligible_turns = [0, 0, 0, 0]
    reconnects: list[ReconnectEvidence] = []
    handoffs_after_reenable = 0
    started_monotonic = monotonic()

    def elapsed() -> float:
        return monotonic() - started_monotonic

    async def make_reconnect(index: int, previous: GenerationBinding) -> GenerationBinding:
        replacement = _binding(previous.camera_id, version=previous.camera_version + 1)
        reconnects.append(
            ReconnectEvidence(
                camera_index=index + 1,
                camera_id=str(previous.camera_id),
                elapsed_seconds=elapsed(),
                previous_source_generation_id=str(previous.source_generation_id),
                replacement_source_generation_id=str(replacement.source_generation_id),
                previous_camera_session_id=str(previous.camera_session_id),
                replacement_camera_session_id=str(replacement.camera_session_id),
            )
        )
        return replacement

    async def consume(index: int, handoff: PipelineHandoff) -> None:
        nonlocal handoffs_after_reenable
        kind = str(handoff.lifecycle.kind)
        handoff_counts[index][kind] += 1
        if kind == "start":
            first_seen_values[index].append(handoff.lifecycle.first_seen.isoformat())
            observation = handoff.lifecycle.observation
            previous = identity_last_ended[index]
            if previous is None:
                identity_ambiguous[index] += 1
            else:
                previous_generation, previous_box, previous_time, previous_reason = previous
                gap = observation.ingress_monotonic - previous_time
                same_interval = (
                    previous_generation == str(handoff.lifecycle.source_generation_id)
                    and previous_reason == "explicit"
                    and 0.0 <= gap <= _IDENTITY_PROXY_MAX_GAP_SECONDS
                )
                if same_interval and _bbox_iou(previous_box, observation.bounding_box) >= (
                    _IDENTITY_PROXY_MIN_IOU
                ):
                    identity_transitions[index] += 1
                else:
                    identity_ambiguous[index] += 1
            identity_last_ended[index] = None
        elif kind == "end":
            observation = handoff.lifecycle.observation
            identity_ambiguous[index] += 1
            identity_last_ended[index] = (
                str(handoff.lifecycle.source_generation_id),
                observation.bounding_box,
                observation.ingress_monotonic,
                None if handoff.lifecycle.end_reason is None else str(handoff.lifecycle.end_reason),
            )
        if handoff.candidate is not None and kind in ("start", "update"):
            identity_evaluated[index] += 1
        if handoff.candidate is not None:
            crop_dimensions[index].append(
                (handoff.candidate.crop.width, handoff.candidate.crop.height)
            )
        if index == 0 and reenabled_at is not None:
            handoffs_after_reenable += 1

    reenabled_at: float | None = None
    for index in range(4):
        binding = _binding(CameraId(uuid4()))
        initial_bindings.append(binding)
        detector = _RealDetector(
            client,
            camera_index=index + 1,
            mode=mode,
            slow_signal=slow_signal,
        )
        detectors.append(detector)
        handoff_counts.append(Counter())
        crop_dimensions.append([])
        first_seen_values.append([])
        workers.append(
            IngestWorker(
                IngestWorkerConfiguration(
                    generation=binding,
                    source_url=f"rtsp://{rtsp_host}:{rtsp_port}/camera-{index + 1}",
                    threshold=0.5,
                    detector_deadline_seconds=2.0,
                    generation_reconnect=lambda previous, index=index: make_reconnect(
                        index, previous
                    ),
                ),
                detector,
                lambda handoff, index=index: consume(index, handoff),
            )
        )

    class _RecordingCoordinator(IngestCoordinator):
        @override
        async def _dispatch(self, camera_id: CameraId) -> None:
            index = next(
                index for index, worker in enumerate(workers) if worker.camera_id == camera_id
            )
            before = workers[index].stats
            dispatch_turns[index] += 1
            await super()._dispatch(camera_id)
            after = workers[index].stats
            if after.detector_requests > before.detector_requests:
                eligible_turns[index] += 1

    coordinator = _RecordingCoordinator()
    for worker in workers:
        coordinator.add(worker)
    stop_event = anyio.Event()
    toggle_results: list[ToggleEvidence] = []

    async def run_coordinator() -> None:
        await coordinator.run(stop_event=stop_event)

    async def exercise_toggle() -> None:
        nonlocal reenabled_at
        await anyio.sleep(min(8.0, duration / 2.0))
        before = detectors[0].observation.calls
        end_events = await workers[0].set_detection_enabled(False)
        calls_at_disable = detectors[0].observation.calls
        frames_at_disable = workers[0].stats.frames_decoded
        disabled_at = monotonic()
        await anyio.sleep(min(2.5, max(0.75, duration / 8.0)))
        _ = await workers[0].set_detection_enabled(True)
        reenabled_at = monotonic()
        calls_at_reenable = detectors[0].observation.calls
        frames_at_reenable = workers[0].stats.frames_decoded
        await anyio.sleep(min(2.5, max(0.75, duration / 8.0)))
        after = detectors[0].observation.calls
        toggle_results.append(
            build_toggle_evidence(
                disabled_at=disabled_at,
                reenabled_at=reenabled_at,
                calls_before=before,
                calls_at_disable=calls_at_disable,
                calls_at_reenable=calls_at_reenable,
                calls_after=after,
                frames_at_disable=frames_at_disable,
                frames_at_reenable=frames_at_reenable,
                end_events=tuple(str(event.lifecycle.kind) for event in end_events),
                handoffs_after_reenable=handoffs_after_reenable,
            )
        )

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(run_coordinator)
            task_group.start_soon(exercise_toggle)
            await anyio.sleep(duration)
            stop_event.set()
    finally:
        await transport.close()
    toggle_result = toggle_results[0]
    evidence = build_driver_evidence(
        DriverEvidenceInputs(
            mode=mode,
            started_monotonic=started_monotonic,
            workers=workers,
            detectors=[detector.observation for detector in detectors],
            handoff_counts=handoff_counts,
            crop_dimensions=crop_dimensions,
            first_seen_values=first_seen_values,
            identity_observations=tuple(
                IdentityObservation(
                    method="track-lifecycle-fragmentation-proxy",
                    real_identity_status="unknown",
                    evaluated_detections=identity_evaluated[index],
                    ambiguous_exclusions=identity_ambiguous[index],
                    observed_transitions=identity_transitions[index],
                )
                for index in range(4)
            ),
            dispatch_turns=dispatch_turns,
            eligible_turns=eligible_turns,
            reconnects=reconnects,
            toggle=toggle_result,
            initial_bindings=initial_bindings,
            peak_global_detector_requests=coordinator.peak_in_flight_count,
            direct_identity_observation=load_identity_observation(repository_root),
        )
    )
    _ = output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    anyio.run(_main)
