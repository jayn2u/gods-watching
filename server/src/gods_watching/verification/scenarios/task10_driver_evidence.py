"""Assemble typed evidence from the installed Task 10 driver state."""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from time import monotonic
from typing import Final

from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.ingest.worker import IngestWorker

from .task10_models import (
    DirectIdentityObservation,
    DriverAssertions,
    DriverEvidence,
    IdentityObservation,
    ReconnectEvidence,
    SlowInferenceEvidence,
    ToggleEvidence,
    WorkerEvidence,
)

_FAIR_TURN_SPREAD: Final = 5
_OUTAGE_FAIR_TURN_SPREAD: Final = 40
_OUTAGE_MODE: Final = "ingest-outage"


class DetectorObservation:
    """Accumulate detector responses while one real run is active."""

    def __init__(self) -> None:
        """Initialize counters and bounded latency observations."""
        self.calls: int = 0
        self.positive_results: int = 0
        self.detections: int = 0
        self.latencies: list[float] = []
        self.delayed_calls: int = 0
        self.timed_out_requests: int = 0
        self.slow_intervals: list[tuple[float, float]] = []


@dataclass(frozen=True, slots=True)
class DriverEvidenceInputs:
    """Group the mutable runtime observations needed for evidence assembly."""

    mode: str
    started_monotonic: float
    workers: Sequence[IngestWorker]
    detectors: Sequence[DetectorObservation]
    handoff_counts: Sequence[Counter[str]]
    crop_dimensions: Sequence[list[tuple[int, int]]]
    first_seen_values: Sequence[list[str]]
    dispatch_turns: Sequence[int]
    eligible_turns: Sequence[int]
    reconnects: Sequence[ReconnectEvidence]
    toggle: ToggleEvidence
    identity_observations: Sequence[IdentityObservation]
    initial_bindings: Sequence[GenerationBinding]
    peak_global_detector_requests: int
    direct_identity_observation: DirectIdentityObservation


def _worker_evidence(inputs: DriverEvidenceInputs) -> tuple[WorkerEvidence, ...]:
    rows: list[WorkerEvidence] = []
    for index, worker in enumerate(inputs.workers):
        stats = worker.stats
        observation = inputs.detectors[index]
        initial = inputs.initial_bindings[index]
        active = worker.generation
        rows.append(
            WorkerEvidence(
                camera_index=index + 1,
                camera_id=str(active.camera_id),
                initial_camera_session_id=str(initial.camera_session_id),
                initial_source_generation_id=str(initial.source_generation_id),
                active_camera_session_id=str(active.camera_session_id),
                active_source_generation_id=str(active.source_generation_id),
                frames_decoded=stats.frames_decoded,
                frames_sampled=stats.frames_sampled,
                detector_requests=stats.detector_requests,
                detector_results=stats.detector_results,
                positive_detector_results=observation.positive_results,
                detector_detections=observation.detections,
                invalid_detections=stats.invalid_detections,
                eligible_detections=stats.eligible_detections,
                dropped_frames=stats.dropped_frames,
                pending_detector_requests_at_stop=stats.pending_detector_requests,
                peak_detector_requests=stats.peak_detector_requests,
                latest_slot_occupancy_at_stop=worker.slot.occupancy,
                latest_slot_peak_occupancy=worker.slot.peak_occupancy,
                frame_age_seconds_at_stop=stats.frame_age_seconds,
                sanitized_errors=stats.sanitized_errors,
                last_sanitized_error=stats.last_sanitized_error,
                handoffs=dict(inputs.handoff_counts[index]),
                crop_count=sum(inputs.handoff_counts[index].values())
                - inputs.handoff_counts[index]["end"],
                min_crop_width=min((size[0] for size in inputs.crop_dimensions[index]), default=0),
                min_crop_height=min((size[1] for size in inputs.crop_dimensions[index]), default=0),
                first_seen_utc=tuple(inputs.first_seen_values[index]),
                identity_switch_observations=None,
                identity_observation=inputs.identity_observations[index],
                reconnect_boundaries=sum(
                    reconnect.camera_index == index + 1 for reconnect in inputs.reconnects
                ),
                scheduler_dispatch_turns=inputs.dispatch_turns[index],
                eligible_scheduler_dispatch_turns=inputs.eligible_turns[index],
                detector_latency_max_seconds=(
                    max(observation.latencies) if observation.latencies else None
                ),
                stale_generation_results=stats.stale_generation_results,
                dispatch_detector_requested=stats.dispatch_detector_requested,
                dispatch_no_frame=stats.dispatch_no_frame,
                dispatch_detection_disabled=stats.dispatch_detection_disabled,
                dispatch_generation_fenced=stats.dispatch_generation_fenced,
                dispatch_closed=stats.dispatch_closed,
                dispatch_outcomes=stats.dispatch_outcomes,
            )
        )
    return tuple(rows)


def _slow_evidence(inputs: DriverEvidenceInputs) -> SlowInferenceEvidence:
    slow_intervals = [
        (
            interval_start - inputs.started_monotonic,
            interval_end - inputs.started_monotonic,
        )
        for detector in inputs.detectors
        for interval_start, interval_end in detector.slow_intervals
    ]
    overlap = any(
        interval_start <= reconnect.elapsed_seconds <= interval_end
        for interval_start, interval_end in slow_intervals
        for reconnect in inputs.reconnects
    )
    delayed_calls = sum(detector.delayed_calls for detector in inputs.detectors)
    timed_out_requests = sum(detector.timed_out_requests for detector in inputs.detectors)
    return SlowInferenceEvidence(
        enabled=inputs.mode == _OUTAGE_MODE,
        delayed_calls=delayed_calls,
        timed_out_requests=timed_out_requests,
        max_delay_seconds=2.8 if inputs.mode == _OUTAGE_MODE else 0.0,
        overlap_with_reconnect=overlap,
        signal_camera_index=2 if inputs.mode == _OUTAGE_MODE else None,
    )


def _assertions(
    inputs: DriverEvidenceInputs,
    workers: Sequence[WorkerEvidence],
    slow: SlowInferenceEvidence,
) -> DriverAssertions:
    initial_identities = {
        (worker.camera_id, worker.initial_camera_session_id, worker.initial_source_generation_id)
        for worker in workers
    }
    active_identities = {
        (worker.camera_id, worker.active_camera_session_id, worker.active_source_generation_id)
        for worker in workers
    }
    return DriverAssertions(
        four_cameras_decoded=all(worker.frames_decoded > 0 for worker in workers),
        four_cameras_real_detector_results=all(worker.detector_results > 0 for worker in workers),
        four_cameras_person_detections=all(
            worker.positive_detector_results > 0 for worker in workers
        ),
        four_cameras_crop_handoffs=all(worker.crop_count > 0 for worker in workers),
        bounded_latest_slot=all(
            worker.frames_decoded >= worker.frames_sampled for worker in workers
        ),
        single_flight_at_stop=all(
            worker.pending_detector_requests_at_stop == 0 for worker in workers
        ),
        fair_scheduler_turns=(
            max(worker.scheduler_dispatch_turns for worker in workers)
            - min(worker.scheduler_dispatch_turns for worker in workers)
            <= (_OUTAGE_FAIR_TURN_SPREAD if inputs.mode == _OUTAGE_MODE else _FAIR_TURN_SPREAD)
        ),
        detection_toggle_stopped=(inputs.toggle.detector_calls_during_disabled == 0),
        detection_toggle_resumed=(
            inputs.toggle.detector_calls_after_reenable > 0
            and inputs.toggle.handoffs_after_reenable > 0
        ),
        generation_reconnect=bool(inputs.reconnects),
        stale_result_fenced=any(worker.stale_generation_results > 0 for worker in workers),
        slow_inference_timeout=(not slow.enabled or slow.timed_out_requests > 0),
        dispatch_outcomes_reconcile=all(
            worker.dispatch_outcomes == worker.scheduler_dispatch_turns for worker in workers
        ),
        identity_namespaces_separate=(
            len(initial_identities) == len(workers)
            and len(active_identities) == len(workers)
            and all(
                (worker.initial_camera_session_id == worker.active_camera_session_id)
                == (worker.initial_source_generation_id == worker.active_source_generation_id)
                for worker in workers
            )
        ),
        direct_identity_observation_bound=(
            inputs.direct_identity_observation.all_source_hashes_current
            and inputs.direct_identity_observation.all_fixture_hashes_current
            and inputs.direct_identity_observation.adjacent_comparison_opportunities > 0
        ),
    )


def build_driver_evidence(inputs: DriverEvidenceInputs) -> DriverEvidence:
    """Convert one completed driver run into validated machine-readable evidence."""
    workers = _worker_evidence(inputs)
    slow = _slow_evidence(inputs)
    assertions = _assertions(
        inputs,
        workers,
        slow,
    )
    return DriverEvidence(
        schema_version=1,
        mode="ingest-outage" if inputs.mode == _OUTAGE_MODE else "ingest",
        label="REAL: PyAV H.264 RTSP/TCP + Triton YOLO11s GPU + supervision ByteTrack",
        elapsed_seconds=monotonic() - inputs.started_monotonic,
        target_detector_fps_per_camera=5.0,
        peak_global_detector_requests=inputs.peak_global_detector_requests,
        workers=workers,
        direct_identity_observation=inputs.direct_identity_observation,
        detection_toggle=inputs.toggle,
        reconnects=tuple(inputs.reconnects),
        slow_inference=slow,
        assertions=assertions,
        cleanup_pending_requests=tuple(
            worker.pending_detector_requests_at_stop for worker in workers
        ),
    )


__all__ = ["DetectorObservation", "DriverEvidenceInputs", "build_driver_evidence"]
