"""Turn structured Task 10 driver observations into verification checks."""

from typing import Final

from gods_watching.verification.models import Check

from .task10_models import DriverEvidence, SourceControlEvidence

_MIN_CROP_WIDTH: Final = 32
_MIN_CROP_HEIGHT: Final = 64
_MIN_DISPATCH_TURNS: Final = 10
_FAIR_TURN_SPREAD: Final = 5
_OUTAGE_FAIR_TURN_SPREAD: Final = 40
_OUTAGE_MIN_DISPATCH_FPS: Final = 2.0
_CAMERA_TWO_INDEX: Final = 2


def _check(name: str, passed: bool, detail: str) -> Check:
    return Check(name=name, passed=passed, detail=detail)


def build_checks(
    mode: str,
    driver: DriverEvidence,
    source_control: SourceControlEvidence | None,
) -> tuple[Check, ...]:
    """Evaluate the installed scenario's real-runtime acceptance assertions."""
    workers = driver.workers
    assertions = driver.assertions
    source_restarted = mode == "ingest" or (
        source_control is not None
        and source_control.slow_signal_seen
        and source_control.stop_exit_code == 0
        and source_control.restart_exit_code == 0
    )
    all_crops_valid = all(
        worker.min_crop_width >= _MIN_CROP_WIDTH and worker.min_crop_height >= _MIN_CROP_HEIGHT
        for worker in workers
    )
    first_seen_reported = all(worker.first_seen_utc for worker in workers)
    min_dispatch = min(worker.scheduler_dispatch_turns for worker in workers)
    expected_fps = _OUTAGE_MIN_DISPATCH_FPS if mode == "ingest-outage" else 3.5
    expected_dispatch = max(_MIN_DISPATCH_TURNS, int(driver.elapsed_seconds * expected_fps))
    fair_turn_spread = _OUTAGE_FAIR_TURN_SPREAD if mode == "ingest-outage" else _FAIR_TURN_SPREAD
    camera_two_reconnected = any(
        worker.camera_index == _CAMERA_TWO_INDEX and worker.reconnect_boundaries > 0
        for worker in workers
    )
    initial_namespaces = [
        (worker.camera_id, worker.initial_camera_session_id, worker.initial_source_generation_id)
        for worker in workers
    ]
    active_namespaces = [
        (worker.camera_id, worker.active_camera_session_id, worker.active_source_generation_id)
        for worker in workers
    ]
    return (
        _check(
            "real-four-camera-decode",
            assertions.four_cameras_decoded,
            "all four workers decoded frames from H.264 RTSP/TCP",
        ),
        _check(
            "real-triton-yolo-person-results",
            assertions.four_cameras_real_detector_results
            and assertions.four_cameras_person_detections,
            "all four workers received positive detector results from the pinned GPU model",
        ),
        _check(
            "short-run-scheduler-functional-floor",
            (
                assertions.fair_scheduler_turns
                and max(worker.scheduler_dispatch_turns for worker in workers) - min_dispatch
                <= fair_turn_spread
                and min_dispatch >= expected_dispatch
            ),
            (
                "observed dispatch_turns="
                f"{[worker.scheduler_dispatch_turns for worker in workers]}; "
                f"functional_floor_fps={expected_fps}; minimum_expected={expected_dispatch}; "
                f"configured_target_fps={driver.target_detector_fps_per_camera}; "
                "sustained_accepted_request_rate_is_task21"
            ),
        ),
        _check(
            "detector-and-latest-slot-high-water",
            (
                0 < driver.peak_global_detector_requests <= len(workers)
                and all(worker.peak_detector_requests == 1 for worker in workers)
                and all(worker.latest_slot_peak_occupancy == 1 for worker in workers)
                and all(worker.latest_slot_occupancy_at_stop <= 1 for worker in workers)
            ),
            (
                f"peak_global={driver.peak_global_detector_requests}; "
                f"peak_per_camera={[worker.peak_detector_requests for worker in workers]}; "
                "slot_peak="
                f"{[worker.latest_slot_peak_occupancy for worker in workers]}; "
                "slot_at_stop="
                f"{[worker.latest_slot_occupancy_at_stop for worker in workers]}"
            ),
        ),
        _check(
            "scheduler-dispatch-outcome-denominators",
            assertions.dispatch_outcomes_reconcile,
            (
                f"dispatch_turns={[worker.scheduler_dispatch_turns for worker in workers]}; "
                f"outcomes={[worker.dispatch_outcomes for worker in workers]}; "
                "detector_requested="
                f"{[worker.dispatch_detector_requested for worker in workers]}; "
                f"no_frame={[worker.dispatch_no_frame for worker in workers]}; "
                "detection_disabled="
                f"{[worker.dispatch_detection_disabled for worker in workers]}; "
                "generation_fenced="
                f"{[worker.dispatch_generation_fenced for worker in workers]}"
            ),
        ),
        _check(
            "camera-session-generation-identity-separation",
            assertions.identity_namespaces_separate,
            (f"initial={initial_namespaces}; active={active_namespaces}"),
        ),
        _check(
            "direct-real-footage-identity-switch-observation",
            assertions.direct_identity_observation_bound,
            (
                f"method={driver.direct_identity_observation.method}; "
                "identity_switches_counted="
                f"{driver.direct_identity_observation.identity_switches_counted}; "
                f"matched={driver.direct_identity_observation.matched_observations}; "
                "adjacent_opportunities="
                f"{driver.direct_identity_observation.adjacent_comparison_opportunities}; "
                "unknown_outside_denominator="
                f"{driver.direct_identity_observation.unknown_outside_denominator}; "
                f"observation_sha256={driver.direct_identity_observation.observation_sha256}"
            ),
        ),
        _check(
            "decoded-sampled-drop-counters",
            assertions.bounded_latest_slot and all(worker.dropped_frames > 0 for worker in workers),
            (
                f"observed decoded={[worker.frames_decoded for worker in workers]}; "
                f"sampled={[worker.frames_sampled for worker in workers]}; "
                f"dropped={[worker.dropped_frames for worker in workers]}"
            ),
        ),
        _check(
            "crop-handoff-and-ingress-timestamp-records",
            assertions.four_cameras_crop_handoffs and all_crops_valid and first_seen_reported,
            (
                f"observed crop_counts={[worker.crop_count for worker in workers]}; "
                "minimum_crop_dimensions="
                f"{[(worker.min_crop_width, worker.min_crop_height) for worker in workers]}; "
                f"first_seen_record_counts={[len(worker.first_seen_utc) for worker in workers]}"
            ),
        ),
        _check(
            "generation-boundary-and-stale-result-counters",
            assertions.generation_reconnect and assertions.stale_result_fenced,
            (
                f"observed reconnect_boundaries={len(driver.reconnects)}; "
                "stale_generation_results="
                f"{[worker.stale_generation_results for worker in workers]}"
            ),
        ),
        _check(
            "detection-toggle-interval-counters",
            assertions.detection_toggle_stopped
            and assertions.detection_toggle_resumed
            and driver.detection_toggle.decoded_frames_while_disabled > 0,
            (
                f"observed disabled_seconds={driver.detection_toggle.disabled_for_seconds:.3f}; "
                "detector_calls_during_disabled="
                f"{driver.detection_toggle.detector_calls_during_disabled}; "
                "decoded_frames_while_disabled="
                f"{driver.detection_toggle.decoded_frames_while_disabled}; "
                "detector_calls_after_reenable="
                f"{driver.detection_toggle.detector_calls_after_reenable}; "
                f"handoffs_after_reenable={driver.detection_toggle.handoffs_after_reenable}"
            ),
        ),
        _check(
            "pending-detector-requests-at-stop",
            assertions.single_flight_at_stop and not any(driver.cleanup_pending_requests),
            f"observed pending_detector_requests_at_stop={list(driver.cleanup_pending_requests)}",
        ),
        _check(
            "source-generation-replacement-observations",
            source_restarted and camera_two_reconnected,
            (
                "observed reconnect_boundaries="
                f"{sum(worker.reconnect_boundaries for worker in workers)}; "
                f"controller_restart={source_restarted}"
            ),
        ),
        _check(
            "delayed-inference-and-reconnect-counters",
            mode != "ingest-outage"
            or (
                driver.slow_inference.timed_out_requests > 0
                and driver.slow_inference.overlap_with_reconnect
                and assertions.slow_inference_timeout
            ),
            (
                f"observed delayed_calls={driver.slow_inference.delayed_calls}; "
                f"timed_out_requests={driver.slow_inference.timed_out_requests}; "
                f"overlap_with_reconnect={driver.slow_inference.overlap_with_reconnect}"
            ),
        ),
    )


__all__ = ["build_checks"]
