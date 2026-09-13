"""Structured evidence emitted by the installed Task 10 runtime driver."""

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field


class _EvidenceModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid", frozen=True)


class IdentityObservation(_EvidenceModel):
    """Record a real-footage track-fragmentation proxy without identity claims."""

    method: Literal["track-lifecycle-fragmentation-proxy"]
    real_identity_status: Literal["unknown"]
    evaluated_detections: int = Field(ge=0)
    ambiguous_exclusions: int = Field(ge=0)
    observed_transitions: int = Field(ge=0)


class DirectIdentityObservation(_EvidenceModel):
    """Bind a human-labeled real-footage count to current inputs and sources."""

    method: Literal["human-labeled-retained-frame-review"]
    identity_switches_counted: int = Field(ge=0)
    matched_observations: int = Field(gt=0)
    adjacent_comparison_opportunities: int = Field(gt=0)
    reviewed_clips: int = Field(gt=0)
    reviewed_sample_frames: int = Field(gt=0)
    unknown_outside_denominator: bool
    observation_sha256: str
    raw_observations_sha256: str
    production_source_hashes: dict[str, str]
    all_source_hashes_current: bool
    all_fixture_hashes_current: bool


class WorkerEvidence(_EvidenceModel):
    """Capture bounded decode, detector, crop, and scheduling observations."""

    camera_index: int = Field(ge=1, le=4)
    camera_id: str
    initial_camera_session_id: str
    initial_source_generation_id: str
    active_camera_session_id: str
    active_source_generation_id: str
    frames_decoded: int = Field(ge=0)
    frames_sampled: int = Field(ge=0)
    detector_requests: int = Field(ge=0)
    detector_results: int = Field(ge=0)
    positive_detector_results: int = Field(ge=0)
    detector_detections: int = Field(ge=0)
    invalid_detections: int = Field(ge=0)
    eligible_detections: int = Field(ge=0)
    dropped_frames: int = Field(ge=0)
    pending_detector_requests_at_stop: int = Field(ge=0)
    peak_detector_requests: int = Field(ge=0)
    latest_slot_occupancy_at_stop: int = Field(ge=0, le=1)
    latest_slot_peak_occupancy: int = Field(ge=0, le=1)
    frame_age_seconds_at_stop: float | None = Field(default=None, ge=0.0)
    sanitized_errors: int = Field(ge=0)
    last_sanitized_error: str | None = None
    handoffs: dict[str, int]
    crop_count: int = Field(ge=0)
    min_crop_width: int = Field(ge=0)
    min_crop_height: int = Field(ge=0)
    first_seen_utc: tuple[str, ...]
    identity_switch_observations: int | None = Field(default=None, ge=0)
    identity_observation: IdentityObservation
    reconnect_boundaries: int = Field(ge=0)
    scheduler_dispatch_turns: int = Field(ge=0)
    eligible_scheduler_dispatch_turns: int = Field(ge=0)
    detector_latency_max_seconds: float | None = Field(default=None, ge=0.0)
    stale_generation_results: int = Field(default=0, ge=0)
    dispatch_detector_requested: int = Field(ge=0)
    dispatch_no_frame: int = Field(ge=0)
    dispatch_detection_disabled: int = Field(ge=0)
    dispatch_generation_fenced: int = Field(ge=0)
    dispatch_closed: int = Field(ge=0)
    dispatch_outcomes: int = Field(ge=0)


class ToggleEvidence(_EvidenceModel):
    """Capture detector disable and re-enable observations for one camera."""

    disabled_for_seconds: float = Field(ge=0.0)
    detector_calls_before: int = Field(ge=0)
    detector_calls_during_disabled: int = Field(ge=0)
    detector_calls_after_reenable: int = Field(ge=0)
    end_events_on_disable: tuple[str, ...]
    decoded_frames_while_disabled: int = Field(ge=0)
    handoffs_after_reenable: int = Field(ge=0)


class ReconnectEvidence(_EvidenceModel):
    """Capture the old and replacement identities at one source boundary."""

    camera_index: int = Field(ge=1, le=4)
    camera_id: str
    elapsed_seconds: float = Field(ge=0.0)
    previous_source_generation_id: str
    replacement_source_generation_id: str
    previous_camera_session_id: str
    replacement_camera_session_id: str


class SlowInferenceEvidence(_EvidenceModel):
    """Capture delayed detector requests and their overlap with reconnects."""

    enabled: bool
    delayed_calls: int = Field(ge=0)
    timed_out_requests: int = Field(ge=0)
    max_delay_seconds: float = Field(ge=0.0)
    overlap_with_reconnect: bool
    signal_camera_index: int | None = Field(default=None, ge=1, le=4)


class DriverAssertions(_EvidenceModel):
    """Record the machine assertions made by the real driver."""

    four_cameras_decoded: bool
    four_cameras_real_detector_results: bool
    four_cameras_person_detections: bool
    four_cameras_crop_handoffs: bool
    bounded_latest_slot: bool
    single_flight_at_stop: bool
    fair_scheduler_turns: bool
    detection_toggle_stopped: bool
    detection_toggle_resumed: bool
    generation_reconnect: bool
    stale_result_fenced: bool
    slow_inference_timeout: bool
    dispatch_outcomes_reconcile: bool
    identity_namespaces_separate: bool
    direct_identity_observation_bound: bool


class DriverEvidence(_EvidenceModel):
    """Persist all structured observations emitted by one real Task 10 run."""

    schema_version: int
    mode: Literal["ingest", "ingest-outage"]
    label: Literal["REAL: PyAV H.264 RTSP/TCP + Triton YOLO11s GPU + supervision ByteTrack"]
    elapsed_seconds: float = Field(gt=0.0)
    target_detector_fps_per_camera: float = Field(gt=0.0)
    peak_global_detector_requests: int = Field(ge=0)
    workers: tuple[WorkerEvidence, ...]
    direct_identity_observation: DirectIdentityObservation
    detection_toggle: ToggleEvidence
    reconnects: tuple[ReconnectEvidence, ...]
    slow_inference: SlowInferenceEvidence
    assertions: DriverAssertions
    cleanup_pending_requests: tuple[int, ...]


class FixtureStreamProvenance(_EvidenceModel):
    """Identify one prepared fixture stream and its verified content hash."""

    stream_id: str
    prepared_path: str
    sha256: str
    bytes: int = Field(ge=0)
    codec: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class FixtureProvenance(_EvidenceModel):
    """Describe the exact fixture manifest, model assets, and stream inputs."""

    manifest: str
    manifest_sha256: str
    prepared_inputs_redistributable: bool
    streams: tuple[FixtureStreamProvenance, ...]
    yolo_weights_sha256: str
    prepared_model_manifest_sha256: str


class SourceControlEvidence(_EvidenceModel):
    """Record the owned outage controller's stop and restart results."""

    started_utc: str
    waited_seconds: float = Field(ge=0.0)
    slow_signal_seen: bool
    stop_exit_code: int
    restart_exit_code: int
    source_service: Literal["fixture-camera-2"]


class CleanupEvidence(_EvidenceModel):
    """Record owned resource cleanup commands and remaining-resource output."""

    triton_remove_exit_code: int | None = None
    compose_down_exit_code: int | None = None
    inspect_return_code: int | None = None
    remaining_owned_resources: str = ""
