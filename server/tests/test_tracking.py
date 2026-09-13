from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.inference.detector import Detection
from gods_watching.media.models import SourceGenerationId
from gods_watching.tracking import (
    ByteTrackScope,
    DetectionFrame,
    DetectorInputReference,
    LifecycleKind,
    ResetReason,
    StaleFrameError,
    TrackingScope,
    TrackLifecycle,
)


def _scope(
    *,
    camera_id: CameraId | None = None,
    session_id: CameraSessionId | None = None,
    generation_id: SourceGenerationId | None = None,
    threshold: float = 0.5,
) -> ByteTrackScope:
    return ByteTrackScope(
        scope=TrackingScope(
            camera_id=camera_id or CameraId(uuid4()),
            session_id=session_id or CameraSessionId(uuid4()),
            source_generation_id=generation_id or SourceGenerationId(uuid4()),
            detection_enabled=True,
        ),
        threshold=threshold,
    )


def _frame(
    scope: ByteTrackScope,
    *,
    seconds: float,
    detector_result_seconds: float | None = None,
    reference: str,
    detections: tuple[tuple[float, float, float, float, float], ...] = (),
) -> DetectionFrame:
    current = scope.scope
    timestamp = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
    return DetectionFrame(
        camera_id=current.camera_id,
        session_id=current.session_id,
        source_generation_id=current.source_generation_id,
        ingress_utc=timestamp,
        ingress_monotonic=seconds,
        detector_result_monotonic=(
            seconds if detector_result_seconds is None else detector_result_seconds
        ),
        source_width=640,
        source_height=480,
        detector_input_reference=DetectorInputReference(reference),
        detections=tuple(
            Detection(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=confidence,
                class_id=0,
            )
            for x1, y1, x2, y2, confidence in detections
        ),
    )


def _events(scope: ByteTrackScope, *frames: DetectionFrame) -> list[TrackLifecycle]:
    events: list[TrackLifecycle] = []
    for frame in frames:
        events.extend(scope.feed(frame))
    return events


def test_fixed_bytetrack_parameters_and_threshold_compensation() -> None:
    # Given: a camera threshold from the approved 0.1..0.95 settings range
    scope = _scope(threshold=0.5)

    # When: a camera-local tracker scope is created
    parameters = scope.parameters

    # Then: the adapter records the exact pinned ByteTrack configuration
    assert parameters.frame_rate == 5
    assert parameters.lost_track_buffer == 60
    assert parameters.minimum_matching_threshold == 0.8
    assert parameters.minimum_consecutive_frames == 1
    assert abs(parameters.track_activation_threshold - 0.4) < 1e-9
    assert scope.library_max_time_lost == 10


def test_moving_person_publishes_one_track_with_ingress_timestamps() -> None:
    # Given: one person moves no more than ten pixels per sampled frame
    scope = _scope()
    frames = (
        _frame(scope, seconds=0.0, reference="frame-0", detections=((100, 100, 180, 260, 0.9),)),
        _frame(scope, seconds=0.2, reference="frame-1", detections=((105, 100, 185, 260, 0.9),)),
        _frame(scope, seconds=0.4, reference="frame-2", detections=((110, 100, 190, 260, 0.9),)),
    )

    # When: all sampled detector results enter one camera/session scope
    events = _events(scope, *frames)

    # Then: one appearance starts once and receives two updates with its original candidate
    assert [event.kind for event in events] == [
        LifecycleKind.START,
        LifecycleKind.UPDATE,
        LifecycleKind.UPDATE,
    ]
    assert len({event.track_key for event in events}) == 1
    assert events[0].first_seen == frames[0].ingress_utc
    assert events[-1].last_seen == frames[-1].ingress_utc
    assert events[0].t_detect_monotonic == 0.0
    assert events[0].first_candidate.detector_input_reference == DetectorInputReference("frame-0")


def test_first_candidate_preserves_result_receipt_through_confirmation() -> None:
    # Given: an eligible first result arrives after the frame ingress timestamp
    scope = _scope(threshold=0.8)
    first = _frame(
        scope,
        seconds=0.0,
        detector_result_seconds=0.35,
        reference="candidate-0",
        detections=((100, 100, 180, 260, 0.8),),
    )
    second = _frame(
        scope,
        seconds=0.2,
        detector_result_seconds=0.55,
        reference="candidate-1",
        detections=((104, 100, 184, 260, 0.9),),
    )

    # When: the candidate is followed by a detector result that activates a track
    events = _events(scope, first, second)

    # Then: confirmation emits one start while freshness and crop handoff retain frame zero
    starts = [event for event in events if event.kind is LifecycleKind.START]
    assert len(starts) == 1
    assert starts[0].first_seen == first.ingress_utc
    assert starts[0].t_detect_monotonic == first.detector_result_monotonic
    assert starts[0].first_candidate.ingress_monotonic == first.ingress_monotonic
    assert starts[0].first_candidate.detector_result_monotonic == 0.35
    assert starts[0].first_candidate.detector_input_reference == DetectorInputReference(
        "candidate-0"
    )


def test_below_camera_threshold_never_becomes_pending_first_candidate() -> None:
    # Given: a detector result below the camera's indexing threshold
    scope = _scope(threshold=0.8)
    low = _frame(
        scope,
        seconds=0.0,
        reference="below-threshold",
        detections=((100, 100, 180, 260, 0.75),),
    )

    # When: the low result is followed by enough eligible results to confirm a track
    assert scope.feed(low) == ()
    assert scope.pending_candidate_count == 0
    high_first = _frame(
        scope,
        seconds=0.2,
        reference="eligible-first",
        detections=((104, 100, 184, 260, 0.9),),
    )
    high_second = _frame(
        scope,
        seconds=0.4,
        reference="eligible-second",
        detections=((108, 100, 188, 260, 0.9),),
    )
    events = [*scope.feed(high_first), *scope.feed(high_second)]

    # Then: publication begins from an eligible candidate, never the low result
    assert [event.kind for event in events] == [LifecycleKind.START]
    assert events[0].first_candidate.detector_input_reference == DetectorInputReference(
        "eligible-first"
    )


def test_empty_detections_do_not_end_track_before_elapsed_expiry() -> None:
    # Given: a confirmed track followed by an empty sampled detector result
    scope = _scope()
    start = _frame(scope, seconds=0.0, reference="start", detections=((10, 20, 90, 220, 0.9),))
    empty = _frame(scope, seconds=0.2, reference="empty")
    _ = _events(scope, start, empty)

    # When: time is advanced first below and then above the two second elapsed fence
    before_expiry = scope.expire(_frame(scope, seconds=1.9, reference="clock"))
    after_expiry = scope.expire(_frame(scope, seconds=2.1, reference="clock-2"))

    # Then: the track remains active before the fence and ends once after it
    assert before_expiry == ()
    assert [event.kind for event in after_expiry] == [LifecycleKind.END]
    assert after_expiry[0].ended_at == datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=2.1)


def test_one_second_gap_keeps_id_but_long_gap_starts_new_id() -> None:
    # Given: one camera has an overlapping detection after a one second gap
    scope = _scope()
    first = _frame(scope, seconds=0.0, reference="first", detections=((100, 100, 180, 260, 0.9),))
    retained = _frame(
        scope, seconds=1.0, reference="retained", detections=((104, 100, 184, 260, 0.9),)
    )
    retained_events = _events(scope, first, retained)
    retained_key = retained_events[-1].track_key

    # When: a later detector result arrives after the strict two second elapsed fence
    reset_events = _events(
        scope,
        _frame(scope, seconds=3.1, reference="return", detections=((108, 100, 188, 260, 0.9),)),
    )

    # Then: the first key receives the end event and the returning person gets a new key
    assert retained_events[0].track_key == retained_key
    assert retained_events[1].track_key == retained_key
    assert reset_events[0].kind is LifecycleKind.END
    assert reset_events[-1].kind is LifecycleKind.START
    assert reset_events[-1].track_key != retained_key


def test_source_generation_and_detection_toggle_reset_scope() -> None:
    # Given: one track in a source generation
    camera_id = CameraId(uuid4())
    session_id = CameraSessionId(uuid4())
    generation_id = SourceGenerationId(uuid4())
    next_generation = SourceGenerationId(uuid4())
    scope = _scope(camera_id=camera_id, session_id=session_id, generation_id=generation_id)
    first = _frame(scope, seconds=0.0, reference="first", detections=((10, 10, 80, 180, 0.9),))
    first_events = scope.feed(first)

    # When: detection is disabled and a new source generation is then enabled
    disabled_scope = TrackingScope(
        camera_id=camera_id,
        session_id=session_id,
        source_generation_id=generation_id,
        detection_enabled=False,
    )
    disabled_end = scope.reset(
        reason=ResetReason.DETECTION_DISABLED,
        boundary=replace(_frame(scope, seconds=0.2, reference="disabled"), detection_enabled=False),
        replacement_scope=disabled_scope,
    )
    next_session = CameraSessionId(uuid4())
    next_scope = TrackingScope(
        camera_id=camera_id,
        session_id=next_session,
        source_generation_id=next_generation,
        detection_enabled=True,
    )
    source_end = scope.reset(
        reason=ResetReason.SOURCE_GENERATION_CHANGED,
        boundary=replace(
            _frame(scope, seconds=0.4, reference="generation"), detection_enabled=False
        ),
        replacement_scope=next_scope,
    )
    second = _frame(
        scope,
        seconds=0.6,
        reference="second",
        detections=((10, 10, 80, 180, 0.9),),
    )
    second_events = scope.feed(second)

    # Then: each reset fences and closes prior state, and the new generation starts clean
    assert first_events[0].kind is LifecycleKind.START
    assert disabled_end[0].kind is LifecycleKind.END
    assert source_end == ()
    assert second_events[0].kind is LifecycleKind.START
    assert second_events[0].track_key != first_events[0].track_key
    assert second_events[0].source_generation_id == next_generation


def test_equal_local_tracker_numbers_are_scoped_by_camera_and_session() -> None:
    # Given: independent scopes that both start their library local tracker at one
    first = _scope()
    second = _scope()

    # When: identical detections are processed independently
    first_event = first.feed(
        _frame(first, seconds=0.0, reference="camera-a", detections=((10, 10, 80, 180, 0.9),))
    )[0]
    second_event = second.feed(
        _frame(second, seconds=0.0, reference="camera-b", detections=((10, 10, 80, 180, 0.9),))
    )[0]

    # Then: local numbers may match while typed appearance keys remain distinct
    assert first_event.track_key.local_track_id == second_event.track_key.local_track_id
    assert first_event.track_key != second_event.track_key


def test_stale_generation_and_non_monotonic_frames_are_rejected() -> None:
    # Given: a scope that has accepted one ingress frame
    scope = _scope()
    _ = scope.feed(_frame(scope, seconds=1.0, reference="current"))

    # When/Then: old monotonic and old source-generation frames cannot cross the fence
    with pytest.raises(StaleFrameError):
        _ = scope.feed(_frame(scope, seconds=0.5, reference="old"))
    old_generation = SourceGenerationId(uuid4())
    stale = DetectionFrame(
        camera_id=scope.scope.camera_id,
        session_id=scope.scope.session_id,
        source_generation_id=old_generation,
        ingress_utc=datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
        ingress_monotonic=2.0,
        detector_result_monotonic=2.0,
        source_width=640,
        source_height=480,
        detector_input_reference=DetectorInputReference("old-generation"),
        detections=(),
    )
    with pytest.raises(StaleFrameError):
        _ = scope.feed(stale)


def test_separable_crossing_keeps_two_camera_local_tracks() -> None:
    # Given: two people move through each other on separate vertical lanes
    scope = _scope()
    frames = tuple(
        _frame(
            scope,
            seconds=index * 0.2,
            reference=f"crossing-{index}",
            detections=(
                (10 + index * 8, 20, 70 + index * 8, 180, 0.9),
                (170 - index * 8, 280, 230 - index * 8, 440, 0.9),
            ),
        )
        for index in range(4)
    )

    # When: the separable trajectories are fed to one camera/session scope
    events = _events(scope, *frames)

    # Then: two starts remain separate without making an ambiguous overlap promise
    starts = [event for event in events if event.kind is LifecycleKind.START]
    assert len(starts) == 2
    assert len({event.track_key for event in starts}) == 2


def test_threshold_rejects_invalid_scope_configuration() -> None:
    # Given/When/Then: camera thresholds outside the detector contract fail at scope construction
    with pytest.raises(ValueError, match="threshold"):
        _ = _scope(threshold=0.0)
    with pytest.raises(ValueError, match="threshold"):
        _ = _scope(threshold=1.0)
