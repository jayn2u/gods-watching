from datetime import UTC, datetime
from uuid import uuid4

from gods_watching.contracts.appearances import AppearanceResponse, BoundingBox
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.verification.scenarios.task11_active_probe import unique_track_results


def _result(
    *, appearance_id: AppearanceId, session_id: CameraSessionId, track_id: int
) -> AppearanceResponse:
    seen = datetime(2026, 9, 13, 1, 0, tzinfo=UTC)
    return AppearanceResponse(
        appearance_id=appearance_id,
        camera_id=CameraId(uuid4()),
        camera_name="probe",
        session_id=session_id,
        track_id=track_id,
        first_seen=seen,
        last_seen=seen,
        ended_at=None,
        representative_version=1,
        bounding_box=BoundingBox(x_min=1, y_min=1, x_max=40, y_max=90),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.9,
        crop_quality=1.0,
        model_id="openai/clip-vit-base-patch16",
        model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
        similarity=None,
    )


def test_two_people_entering_together_are_still_unique_track_results() -> None:
    # Given: two distinct tracks started in the same frame on one camera session
    session_id = CameraSessionId(uuid4())
    observed = AppearanceId(uuid4())
    results = (
        _result(appearance_id=observed, session_id=session_id, track_id=1),
        _result(appearance_id=AppearanceId(uuid4()), session_id=session_id, track_id=2),
    )

    # When / Then: each track has exactly one appearance, so the result is unique
    assert unique_track_results(results, observed)


def test_duplicate_appearances_for_one_track_are_not_unique() -> None:
    # Given: publication produced two appearances for the same track
    session_id = CameraSessionId(uuid4())
    observed = AppearanceId(uuid4())
    results = (
        _result(appearance_id=observed, session_id=session_id, track_id=1),
        _result(appearance_id=AppearanceId(uuid4()), session_id=session_id, track_id=1),
    )

    # When / Then
    assert not unique_track_results(results, observed)


def test_observed_appearance_must_appear_exactly_once() -> None:
    # Given: the observed appearance is missing from the results
    session_id = CameraSessionId(uuid4())
    results = (_result(appearance_id=AppearanceId(uuid4()), session_id=session_id, track_id=1),)

    # When / Then
    assert not unique_track_results(results, AppearanceId(uuid4()))
