from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import TypeAdapter, ValidationError

from gods_watching.contracts.appearances import AppearanceResponse, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest, CameraResponse
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.search import SearchRequest, TextSearchRequest
from gods_watching.contracts.settings import SettingsResponse


def test_valid_camera_contract_when_rtsp_source_is_routable() -> None:
    # Given
    raw = {
        "name": "East entrance",
        "source_url": "rtsp://operator:secret@fixture-publisher:8554/east",
    }

    # When
    camera = CameraCreateRequest.model_validate(raw)

    # Then
    assert camera.source_url.host == "fixture-publisher"
    assert camera.detection_threshold == 0.5


@pytest.mark.parametrize(
    "source_url", ["file:///tmp/camera.mp4", "https://camera/live", "data:text/plain,x"]
)
def test_invalid_camera_contract_when_source_scheme_is_not_rtsp(source_url: str) -> None:
    # Given
    raw = {"name": "East entrance", "source_url": source_url}

    # When / Then
    with pytest.raises(ValidationError):
        _result = CameraCreateRequest.model_validate(raw)


@pytest.mark.parametrize("port", [0, 65536])
def test_invalid_camera_contract_when_port_is_out_of_range(port: int) -> None:
    # Given
    raw = {"name": "East entrance", "source_url": f"rtsp://camera.internal:{port}/live"}

    # When / Then
    with pytest.raises(ValidationError):
        _result = CameraCreateRequest.model_validate(raw)


def test_valid_camera_contract_when_rtsp_port_is_absent() -> None:
    # Given
    raw = {"name": "East entrance", "source_url": "rtsp://camera.internal/live"}

    # When
    camera = CameraCreateRequest.model_validate(raw)

    # Then
    assert camera.source_url.port is None


def test_valid_search_contract_when_text_is_normalized() -> None:
    # Given
    raw = {"mode": "text", "query": "  a person   with a red bag  ", "limit": 5}

    # When
    request = TextSearchRequest.model_validate(raw)

    # Then
    assert isinstance(request, TextSearchRequest)
    assert request.query == "a person with a red bag"


@pytest.mark.parametrize("query", ["   ", "검은 옷", "person 🎒", "person\nwith bag"])
def test_invalid_search_contract_when_text_is_blank_or_unsupported(query: str) -> None:
    # Given
    raw = {"mode": "text", "query": query}

    # When / Then
    with pytest.raises(ValidationError):
        TypeAdapter(SearchRequest).validate_python(raw)


def test_invalid_search_contract_when_time_range_is_reversed() -> None:
    # Given
    start = datetime(2026, 9, 6, 12, tzinfo=UTC)
    raw = {
        "mode": "browse",
        "from": start.isoformat(),
        "to": (start - timedelta(seconds=1)).isoformat(),
    }

    # When / Then
    with pytest.raises(ValidationError):
        TypeAdapter(SearchRequest).validate_python(raw)


def test_invalid_search_contract_when_identifier_is_not_uuid() -> None:
    # Given
    raw = {"mode": "similar", "appearance_id": "not-an-id"}

    # When / Then
    with pytest.raises(ValidationError):
        TypeAdapter(SearchRequest).validate_python(raw)


def test_valid_contracts_serialize_with_snake_case_and_without_rtsp_credentials() -> None:
    # Given
    now = datetime(2026, 9, 6, 12, tzinfo=UTC)
    camera_id = CameraId(uuid4())
    session_id = CameraSessionId(uuid4())
    appearance_id = AppearanceId(uuid4())
    camera = CameraResponse(
        camera_id=camera_id,
        name="East entrance",
        source_host="fixture-publisher",
        source_port=8554,
        detection_enabled=True,
        detection_threshold=0.5,
        version=1,
        deleted_at=None,
    )
    appearance = AppearanceResponse(
        appearance_id=appearance_id,
        camera_id=camera_id,
        camera_name=camera.name,
        session_id=session_id,
        track_id=7,
        first_seen=now,
        last_seen=now,
        ended_at=None,
        representative_version=1,
        bounding_box=BoundingBox(x_min=1, y_min=2, x_max=33, y_max=66),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.87,
        crop_quality=12.5,
        model_id="openai/clip-vit-base-patch16",
        model_revision="57c216476eefef5ab752ec549e440a49ae4ae5f3",
        similarity=None,
    )

    # When
    payload = {
        "camera": camera.model_dump(mode="json"),
        "appearance": appearance.model_dump(mode="json"),
    }

    # Then
    assert payload["camera"]["camera_id"] == str(camera_id)
    assert payload["appearance"]["representative_version"] == 1
    assert "source_url" not in payload["camera"]


def test_valid_settings_contract_when_four_wall_slots_are_declared() -> None:
    # Given
    camera_id = CameraId(uuid4())

    # When
    settings = SettingsResponse(
        retention_days=7, quota_bytes=100_000_000_000, wall_slot_ids=(camera_id, None, None, None)
    )

    # Then
    assert settings.wall_slot_ids[0] == camera_id
