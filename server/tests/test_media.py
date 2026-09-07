from pathlib import Path
from uuid import UUID, uuid4

import anyio
import pytest
from anyio.lowlevel import checkpoint

from gods_watching.contracts.identifiers import CameraId
from gods_watching.media import (
    AuthorizedMediaSession,
    GatewayResponse,
    GenerationEvent,
    GenerationEventKind,
    InvalidGatewayLocationError,
    MediaControlAdapter,
    MediaPath,
    MediaSessionId,
    SourceGenerationCoordinator,
    WhepProxyService,
    WhepResourceId,
)
from gods_watching.verification.context import (
    ResourceLedger,
    ScenarioContext,
    ScenarioContextConfig,
)
from gods_watching.verification.models import RunId
from gods_watching.verification.scenarios.task09_runtime import (
    finalize_publisher,
    prepare_runtime,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


class RecordingGateway:
    def __init__(
        self,
        *,
        location: str = "/camera-1/whep/00000000-0000-0000-0000-000000000001",
    ) -> None:
        self.location: str = location
        self.posts: list[tuple[MediaPath, bytes]] = []
        self.patches: list[tuple[str, bytes]] = []
        self.deletes: list[str] = []

    async def create(self, path: MediaPath, offer: bytes) -> GatewayResponse:
        self.posts.append((path, offer))
        return GatewayResponse(
            status_code=201,
            body=b"answer",
            content_type="application/sdp",
            location=self.location,
            etag="*",
        )

    async def patch(self, location: str, fragment: bytes) -> GatewayResponse:
        self.patches.append((location, fragment))
        return GatewayResponse(status_code=204, body=b"")

    async def delete(self, location: str) -> GatewayResponse:
        self.deletes.append(location)
        return GatewayResponse(status_code=200, body=b"")


class RecordingEventSink:
    def __init__(self) -> None:
        self.events: list[GenerationEvent] = []

    async def publish(self, event: GenerationEvent) -> None:
        self.events.append(event)


class BlockingEventSink:
    def __init__(self) -> None:
        self.events: list[GenerationEvent] = []
        self.end_started: anyio.Event = anyio.Event()
        self.release_end: anyio.Event = anyio.Event()
        self.block_next_end: bool = False

    async def publish(self, event: GenerationEvent) -> None:
        self.events.append(event)
        if self.block_next_end and event.kind is GenerationEventKind.ENDED:
            self.block_next_end = False
            self.end_started.set()
            await self.release_end.wait()


class RecordingControlGateway:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def upsert_path(self, path: MediaPath, source_url: str) -> None:
        _ = path
        self.operations.append(("upsert", source_url))

    async def delete_path(self, path: MediaPath) -> None:
        _ = path
        self.operations.append(("delete", ""))


def _service(gateway: RecordingGateway) -> WhepProxyService:
    return WhepProxyService(
        gateway=gateway,
        resolve_path=lambda _camera_id: MediaPath("camera-1"),
    )


def test_whep_post_rewrites_and_hides_upstream_resource_location() -> None:
    # Given: an authorized application session and a credentialed private gateway
    gateway = RecordingGateway()
    service = _service(gateway)
    camera_id = CameraId(uuid4())
    session = AuthorizedMediaSession(session_id=MediaSessionId("operator-session"))

    # When: the browser offer is proxied to MediaMTX
    response = anyio.run(service.create, camera_id, session, b"offer")

    # Then: the client receives only an application-owned opaque resource URL
    assert response.status_code == 201
    assert response.location is not None
    assert response.location.startswith(f"/api/live/{camera_id}/whep/")
    assert "00000000-0000-0000-0000-000000000001" not in response.location
    assert gateway.posts == [(MediaPath("camera-1"), b"offer")]


@pytest.mark.parametrize(
    "location",
    [
        "https://attacker.invalid/camera-1/whep/00000000-0000-0000-0000-000000000001",
        "/camera-2/whep/00000000-0000-0000-0000-000000000001",
        "/camera-1/whep/not-a-resource",
        "/camera-1/whep/00000000-0000-0000-0000-000000000001?jwt=secret",
    ],
)
def test_whep_post_rejects_untrusted_gateway_location(location: str) -> None:
    # Given: a gateway response that tries to escape the expected WHEP resource path
    service = _service(RecordingGateway(location=location))
    session = AuthorizedMediaSession(session_id=MediaSessionId("operator-session"))

    # When / Then: parsing fails closed before a browser resource is registered
    with pytest.raises(InvalidGatewayLocationError):
        _ = anyio.run(service.create, CameraId(uuid4()), session, b"offer")


def test_whep_resource_is_owned_by_creating_session() -> None:
    # Given: one WHEP resource created by the first authorized session
    gateway = RecordingGateway()
    service = _service(gateway)
    camera_id = CameraId(uuid4())
    first = AuthorizedMediaSession(session_id=MediaSessionId("first"))
    second = AuthorizedMediaSession(session_id=MediaSessionId("second"))
    created = anyio.run(service.create, camera_id, first, b"offer")
    assert created.location is not None
    resource_id = WhepResourceId(UUID(created.location.rsplit("/", 1)[1]))

    # When: a different session tries to trickle an ICE candidate
    denied = anyio.run(service.patch, camera_id, second, resource_id, b"candidate")

    # Then: existence is hidden and nothing reaches the private gateway
    assert denied.status_code == 404
    assert gateway.patches == []


def test_whep_delete_and_session_teardown_forward_to_gateway_once() -> None:
    # Given: two resources owned by one application session
    gateway = RecordingGateway()
    service = _service(gateway)
    camera_id = CameraId(uuid4())
    session = AuthorizedMediaSession(session_id=MediaSessionId("operator-session"))
    first = anyio.run(service.create, camera_id, session, b"offer-1")
    second = anyio.run(service.create, camera_id, session, b"offer-2")
    assert first.location is not None
    first_id = WhepResourceId(UUID(first.location.rsplit("/", 1)[1]))

    # When: the browser deletes one and session cleanup owns the abandoned other
    deleted = anyio.run(service.delete, camera_id, session, first_id)
    closed = anyio.run(service.close_session, session.session_id)

    # Then: each upstream resource is deleted exactly once
    assert deleted.status_code == 200
    assert closed == 1
    assert len(gateway.deletes) == 2
    assert first.location != second.location


def test_source_loss_emits_loss_then_ends_generation() -> None:
    # Given: an active camera source generation
    sink = RecordingEventSink()
    coordinator = SourceGenerationCoordinator(sink=sink)
    camera_id = CameraId(uuid4())
    generation_id = anyio.run(coordinator.begin, camera_id)

    # When: the media source becomes unavailable
    ended = anyio.run(coordinator.source_lost, camera_id)

    # Then: downstream supervision sees loss and terminal cancellation in order
    assert ended == generation_id
    assert [event.kind for event in sink.events] == [
        GenerationEventKind.STARTED,
        GenerationEventKind.SOURCE_LOST,
        GenerationEventKind.ENDED,
    ]
    assert {event.generation_id for event in sink.events} == {generation_id}


def test_activation_cancels_old_generation_before_concurrent_replacement() -> None:
    async def exercise() -> None:
        # Given: an active source and a sink that pauses after old-work cancellation
        sink = BlockingEventSink()
        gateway = RecordingControlGateway()
        adapter = MediaControlAdapter(
            gateway=gateway,
            generations=SourceGenerationCoordinator(sink=sink),
        )
        camera_id = CameraId(uuid4())
        _ = await adapter.activate(camera_id, "rtsp://old/source")
        sink.block_next_end = True

        # When: replacements race while the first one is still cancelling old work
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(adapter.activate, camera_id, "rtsp://new/source")
            await sink.end_started.wait()
            task_group.start_soon(adapter.activate, camera_id, "rtsp://newer/source")
            await checkpoint()
            assert gateway.operations == [("upsert", "rtsp://old/source")]
            sink.release_end.set()

        # Then: each replacement is applied only after its predecessor is ended
        assert gateway.operations == [
            ("upsert", "rtsp://old/source"),
            ("upsert", "rtsp://new/source"),
            ("upsert", "rtsp://newer/source"),
        ]
        assert [event.kind for event in sink.events] == [
            GenerationEventKind.STARTED,
            GenerationEventKind.ENDED,
            GenerationEventKind.STARTED,
            GenerationEventKind.ENDED,
            GenerationEventKind.STARTED,
        ]

    anyio.run(exercise)


def test_generation_events_serialize_replacement_and_source_loss() -> None:
    async def exercise() -> None:
        # Given: a running generation and a sink paused during its terminal event
        sink = BlockingEventSink()
        coordinator = SourceGenerationCoordinator(sink=sink)
        camera_id = CameraId(uuid4())
        _ = await coordinator.begin(camera_id)
        sink.block_next_end = True

        # When: source replacement and source loss arrive at the same boundary
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(coordinator.begin, camera_id)
            await sink.end_started.wait()
            task_group.start_soon(coordinator.source_lost, camera_id)
            await checkpoint()
            assert [event.kind for event in sink.events] == [
                GenerationEventKind.STARTED,
                GenerationEventKind.ENDED,
            ]
            sink.release_end.set()

        # Then: the replacement starts before the later loss can end it
        assert [event.kind for event in sink.events] == [
            GenerationEventKind.STARTED,
            GenerationEventKind.ENDED,
            GenerationEventKind.STARTED,
            GenerationEventKind.SOURCE_LOST,
            GenerationEventKind.ENDED,
        ]

    anyio.run(exercise)


def test_mediamtx_template_disables_recording_hls_and_playback() -> None:
    # Given: the committed production MediaMTX template
    config_path = REPOSITORY_ROOT / "deploy/mediamtx.yml"

    # When: its explicit service and storage switches are inspected
    config = config_path.read_text(encoding="utf-8")

    # Then: only RTSP/TCP and WHEP transport are enabled
    assert "rtspTransports: [tcp]" in config
    assert "hls: no" in config
    assert "playback: no" in config
    assert "record: no" in config
    assert "webrtcICEServers2: []" in config
    assert "webrtcAdditionalHosts: []" in config
    assert "authInternalUsers:" in config
    assert "user: any" not in config


def test_media_publishers_are_finite_processes_without_seamless_loop(tmp_path: Path) -> None:
    context = ScenarioContext(
        ScenarioContextConfig(
            run_id=RunId("publisher-contract"),
            run_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            compose_project="publisher-contract",
            allocated_port=30_000,
        ),
        ledger=ResourceLedger(tmp_path / "resource-manifest.json"),
    )
    context.runtime_root.mkdir()
    _ = prepare_runtime(context, "test-publisher/camera-1")

    publishers = finalize_publisher(context, 30_001)
    first = publishers.first.read_text(encoding="utf-8")
    second = publishers.second.read_text(encoding="utf-8")

    assert "-stream_loop" not in first
    assert "-stream_loop" not in second
    assert " -t 1 " in first
    assert " -t 1 " not in second
