"""Real database and RTSP-driver coverage for Task 12b services."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import anyio
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from gods_watching.cameras import (
    CameraDeletedError,
    CameraRepository,
    CameraService,
    ParsedRtspSource,
    ProbeFailureCode,
    ProbeResult,
    RtspProbeError,
    RtspSourceProbe,
    StaleCameraVersionError,
    parse_rtsp_source,
)
from gods_watching.contracts.appearances import AppearancePublication, BoundingBox
from gods_watching.contracts.cameras import CameraCreateRequest, CameraPatchRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId
from gods_watching.contracts.settings import SettingsPatchRequest
from gods_watching.fixtures.publisher import build_publisher_command
from gods_watching.settings import SettingsService
from gods_watching.storage import (
    Appearance,
    CredentialCipher,
    StorageRepository,
)
from gods_watching.verification.context import (
    ResourceLedger,
    ScenarioContext,
    ScenarioContextConfig,
    allocate_loopback_port,
)
from gods_watching.verification.models import RunId
from gods_watching.verification.scenarios.task09_runtime import (
    READER_USER,
    GatewayPorts,
    RuntimeSecrets,
    gateway_command,
    prepare_runtime,
    wait_for_gateway,
)


@dataclass(frozen=True, slots=True)
class _FailingProbe:
    """Return a typed connection failure without touching the network."""

    async def probe(self, source: ParsedRtspSource) -> ProbeResult:
        del source
        raise RtspProbeError(
            code=ProbeFailureCode.UNREACHABLE,
            detail="RTSP source could not be reached",
        )


def _service() -> CameraService:
    cipher = CredentialCipher(Fernet.generate_key())
    storage = StorageRepository(cipher)
    return CameraService(CameraRepository(storage))


def _request(
    name: str,
    source: str = "rtsp://operator:secret@fixture:8554/lobby",
) -> CameraCreateRequest:
    return CameraCreateRequest.model_validate({"name": name, "source_url": source})


def _patch_source(source: str) -> CameraPatchRequest:
    return CameraPatchRequest.model_validate({"source_url": source})


@asynccontextmanager
async def _media_gateway(context: ScenarioContext) -> AsyncIterator[None]:
    """Own the exact MediaMTX container and remove it after the process exits."""
    container_name = f"{context.compose_project}-media"
    try:
        async with context.process(name="media-gateway", command=gateway_command(context)):
            yield
    finally:
        with anyio.CancelScope(shield=True):
            _ = await anyio.run_process(
                ("/usr/bin/docker", "container", "stop", "--timeout", "2", container_name),
                check=False,
            )
            _ = await anyio.run_process(
                ("/usr/bin/docker", "container", "remove", container_name),
                check=False,
            )


async def _wait_for_path(
    ports: GatewayPorts,
    media_path: str,
    secrets: RuntimeSecrets,
    *,
    ready: bool,
) -> None:
    """Observe one authenticated fixture path becoming ready or not-ready."""
    source = f"rtsp://{READER_USER}:{secrets.reader_password}@127.0.0.1:{ports.rtsp}/{media_path}"
    with anyio.fail_after(10):
        while True:
            completed = await anyio.run_process(
                (
                    "/usr/bin/ffprobe",
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-show_entries",
                    "stream=codec_name",
                    source,
                ),
                check=False,
            )
            if (completed.returncode == 0) is ready:
                return
            await anyio.sleep(0.05)


@pytest.mark.anyio
async def test_camera_crud_source_generation_and_name_edit(session: AsyncSession) -> None:
    # Given: an empty real PostgreSQL transaction and a durable camera service
    service = _service()
    created = await service.create(session, _request("Lobby"))
    first_session = created.lifecycle.activation
    assert first_session is not None

    # When: a name-only edit and then a source edit are committed through the service
    renamed = await service.update(
        session,
        created.camera.camera_id,
        CameraPatchRequest(name="Lobby renamed"),
        expected_version=created.camera.version,
    )
    edited = await service.update(
        session,
        created.camera.camera_id,
        _patch_source("rtsp://operator:new-secret@fixture:8554/yard"),
        expected_version=renamed.camera.version,
    )

    # Then: name-only state keeps its generation, while source state fences old work
    assert renamed.lifecycle.activation is None
    assert renamed.lifecycle.cancellation is None
    assert edited.lifecycle.cancellation is not None
    assert edited.lifecycle.activation is not None
    assert edited.lifecycle.cancellation.session_id == first_session.session_id
    assert edited.lifecycle.activation.session_id != first_session.session_id
    public = await service.get(session, created.camera.camera_id)
    assert public.name == "Lobby renamed"
    assert public.source_host == "fixture"
    assert public.source_port == 8554


@pytest.mark.anyio
async def test_invalid_source_edit_preserves_working_source(session: AsyncSession) -> None:
    # Given: a committed camera and a probe that reports an unreachable replacement
    service = _service()
    created = await service.create(session, _request("Stable"))
    failing = CameraService(service.repository, source_probe=_FailingProbe())

    # When / Then: probe failure happens before any camera row or generation mutation
    with pytest.raises(RtspProbeError):
        _ = await failing.update(
            session,
            created.camera.camera_id,
            _patch_source("rtsp://operator:bad@offline:8554/live"),
            expected_version=created.camera.version,
        )
    public = await service.get(session, created.camera.camera_id)
    assert public.version == created.camera.version
    assert public.source_host == "fixture"
    assert public.source_port == 8554


@pytest.mark.anyio
async def test_soft_delete_keeps_appearance_label_and_live_history(session: AsyncSession) -> None:
    # Given: one camera session with a real appearance row in PostgreSQL
    service = _service()
    created = await service.create(session, _request("Archived lobby"))
    activation = created.lifecycle.activation
    assert activation is not None
    observed_at = datetime.now(UTC)
    publication = AppearancePublication(
        appearance_id=AppearanceId(uuid4()),
        camera_id=activation.camera_id,
        session_id=activation.session_id,
        track_id=1,
        first_seen=observed_at,
        last_seen=observed_at,
        ended_at=None,
        representative_version=1,
        crop_object_key="aa/bb/00000000-0000-4000-8000-000000000010.jpg",
        bounding_box=BoundingBox(x_min=1, y_min=2, x_max=100, y_max=200),
        source_width=1920,
        source_height=1080,
        detector_confidence=0.9,
        crop_quality=1.0,
        byte_size=10,
        embedded_at=observed_at,
        model_id="test-model",
        model_revision="test-revision",
        embedding=tuple(1.0 if index == 0 else 0.0 for index in range(512)),
    )
    _ = await service.repository.storage.publish_appearance(session, publication)

    # When: the camera is removed with its current version fence
    removed = await service.delete(
        session,
        created.camera.camera_id,
        expected_version=created.camera.version,
    )

    # Then: the source is disabled and tombstoned from camera listings, while history remains
    assert removed.camera.deleted_at is not None
    assert removed.camera.detection_enabled is False
    with pytest.raises(CameraDeletedError):
        _ = await service.get(session, created.camera.camera_id)
    archived = await service.get(session, created.camera.camera_id, include_deleted=True)
    assert archived.name == "Archived lobby"
    persisted = await session.scalar(
        select(Appearance).where(Appearance.id == publication.appearance_id)
    )
    assert persisted is not None
    assert persisted.tombstoned_at is None


@pytest.mark.anyio
async def test_optimistic_version_race_allows_one_writer(engine: AsyncEngine) -> None:
    # Given: one committed camera and two independent real database sessions
    service = _service()
    factory = async_sessionmaker(engine, expire_on_commit=False)
    name = f"race-{uuid4()}"
    async with factory.begin() as seed:
        created = await service.create(seed, _request(name))
    outcomes: list[str] = []

    async def attempt(new_name: str) -> None:
        async with factory() as child:
            try:
                async with child.begin():
                    _ = await service.update(
                        child,
                        created.camera.camera_id,
                        CameraPatchRequest(name=new_name),
                        expected_version=created.camera.version,
                    )
                outcomes.append("committed")
            except StaleCameraVersionError:
                outcomes.append("stale")

    # When: both writers race on the same optimistic version
    async with anyio.create_task_group() as group:
        group.start_soon(attempt, f"{name}-a")
        group.start_soon(attempt, f"{name}-b")

    # Then: PostgreSQL row locking and the service fence permit exactly one commit
    assert sorted(outcomes) == ["committed", "stale"]


@pytest.mark.anyio
async def test_settings_survive_service_restart(engine: AsyncEngine) -> None:
    # Given: a real settings row and a fresh service instance
    factory = async_sessionmaker(engine, expire_on_commit=False)
    settings = SettingsService()
    wall = (CameraId(uuid4()), None, None, None)
    async with factory() as writer, writer.begin():
        updated = await settings.update(
            writer,
            SettingsPatchRequest(
                retention_days=21,
                quota_bytes=123_456_789,
                wall_slot_ids=wall,
            ),
        )
    # When: settings are loaded after the writer session has closed
    async with factory() as reader, reader.begin():
        restored = await SettingsService().get(reader)

    # Then: all values persist and the quota accounting contract remains explicit
    assert restored == updated
    assert restored.retention_days == 21
    assert restored.quota_bytes == 123_456_789
    assert restored.wall_slot_ids == wall
    assert settings.accounting.eviction_owner == "retention service (Task 13)"


@pytest.mark.anyio
async def test_real_rtsp_probe_decodes_one_1080p_h264_frame(tmp_path: Path) -> None:
    # Given: a cleanup-owned MediaMTX gateway and one finite source publisher
    repository_root = Path(__file__).parents[3]
    context = ScenarioContext(
        ScenarioContextConfig(
            run_id=RunId(f"camera-probe-{uuid4()}"),
            run_root=tmp_path,
            runtime_root=tmp_path / "runtime",
            compose_project=f"gw-camera-probe-{uuid4().hex[:10]}",
            allocated_port=allocate_loopback_port(),
        ),
        ledger=ResourceLedger(tmp_path / "resource-manifest.json"),
    )
    context.runtime_root.mkdir()
    media_path = f"test-publisher/{uuid4().hex}"
    secrets = prepare_runtime(context, media_path)

    # When: the real probe connects over RTSP/TCP while the finite process is live
    async with _media_gateway(context):
        ports = await wait_for_gateway(context)
        destination = (
            f"rtsp://task9-test-publisher:{secrets.publisher_password}"
            f"@127.0.0.1:{ports.rtsp}/{media_path}"
        )
        source_path = repository_root / "runtime/assets/fixtures/crosswalk.mp4"
        command = build_publisher_command(source_path, destination)
        finite_command = (*command[:-5], "-t", "5", *command[-5:])
        async with context.process(name="camera-probe-publisher", command=finite_command):
            await _wait_for_path(ports, media_path, secrets, ready=True)
            source_url = (
                f"rtsp://{READER_USER}:{secrets.reader_password}"
                f"@127.0.0.1:{ports.rtsp}/{media_path}"
            )
            result = await RtspSourceProbe().probe(parse_rtsp_source(source_url))
            assert result.codec == "h264"
            assert (result.width, result.height) == (1920, 1080)
        await _wait_for_path(ports, media_path, secrets, ready=False)
