"""Exercise delayed generation/version completions and restart crop recovery."""

from typing import Final
from uuid import UUID, uuid4

import anyio
from cryptography.fernet import Fernet
from sqlalchemy import select

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    PublicationAck,
)
from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.inference.clip import ClipAdapter, ClipTransport
from gods_watching.media.models import SourceGenerationId
from gods_watching.storage import (
    Appearance,
    Camera,
    CredentialCipher,
    CropGarbage,
    CropObjectStore,
    Database,
    StorageRepository,
)

from .task11_errors import Task11ExecutionError
from .task11_models import StaleRecoveryEvidence
from .task11_stale_support import DelayedTransport, build_handoff, build_publisher

_STALE_TRACK_ID: Final = 99
_UPGRADED_VERSION: Final = 2


async def run_stale_probe(  # noqa: PLR0915
    *, database: Database, crop_store: CropObjectStore, transport: ClipTransport
) -> StaleRecoveryEvidence:
    """Prove stale completions cannot replace the current representative."""
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    request = CameraCreateRequest.model_validate(
        {
            "name": f"Task 11 stale {uuid4().hex}",
            "source_url": "rtsp://fixture:8554/stale",
        }
    )
    async with database.transaction() as session:
        camera = await repository.add_camera(
            session, name=request.name, source_url=request.source_url
        )
        camera_session = await repository.start_camera_session(
            session, camera.id, cause="task11-stale"
        )
    binding = GenerationBinding(
        camera_id=CameraId(camera.id),
        camera_session_id=CameraSessionId(camera_session.id),
        db_generation_id=CameraGenerationId(camera_session.generation_id),
        source_generation_id=SourceGenerationId(uuid4()),
        camera_version=camera.version,
    )
    initial_publisher = build_publisher(
        database=database,
        repository=repository,
        crop_store=crop_store,
        clip=ClipAdapter(transport),
    )
    initial = await initial_publisher.accept_handoff(
        build_handoff(binding=binding, track_id=41, sequence=1, inset=False),
        now_monotonic=10.0,
    )
    if initial.outcome.value != "published":
        raise Task11ExecutionError(detail="initial clipped representative was not published")
    delayed_version = DelayedTransport(transport)
    old_publisher = build_publisher(
        database=database,
        repository=repository,
        crop_store=crop_store,
        clip=ClipAdapter(delayed_version),
    )
    new_publisher = build_publisher(
        database=database,
        repository=repository,
        crop_store=crop_store,
        clip=ClipAdapter(transport),
    )
    old_results: list[PublicationAck] = []

    async def publish_old_version() -> None:
        old_results.append(
            await old_publisher.accept_handoff(
                build_handoff(binding=binding, track_id=41, sequence=2, inset=True),
                now_monotonic=20.0,
            )
        )

    current: PublicationAck | None = None
    async with anyio.create_task_group() as task_group:
        task_group.start_soon(publish_old_version)
        await delayed_version.started.wait()
        current = await new_publisher.accept_handoff(
            build_handoff(binding=binding, track_id=41, sequence=2, inset=True),
            now_monotonic=20.0,
        )
        delayed_version.release.set()
    if not old_results:
        raise Task11ExecutionError(detail="delayed version did not return an acknowledgement")
    if current is None:
        raise Task11ExecutionError(detail="current version did not return an acknowledgement")
    late_version = old_results[0]
    async with database.transaction() as session:
        row = await session.get(Appearance, UUID(str(initial.appearance_id)))
        old_gc = await session.scalar(
            select(CropGarbage).where(CropGarbage.appearance_id == UUID(str(initial.appearance_id)))
        )
    if row is None:
        raise Task11ExecutionError(detail="current representative disappeared")
    current_crop = row.crop_object_key
    crop_count_after_version = len(tuple(crop_store.root.rglob("*.jpg")))

    delayed_generation = DelayedTransport(transport)
    generation_publisher = build_publisher(
        database=database,
        repository=repository,
        crop_store=crop_store,
        clip=ClipAdapter(delayed_generation),
    )
    generation_results: list[PublicationAck] = []

    async def publish_old_generation() -> None:
        generation_results.append(
            await generation_publisher.accept_handoff(
                build_handoff(
                    binding=binding,
                    track_id=_STALE_TRACK_ID,
                    sequence=1,
                    inset=True,
                ),
                now_monotonic=20.0,
            )
        )

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(publish_old_generation)
        await delayed_generation.started.wait()
        async with database.transaction() as session:
            durable_camera = await session.get(Camera, camera.id)
            if durable_camera is None:
                raise Task11ExecutionError(detail="stale probe camera disappeared")
            durable_camera.version += 1
        delayed_generation.release.set()
    if not generation_results:
        raise Task11ExecutionError(detail="delayed generation did not return an acknowledgement")
    stale_generation = generation_results[0]
    async with database.transaction() as session:
        stale_row = await session.scalar(
            select(Appearance).where(
                Appearance.camera_id == camera.id,
                Appearance.track_id == _STALE_TRACK_ID,
            )
        )

    orphan = crop_store.write(b"interrupted-publication")
    temporary = crop_store.root / "ff" / "ff" / f".{uuid4()}.tmp"
    _ = temporary.parent.mkdir(parents=True, exist_ok=True)
    _ = temporary.write_bytes(b"partial")
    recovery = AppearanceHandoffConsumer(
        build_publisher(
            database=database,
            repository=repository,
            crop_store=crop_store,
            clip=ClipAdapter(transport),
        )
    )
    report = await recovery.start()
    async with database.transaction() as session:
        recovered = await session.get(Appearance, row.id)
    return StaleRecoveryEvidence(
        old_generation_outcome=stale_generation.outcome.value,
        old_generation_row_absent=stale_row is None,
        late_version_outcome=late_version.outcome.value,
        committed_version=0 if recovered is None else recovered.representative_version,
        current_crop_preserved=(
            recovered is not None
            and recovered.crop_object_key == current_crop
            and (crop_store.root / current_crop).is_file()
        ),
        rejected_crop_cleaned=(
            len(tuple(crop_store.root.rglob("*.jpg"))) == crop_count_after_version
        ),
        old_version_gc_enqueued=old_gc is not None,
        orphan_removed_on_restart=(
            report.removed_jpegs >= 1 and not (crop_store.root / orphan.object_key).exists()
        ),
        temporary_removed_on_restart=report.removed_temps >= 1 and not temporary.exists(),
        clipped_to_inside_upgrade=(
            initial.representative_version == 1
            and current.outcome.value == "published"
            and current.representative_version == _UPGRADED_VERSION
        ),
    )


__all__ = ["run_stale_probe"]
