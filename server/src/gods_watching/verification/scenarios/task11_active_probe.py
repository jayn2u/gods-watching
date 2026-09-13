"""Drive real RTSP ingest through detector, CLIP, storage, and search."""

from collections.abc import Sequence
from io import BytesIO
from math import sqrt
from typing import Final
from uuid import UUID, uuid4

import anyio
import numpy as np
from cryptography.fernet import Fernet
from PIL import Image

from gods_watching.appearances import (
    AppearanceHandoffConsumer,
    AppearancePublisher,
    ConservativeWriterBudget,
)
from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.appearances import AppearanceResponse
from gods_watching.contracts.cameras import CameraCreateRequest
from gods_watching.contracts.identifiers import AppearanceId, CameraId, CameraSessionId
from gods_watching.contracts.pipeline import GenerationBinding
from gods_watching.contracts.search import BrowseSearchRequest
from gods_watching.inference.clip import ClipAdapter, TritonClipTransport
from gods_watching.inference.detector import DetectorClient, TritonGrpcDetectorTransport
from gods_watching.ingest import IngestCoordinator, IngestWorker, IngestWorkerConfiguration
from gods_watching.media.models import SourceGenerationId
from gods_watching.search import AppearanceLookupService, SearchRepository, SearchService
from gods_watching.storage import (
    Appearance,
    CredentialCipher,
    CropObjectStore,
    Database,
    StorageRepository,
)

from .task11_errors import Task11ExecutionError
from .task11_models import ActivePublicationEvidence

_MODEL_ID: Final = "openai/clip-vit-base-patch32"
_MODEL_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"


def unique_track_results(
    results: Sequence[AppearanceResponse], appearance_id: AppearanceId
) -> bool:
    """Require one appearance per track and exactly one row for the observed appearance.

    Several people may start tracks in the same frame, so the count of results for a
    camera is timing-dependent; duplicate appearances for one track are the defect.
    """
    tracks = [(result.session_id, result.track_id) for result in results]
    observed = sum(1 for result in results if result.appearance_id == appearance_id)
    return len(tracks) == len(set(tracks)) and observed == 1


async def run_active_probe(  # noqa: PLR0915
    *, database: Database, crop_store: CropObjectStore, triton_url: str, rtsp_url: str
) -> ActivePublicationEvidence:
    """Observe one search-visible active representative before stopping its worker."""
    repository = StorageRepository(CredentialCipher(Fernet.generate_key()))
    request = CameraCreateRequest.model_validate(
        {"name": f"Task 11 active {uuid4().hex}", "source_url": rtsp_url}
    )
    async with database.transaction() as session:
        camera = await repository.add_camera(
            session, name=request.name, source_url=request.source_url
        )
        camera_session = await repository.start_camera_session(
            session, camera.id, cause="task11-real-ingest"
        )
    binding = GenerationBinding(
        camera_id=CameraId(camera.id),
        camera_session_id=CameraSessionId(camera_session.id),
        db_generation_id=CameraGenerationId(camera_session.generation_id),
        source_generation_id=SourceGenerationId(uuid4()),
        camera_version=camera.version,
    )
    detector_transport = TritonGrpcDetectorTransport(url=triton_url)
    stop_event = anyio.Event()
    active_row: Appearance | None = None
    active_crop = b""
    unique_result = False
    active_track_count = 0
    try:
        async with TritonClipTransport(triton_url) as clip_transport:
            publisher = AppearancePublisher(
                database=database,
                storage=repository,
                crop_store=crop_store,
                clip=ClipAdapter(clip_transport),
                model_id=_MODEL_ID,
                model_revision=_MODEL_REVISION,
                writer_budget=ConservativeWriterBudget(minimum_free_bytes=0),
            )
            consumer = AppearanceHandoffConsumer(publisher)
            _ = await consumer.start()
            worker = IngestWorker(
                IngestWorkerConfiguration(
                    generation=binding,
                    source_url=rtsp_url,
                    detector_deadline_seconds=2.0,
                ),
                DetectorClient(transport=detector_transport, timeout_seconds=2.0),
                consumer,
            )
            coordinator = IngestCoordinator()
            coordinator.add(worker)

            async def run_coordinator() -> None:
                await coordinator.run(stop_event=stop_event)

            async with anyio.create_task_group() as task_group:
                task_group.start_soon(run_coordinator)
                try:
                    with anyio.fail_after(45.0):
                        while active_row is None:
                            async with database.transaction() as session:
                                response = await SearchService(SearchRepository()).search(
                                    session,
                                    BrowseSearchRequest(
                                        mode="browse",
                                        camera_ids=(CameraId(camera.id),),
                                        limit=10,
                                    ),
                                )
                                if response.results:
                                    appearance_id = UUID(str(response.results[0].appearance_id))
                                    unique_result = unique_track_results(
                                        response.results, AppearanceId(appearance_id)
                                    )
                                    active_row = await session.get(Appearance, appearance_id)
                                    payload = await AppearanceLookupService(
                                        SearchRepository(), crop_store
                                    ).get_crop(session, appearance_id)
                                    active_crop = payload.payload
                                    active_track_count = worker.tracking.active_track_count
                            if active_row is None:
                                await anyio.sleep(0.1)
                finally:
                    stop_event.set()
    finally:
        await detector_transport.close()
    if active_row is None:
        raise Task11ExecutionError(detail="active appearance did not become searchable")
    with Image.open(BytesIO(active_crop)) as image:
        jpeg_rgb = image.format == "JPEG" and image.mode == "RGB"
        crop_dimensions = image.size
    vector = np.fromstring((active_row.embedding or "").strip("[]"), sep=",")
    async with database.transaction() as session:
        ended = await session.get(Appearance, active_row.id)
    return ActivePublicationEvidence(
        appearance_id=str(active_row.id),
        active_before_exit=active_row.ended_at is None and active_track_count > 0,
        unique_result=unique_result,
        representative_version=active_row.representative_version,
        jpeg_rgb=jpeg_rgb,
        crop_dimensions=crop_dimensions,
        embedding_dimension=int(vector.size),
        embedding_norm=sqrt(float(np.dot(vector, vector))),
        model_revision=active_row.model_revision,
        ended_after_shutdown=ended is not None and ended.ended_at is not None,
    )


__all__ = ["run_active_probe"]
