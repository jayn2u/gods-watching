"""Build deterministic publication inputs for the Task 11 stale probe."""

from datetime import UTC, datetime
from typing import Final, final

import anyio

from gods_watching.appearances import AppearancePublisher, ConservativeWriterBudget
from gods_watching.contracts.pipeline import (
    CropCandidate,
    GenerationBinding,
    PipelineHandoff,
    RgbCrop,
)
from gods_watching.inference.clip import ClipAdapter, ClipEmbedding, ClipTransport
from gods_watching.storage import CropObjectStore, Database, StorageRepository
from gods_watching.tracking import (
    DetectorInputReference,
    LifecycleKind,
    LocalTrackId,
    TrackKey,
    TrackLifecycle,
    TrackObservation,
)

MODEL_ID: Final = "openai/clip-vit-base-patch32"
MODEL_REVISION: Final = "57c216476eefef5ab752ec549e440a49ae4ae5f3"
WIDTH: Final = 96
HEIGHT: Final = 192


@final
class DelayedTransport:
    """Delay one real Triton image response at a deterministic completion fence."""

    def __init__(self, transport: ClipTransport) -> None:
        """Wrap the production transport with controllable completion events."""
        self._transport = transport
        self.started = anyio.Event()
        self.release = anyio.Event()

    async def embed_image(self, image: bytes) -> ClipEmbedding:
        """Release an image embedding only after the probe opens its fence."""
        self.started.set()
        await self.release.wait()
        return tuple(await self._transport.embed_image(image))

    async def embed_text(self, text: str) -> ClipEmbedding:
        """Delegate text embedding to the wrapped production transport."""
        return tuple(await self._transport.embed_text(text))


def build_handoff(
    *,
    binding: GenerationBinding,
    track_id: int,
    sequence: int,
    inset: bool,
) -> PipelineHandoff:
    """Build a candidate whose rank changes when inset from the source border."""
    observed_at = datetime(2026, 9, 13, 2, 0, sequence, tzinfo=UTC)
    key = TrackKey(
        binding.camera_id,
        binding.camera_session_id,
        binding.source_generation_id,
        LocalTrackId(track_id),
    )
    offset = 1.0 if inset else 0.0
    observation = TrackObservation(
        bounding_box=(offset, offset, WIDTH + offset, HEIGHT + offset),
        confidence=0.9,
        detector_input_reference=DetectorInputReference(f"task11-stale-{track_id}-{sequence}"),
        ingress_utc=observed_at,
        ingress_monotonic=float(sequence),
        detector_result_monotonic=float(sequence) + 0.1,
    )
    kind = LifecycleKind.START if sequence == 1 else LifecycleKind.UPDATE
    return PipelineHandoff(
        generation=binding,
        lifecycle=TrackLifecycle(
            kind=kind,
            track_key=key,
            source_generation_id=binding.source_generation_id,
            scope_epoch=0,
            sequence=sequence,
            observation=observation,
            first_candidate=observation,
            first_seen=observed_at,
            last_seen=observed_at,
            t_detect_monotonic=float(sequence) + 0.1,
        ),
        candidate=CropCandidate(
            track_key=key,
            observation=observation,
            crop=RgbCrop(
                data=bytes(index % 251 for index in range(WIDTH * HEIGHT * 3)),
                width=WIDTH,
                height=HEIGHT,
            ),
            source_width=WIDTH + 2,
            source_height=HEIGHT + 2,
        ),
    )


def build_publisher(
    *,
    database: Database,
    repository: StorageRepository,
    crop_store: CropObjectStore,
    clip: ClipAdapter,
) -> AppearancePublisher:
    """Create the real publisher with a deterministic monotonic clock."""
    return AppearancePublisher(
        database=database,
        storage=repository,
        crop_store=crop_store,
        clip=clip,
        model_id=MODEL_ID,
        model_revision=MODEL_REVISION,
        writer_budget=ConservativeWriterBudget(minimum_free_bytes=0),
        monotonic_clock=lambda: 20.0,
    )


__all__ = ["DelayedTransport", "build_handoff", "build_publisher"]
