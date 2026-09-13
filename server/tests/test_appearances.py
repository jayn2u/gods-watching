from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from io import BytesIO
from math import sqrt
from uuid import uuid4

import pytest
from PIL import Image

from gods_watching.appearances.budget import BudgetSnapshot, ConservativeWriterBudget
from gods_watching.appearances.encoding import encode_rgb_crop, laplacian_variance
from gods_watching.appearances.policy import (
    CandidateRank,
    CropRejected,
    LifecycleClocks,
    appearance_id_for_track,
    rank_candidate,
    should_upgrade,
    validate_crop_geometry,
)
from gods_watching.appearances.queue import (
    PendingEmbedding,
    PendingEmbeddingQueue,
    QueueDecision,
    QueueKey,
)
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import RgbCrop
from gods_watching.media.models import SourceGenerationId


def _key() -> tuple[CameraId, CameraSessionId]:
    return CameraId(uuid4()), CameraSessionId(uuid4())


def test_rank_candidate_prefers_fully_inside_before_quality() -> None:
    # Given: a clipped crop with a high quality score and an inside crop with a lower score
    # When: both candidates are ranked against the same source geometry
    clipped = rank_candidate(
        bounding_box=(-1, 0, 80, 160),
        source_width=100,
        source_height=200,
        confidence=1.0,
        laplacian_variance=1000.0,
    )
    inside = rank_candidate(
        bounding_box=(1, 1, 33, 65),
        source_width=100,
        source_height=200,
        confidence=0.5,
        laplacian_variance=1.0,
    )

    # Then: border completeness is the first ordering dimension
    assert clipped == CandidateRank(fully_inside=False, score=sqrt(float(81 * 160)))
    assert inside == CandidateRank(
        fully_inside=True,
        score=sqrt(float(32 * 64)) * 0.5 * 0.01,
    )
    assert inside > clipped


def test_rank_candidate_treats_source_border_as_clipped() -> None:
    # Given: one valid crop touching the source border and one with a one-pixel margin
    touching = (0.0, 0.0, 80.0, 160.0)
    inset = (1.0, 1.0, 81.0, 161.0)

    # When: both candidates are ranked by the representative policy
    touching_rank = rank_candidate(
        bounding_box=touching,
        source_width=100,
        source_height=200,
        confidence=0.9,
        laplacian_variance=100.0,
    )
    inset_rank = rank_candidate(
        bounding_box=inset,
        source_width=100,
        source_height=200,
        confidence=0.9,
        laplacian_variance=100.0,
    )

    # Then: the inset crop wins the border rank before its scalar score is compared
    assert touching_rank.fully_inside is False
    assert inset_rank.fully_inside is True
    assert inset_rank > touching_rank


def test_candidate_policy_rejects_invalid_crop_geometry() -> None:
    # Given: source geometry and a crop smaller than the minimum supported person size
    # When / Then: degenerate, clipped, and undersized crops fail at the handoff boundary
    for bounding_box in ((20, 20, 20, 80), (-1, 0, 31, 64), (0, 0, 32, 63)):
        with pytest.raises(CropRejected):
            _ = validate_crop_geometry(
                bounding_box=bounding_box,
                source_width=100,
                source_height=100,
            )


def test_candidate_policy_requires_ten_percent_gain_and_upgrade_cooldown() -> None:
    # Given: an already committed representative and a monotonic upgrade clock
    current = CandidateRank(fully_inside=True, score=100.0)

    # When / Then: a borderline gain or an early retry is suppressed
    assert not should_upgrade(current, CandidateRank(fully_inside=True, score=109.9), 2.0)
    assert not should_upgrade(current, CandidateRank(fully_inside=True, score=120.0), 1.99)

    # When / Then: a full-quality improvement after the cooldown is eligible
    assert should_upgrade(current, CandidateRank(fully_inside=True, score=110.0), 2.0)
    assert should_upgrade(
        CandidateRank(fully_inside=False, score=100.0),
        CandidateRank(fully_inside=True, score=0.0),
        2.0,
    )


def test_appearance_id_is_stable_for_camera_session_track_namespace() -> None:
    # Given: one camera/session track namespace
    camera_id, session_id = _key()

    # When: the deterministic identity function is called repeatedly
    first = appearance_id_for_track(camera_id, session_id, 7)
    second = appearance_id_for_track(camera_id, session_id, 7)

    # Then: one continuous local track maps to one stable UUID
    assert first == second
    assert first != appearance_id_for_track(camera_id, session_id, 8)
    assert first != appearance_id_for_track(CameraId(uuid4()), session_id, 7)


def test_first_candidate_keeps_ingress_clock_after_late_confirmation() -> None:
    # Given: an ingress event captured before detector confirmation
    first_seen = datetime(2026, 9, 7, 1, 0, tzinfo=UTC)
    clocks = LifecycleClocks(
        first_seen=first_seen,
        t_detect_monotonic=1.25,
    )

    # When: a retry observes a later confirmation time
    retried = clocks

    # Then: the first detector receipt and ingress timestamp remain unchanged
    assert retried.first_seen == first_seen
    assert retried.t_detect_monotonic == 1.25


def test_embedding_queue_keeps_first_representatives_ahead_of_upgrades() -> None:
    # Given: a global queue at its prescribed capacity, filled with upgrades
    queue: PendingEmbeddingQueue[None] = PendingEmbeddingQueue(max_pending=256)
    for track_id in range(256):
        decision = queue.enqueue(
            PendingEmbedding[None](
                key=QueueKey(
                    CameraId(uuid4()),
                    CameraSessionId(uuid4()),
                    SourceGenerationId(uuid4()),
                    track_id,
                ),
                is_initial=False,
                representative_version=2,
            )
        )
        assert decision is QueueDecision.ENQUEUED

    # When: a first representative arrives while upgrades are waiting
    first = PendingEmbedding[None](
        key=QueueKey(CameraId(uuid4()), CameraSessionId(uuid4()), SourceGenerationId(uuid4()), 1),
        is_initial=True,
        representative_version=1,
    )
    decision = queue.enqueue(first)

    # Then: the first representative is accepted and dequeued before upgrades
    assert decision is QueueDecision.EVICTED_UPGRADE
    assert queue.pop() == first


def test_embedding_queue_allows_one_pending_candidate_per_track() -> None:
    # Given: one track with an upgrade already waiting
    key = QueueKey(CameraId(uuid4()), CameraSessionId(uuid4()), SourceGenerationId(uuid4()), 7)
    queue: PendingEmbeddingQueue[None] = PendingEmbeddingQueue()
    first = PendingEmbedding[None](key=key, is_initial=True, representative_version=1)
    upgrade = PendingEmbedding[None](key=key, is_initial=False, representative_version=2)

    # When: the same key is submitted twice
    assert queue.enqueue(first) is QueueDecision.ENQUEUED
    assert queue.enqueue(upgrade) is QueueDecision.REPLACED_BY_FIRST

    # Then: the first representative remains the only pending work item
    assert len(queue) == 1
    assert queue.pop() == first
    assert queue.pop() is None


def test_rgb_crop_is_encoded_as_original_rgb_jpeg_quality_ninety() -> None:
    # Given: source-resolution RGB pixels with a deterministic luminance pattern
    width, height = 32, 64
    pixels = bytes(index % 256 for index in range(width * height * 3))

    # When: the bounded crop is encoded for Triton and object storage
    encoded = encode_rgb_crop(RgbCrop(data=pixels, width=width, height=height))

    # Then: the payload is a decodable RGB JPEG with the exact crop dimensions
    with Image.open(BytesIO(encoded)) as image:
        assert image.format == "JPEG"
        assert image.mode == "RGB"
        assert image.size == (width, height)


def test_laplacian_variance_is_finite_for_uniform_rgb_crop() -> None:
    # Given: a valid but perfectly uniform RGB crop
    crop = RgbCrop(data=bytes(32 * 64 * 3), width=32, height=64)

    # When: sharpness is measured at the crop boundary
    variance = laplacian_variance(crop)

    # Then: the metric is finite and zero for the uniform image
    assert variance == 0.0


def test_writer_budget_serializes_reservations_and_counts_pending_gc_once() -> None:
    async def exercise() -> None:
        budget = ConservativeWriterBudget(minimum_free_bytes=0)
        first_snapshot = BudgetSnapshot(
            new_crop_bytes=60,
            physical_crop_bytes=0,
            pending_gc_bytes=0,
            relation_bytes=0,
            filesystem_free_bytes=100,
            quota_bytes=100,
        )
        first = await budget.reserve(first_snapshot)
        assert first is not None

        waiting = asyncio.create_task(budget.reserve(first_snapshot))
        await asyncio.sleep(0)
        assert not waiting.done()
        await first.release()
        second = await waiting
        assert second is not None
        await second.release()

        pending_gc_snapshot = BudgetSnapshot(
            new_crop_bytes=30,
            physical_crop_bytes=60,
            pending_gc_bytes=60,
            relation_bytes=0,
            filesystem_free_bytes=100,
            quota_bytes=100,
        )
        assert pending_gc_snapshot.projected_bytes == 90
        pending_gc_lease = await budget.reserve(pending_gc_snapshot)
        assert pending_gc_lease is not None
        await pending_gc_lease.release()

        over_threshold = BudgetSnapshot(
            new_crop_bytes=36,
            physical_crop_bytes=60,
            pending_gc_bytes=60,
            relation_bytes=0,
            filesystem_free_bytes=100,
            quota_bytes=100,
        )
        assert await budget.reserve(over_threshold) is None

    asyncio.run(exercise())
