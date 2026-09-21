from datetime import UTC, datetime
from functools import partial
from typing import TYPE_CHECKING, override
from uuid import uuid4

import anyio
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

from gods_watching.cameras.lifecycle import CameraGenerationId
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.contracts.pipeline import (
    CropCandidate,
    GenerationBinding,
    PipelineHandoff,
    RgbCrop,
)
from gods_watching.inference.detector import Detection, DetectorRequest, DetectorResult
from gods_watching.ingest.decoder import (
    DecoderConfiguration,
    DecoderStatusSink,
    PyAvRtspDecoder,
)
from gods_watching.ingest.models import DecodedFrame, IngestStats, LatestFrameSlot
from gods_watching.ingest.scheduler import FairRoundRobinScheduler
from gods_watching.ingest.worker import (
    IngestCoordinator,
    IngestWorker,
    IngestWorkerConfiguration,
    PipelineHandoffConsumer,
)
from gods_watching.media.models import SourceGenerationId
from gods_watching.tracking import (
    ByteTrackScope,
    DetectionFrame,
    DetectorInputReference,
    LifecycleKind,
    TrackingScope,
)


def test_ingest_stats_reports_detector_result_rate_in_bounded_window() -> None:
    # Given: three completed detector results spanning half a second
    stats = IngestStats()
    stats.record_detector_result(1.0)
    stats.record_detector_result(1.25)
    stats.record_detector_result(1.5)

    # When: runtime telemetry is sampled
    snapshot = stats.snapshot(now_monotonic=1.5, dropped_frames=0)

    # Then: accepted detector throughput is measured from completion times
    assert snapshot.detector_framerate == 4.0


def test_pipeline_start_handoff_requires_one_bounded_rgb_crop() -> None:
    camera_id = CameraId(uuid4())
    session_id = CameraSessionId(uuid4())
    source_generation_id = SourceGenerationId(uuid4())
    scope = ByteTrackScope(
        scope=TrackingScope(
            camera_id=camera_id,
            session_id=session_id,
            source_generation_id=source_generation_id,
            detection_enabled=True,
        ),
        threshold=0.5,
    )
    frame = DetectionFrame(
        camera_id=camera_id,
        session_id=session_id,
        source_generation_id=source_generation_id,
        ingress_utc=datetime(2026, 1, 1, tzinfo=UTC),
        ingress_monotonic=0.0,
        detector_result_monotonic=0.1,
        source_width=640,
        source_height=480,
        detector_input_reference=DetectorInputReference("frame-0"),
        detections=(Detection(100, 100, 180, 260, 0.9, 0),),
    )
    lifecycle = scope.feed(frame)[0]
    binding = GenerationBinding(
        camera_id=camera_id,
        camera_session_id=session_id,
        db_generation_id=CameraGenerationId(uuid4()),
        source_generation_id=source_generation_id,
        camera_version=1,
    )

    # When the start handoff is assembled without the crop bytes
    # Then the contract rejects it instead of allowing unbounded/raw-frame fallback.
    with pytest.raises(ValueError, match="candidate"):
        _ = PipelineHandoff(generation=binding, lifecycle=lifecycle, candidate=None)

    candidate = CropCandidate(
        track_key=lifecycle.track_key,
        observation=lifecycle.first_candidate,
        crop=RgbCrop(data=bytes(32 * 64 * 3), width=32, height=64),
        source_width=640,
        source_height=480,
    )
    handoff = PipelineHandoff(
        generation=binding,
        lifecycle=lifecycle,
        candidate=candidate,
    )
    assert handoff.candidate is candidate


def test_latest_frame_slot_replaces_old_frame_and_counts_one_drop() -> None:
    # Given: a bounded latest-frame slot
    slot = LatestFrameSlot()
    first = DecodedFrame(
        source_generation_id=SourceGenerationId(uuid4()),
        ingress_utc=datetime(2026, 1, 1, tzinfo=UTC),
        ingress_monotonic=1.0,
        reference=DetectorInputReference("frame-1"),
        width=32,
        height=64,
        rgb_bytes=bytes(32 * 64 * 3),
        encoded_image=b"jpeg-1",
    )
    second = DecodedFrame(
        source_generation_id=first.source_generation_id,
        ingress_utc=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
        ingress_monotonic=2.0,
        reference=DetectorInputReference("frame-2"),
        width=32,
        height=64,
        rgb_bytes=bytes(32 * 64 * 3),
        encoded_image=b"jpeg-2",
    )

    # When: decode publishes two frames before the detector samples
    slot.put(first)
    slot.put(second)

    # Then: only the newest frame remains and the overwrite is observable
    assert slot.take_latest() == second
    assert slot.take_latest() is None
    assert slot.dropped_frames == 1
    assert slot.peak_occupancy == 1
    assert slot.occupancy == 0


def test_worker_accounts_for_every_dispatch_and_detector_high_water() -> None:
    # Given: a worker with one empty, one disabled, and one enabled dispatch
    binding = _new_binding()
    detector = _ImmediateDetector()

    async def consume(handoff: PipelineHandoff) -> None:
        del handoff

    worker = _new_worker(binding, detector, consume)

    async def exercise() -> None:
        assert await worker.sample_once() == ()
        worker.receive(_new_frame(binding, seconds=1.0, reference="disabled"))
        _ = await worker.set_detection_enabled(False)
        assert await worker.sample_once() == ()
        _ = await worker.set_detection_enabled(True)
        worker.receive(_new_frame(binding, seconds=1.2, reference="enabled"))
        _ = await worker.sample_once()

    anyio.run(exercise)

    stats = worker.stats
    assert stats.scheduler_dispatches == 3
    assert stats.dispatch_no_frame == 1
    assert stats.dispatch_detection_disabled == 1
    assert stats.dispatch_detector_requested == 1
    assert stats.dispatch_generation_fenced == 0
    assert stats.dispatch_closed == 0
    assert stats.dispatch_outcomes == stats.scheduler_dispatches
    assert stats.peak_detector_requests == 1
    assert stats.pending_detector_requests == 0


def test_round_robin_scheduler_skips_inflight_camera_and_rotates() -> None:
    # Given: three cameras with a five-frame-per-second target
    camera_ids = tuple(CameraId(uuid4()) for _ in range(3))
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    for camera_id in camera_ids:
        scheduler.register(camera_id)

    # When: dispatch time advances through four fair slots while camera zero is busy
    selected: list[CameraId] = []
    selected.append(_required_camera(scheduler.next_due(now_monotonic=0.0)))
    _ = scheduler.mark_dispatched(selected[-1], now_monotonic=0.0)
    selected.append(_required_camera(scheduler.next_due(now_monotonic=0.2)))
    _ = scheduler.mark_dispatched(selected[-1], now_monotonic=0.2)
    selected.append(_required_camera(scheduler.next_due(now_monotonic=0.4)))
    _ = scheduler.mark_dispatched(selected[-1], now_monotonic=0.4)
    scheduler.complete(selected[1])
    selected.append(_required_camera(scheduler.next_due(now_monotonic=0.6)))

    # Then: the in-flight camera is skipped and the next camera gets the turn
    assert selected == [camera_ids[0], camera_ids[1], camera_ids[2], camera_ids[1]]
    assert scheduler.in_flight_count == 2


def _required_camera(camera_id: CameraId | None) -> CameraId:
    assert camera_id is not None
    return camera_id


def test_detection_toggle_closes_tracks_and_stops_detector_work() -> None:
    # Given: one active worker track and an enabled detector
    binding = _new_binding()
    detector = _ImmediateDetector()
    handoffs: list[PipelineHandoff] = []

    async def consume(handoff: PipelineHandoff) -> None:
        handoffs.append(handoff)

    worker = _new_worker(binding, detector, consume)

    async def exercise() -> None:
        worker.receive(_new_frame(binding, seconds=1.0, reference="toggle-1"))
        starts = await worker.sample_once()
        ended = await worker.set_detection_enabled(False)
        worker.receive(_new_frame(binding, seconds=2.0, reference="toggle-2"))
        disabled_sample = await worker.sample_once()

        assert len(starts) == 1
        assert starts[0].lifecycle.kind.value == "start"
        assert len(ended) == 1
        assert ended[0].lifecycle.kind.value == "end"
        assert disabled_sample == ()
        assert worker.stats.detector_requests == 1
        assert worker.detection_enabled is False

    _ = anyio.run(exercise)


def test_cancellation_cannot_drop_a_terminal_handoff() -> None:
    binding = _new_binding()
    detector = _ImmediateDetector()
    delivered: list[LifecycleKind] = []
    end_delivery_started = anyio.Event()

    async def consume(handoff: PipelineHandoff) -> None:
        if handoff.lifecycle.kind is LifecycleKind.END:
            end_delivery_started.set()
            await anyio.sleep(0.05)
        delivered.append(handoff.lifecycle.kind)

    worker = _new_worker(binding, detector, consume)

    async def exercise() -> None:
        worker.receive(_new_frame(binding, seconds=1.0, reference="cancel-end"))
        assert [item.lifecycle.kind for item in await worker.sample_once()] == [LifecycleKind.START]

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(worker.close)
            await end_delivery_started.wait()
            task_group.cancel_scope.cancel()

        assert delivered == [LifecycleKind.START, LifecycleKind.END]

    anyio.run(exercise)


def test_detector_result_after_threshold_reset_cannot_start_a_new_track() -> None:
    # Given: one detector request paused before returning its result
    binding = _new_binding()
    detector = _BlockingDetector()
    handoffs: list[PipelineHandoff] = []

    async def consume(handoff: PipelineHandoff) -> None:
        handoffs.append(handoff)

    worker = _new_worker(binding, detector, consume)

    async def exercise() -> None:
        worker.receive(_new_frame(binding, seconds=1.0, reference="stale-threshold"))
        results: list[tuple[PipelineHandoff, ...]] = []

        async def sample() -> None:
            results.append(await worker.sample_once())

        async with anyio.create_task_group() as task_group:
            task_group.start_soon(sample)
            await detector.started.wait()
            assert await worker.set_threshold(0.6) == ()
            detector.release.set()

        assert results == [()]
        assert handoffs == []
        assert worker.tracking.active_track_count == 0

    _ = anyio.run(exercise)


def test_invalid_crop_is_counted_and_cannot_orphan_a_track() -> None:
    # Given: a detector response whose first box is below the bounded crop dimensions
    binding = _new_binding()
    detector = _SequenceDetector(
        (
            DetectorResult(detections=(Detection(100, 100, 131, 163, 0.9, 0),)),
            DetectorResult(detections=(Detection(100, 100, 180, 260, 0.9, 0),)),
            DetectorResult(detections=(Detection(102, 100, 182, 260, 0.9, 0),)),
        )
    )
    handoffs: list[PipelineHandoff] = []

    async def consume(handoff: PipelineHandoff) -> None:
        handoffs.append(handoff)

    worker = _new_worker(binding, detector, consume)

    async def exercise() -> None:
        # When: the invalid result is sampled, then a valid result arrives
        worker.receive(_new_frame(binding, seconds=1.0, reference="invalid-crop"))
        assert await worker.sample_once() == ()

        # Then: the invalid detection is visible in counters and leaves no active orphan
        assert worker.tracking.active_track_count == 0
        assert worker.stats.detector_detections == 1
        assert worker.stats.invalid_detections == 1
        assert worker.stats.eligible_detections == 0

        worker.receive(_new_frame(binding, seconds=1.2, reference="valid-crop"))
        assert await worker.sample_once() == ()
        worker.receive(_new_frame(binding, seconds=1.4, reference="valid-crop-2"))
        starts = await worker.sample_once()
        assert [handoff.lifecycle.kind for handoff in starts] == [LifecycleKind.START]
        assert len(handoffs) == 1
        assert worker.tracking.active_track_count == 1

    _ = anyio.run(exercise)


class _ImmediateDetector:
    async def detect(self, request: DetectorRequest) -> DetectorResult:
        return DetectorResult(detections=(Detection(100, 100, 180, 260, request.confidence, 0),))


class _BlockingDetector:
    def __init__(self) -> None:
        self.started: anyio.Event = anyio.Event()
        self.release: anyio.Event = anyio.Event()

    async def detect(self, request: DetectorRequest) -> DetectorResult:
        self.started.set()
        await self.release.wait()
        return DetectorResult(detections=(Detection(100, 100, 180, 260, request.confidence, 0),))


class _SequenceDetector:
    def __init__(self, results: tuple[DetectorResult, ...]) -> None:
        self._results: Iterator[DetectorResult] = iter(results)

    async def detect(self, request: DetectorRequest) -> DetectorResult:
        _ = request
        return next(self._results)


def _new_binding() -> GenerationBinding:
    return GenerationBinding(
        camera_id=CameraId(uuid4()),
        camera_session_id=CameraSessionId(uuid4()),
        db_generation_id=CameraGenerationId(uuid4()),
        source_generation_id=SourceGenerationId(uuid4()),
        camera_version=1,
    )


def _new_frame(binding: GenerationBinding, *, seconds: float, reference: str) -> DecodedFrame:
    return DecodedFrame(
        source_generation_id=binding.source_generation_id,
        ingress_utc=datetime(2026, 1, 1, tzinfo=UTC),
        ingress_monotonic=seconds,
        reference=DetectorInputReference(reference),
        width=640,
        height=480,
        rgb_bytes=bytes(640 * 480 * 3),
        encoded_image=b"jpeg",
    )


def _new_worker(
    binding: GenerationBinding,
    detector: _ImmediateDetector | _BlockingDetector | _SequenceDetector,
    consume: PipelineHandoffConsumer,
) -> IngestWorker:
    return IngestWorker(
        IngestWorkerConfiguration(generation=binding, source_url="rtsp://camera/test"),
        detector,
        consume,
    )


class _ParkedDecoder(PyAvRtspDecoder):
    """Stand in for RTSP decode by parking until the worker stops the decoder."""

    def __init__(self, binding: GenerationBinding) -> None:
        super().__init__(
            DecoderConfiguration(
                source_url="rtsp://camera/test",
                source_generation_id=binding.source_generation_id,
                slot=LatestFrameSlot(),
            )
        )
        self.started: anyio.Event = anyio.Event()
        self.finished: anyio.Event = anyio.Event()
        self._parked: anyio.Event = anyio.Event()

    @override
    def stop(self) -> None:
        super().stop()
        self._parked.set()

    @override
    async def run_forever(self, *, status_sink: DecoderStatusSink | None = None) -> None:
        del status_sink
        self.started.set()
        try:
            await self._parked.wait()
        finally:
            self.finished.set()


async def _discard(handoff: PipelineHandoff) -> None:
    del handoff


def _new_parked_worker(binding: GenerationBinding, decoder: _ParkedDecoder) -> IngestWorker:
    return IngestWorker(
        IngestWorkerConfiguration(generation=binding, source_url="rtsp://camera/test"),
        _SequenceDetector(()),
        _discard,
        decoder=decoder,
    )


def test_coordinator_starts_decoder_for_camera_added_while_running() -> None:
    async def exercise() -> None:
        # Given: a running coordinator that already decodes one camera
        coordinator = IngestCoordinator()
        first_binding = _new_binding()
        second_binding = _new_binding()
        first_decoder = _ParkedDecoder(first_binding)
        second_decoder = _ParkedDecoder(second_binding)
        coordinator.add(_new_parked_worker(first_binding, first_decoder))
        stop = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(partial(coordinator.run, stop_event=stop))
            with anyio.fail_after(2.0):
                await first_decoder.started.wait()

            # When: the worker process adds a second camera without restarting the loop
            coordinator.add(_new_parked_worker(second_binding, second_decoder))

            # Then: the new camera decodes and both decoders stop with the coordinator
            with anyio.fail_after(2.0):
                await second_decoder.started.wait()
            stop.set()
        assert first_decoder.finished.is_set()
        assert second_decoder.finished.is_set()

    anyio.run(exercise)


def test_coordinator_remove_stops_only_that_decoder_while_running() -> None:
    async def exercise() -> None:
        # Given: a running coordinator decoding two cameras
        coordinator = IngestCoordinator()
        first_binding = _new_binding()
        second_binding = _new_binding()
        first_decoder = _ParkedDecoder(first_binding)
        second_decoder = _ParkedDecoder(second_binding)
        coordinator.add(_new_parked_worker(first_binding, first_decoder))
        coordinator.add(_new_parked_worker(second_binding, second_decoder))
        stop = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(partial(coordinator.run, stop_event=stop))
            with anyio.fail_after(2.0):
                await first_decoder.started.wait()
                await second_decoder.started.wait()

            # When: one camera is disabled or deleted
            _ = await coordinator.remove(first_binding.camera_id)

            # Then: only its decoder stops and the other camera keeps decoding
            with anyio.fail_after(2.0):
                await first_decoder.finished.wait()
            assert not second_decoder.finished.is_set()
            assert [worker.camera_id for worker in coordinator.workers] == [
                second_binding.camera_id
            ]
            stop.set()

    anyio.run(exercise)


def test_coordinator_restarts_camera_with_a_new_generation_while_running() -> None:
    async def exercise() -> None:
        # Given: a running coordinator decoding one camera generation
        coordinator = IngestCoordinator()
        old_binding = _new_binding()
        new_binding = GenerationBinding(
            camera_id=old_binding.camera_id,
            camera_session_id=CameraSessionId(uuid4()),
            db_generation_id=CameraGenerationId(uuid4()),
            source_generation_id=SourceGenerationId(uuid4()),
            camera_version=old_binding.camera_version + 1,
        )
        old_decoder = _ParkedDecoder(old_binding)
        new_decoder = _ParkedDecoder(new_binding)
        coordinator.add(_new_parked_worker(old_binding, old_decoder))
        stop = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(partial(coordinator.run, stop_event=stop))
            with anyio.fail_after(2.0):
                await old_decoder.started.wait()

            # When: a source edit replaces the camera's generation
            _ = await coordinator.remove(old_binding.camera_id)
            coordinator.add(_new_parked_worker(new_binding, new_decoder))

            # Then: the old decoder stops and the replacement decodes
            with anyio.fail_after(2.0):
                await old_decoder.finished.wait()
                await new_decoder.started.wait()
            stop.set()

    anyio.run(exercise)


class _FailingDecoder(_ParkedDecoder):
    """Fail after starting, like a reconnect whose session the API already replaced."""

    @override
    async def run_forever(self, *, status_sink: DecoderStatusSink | None = None) -> None:
        del status_sink
        self.started.set()
        try:
            detail = "camera generation was replaced during reconnect"
            raise RuntimeError(detail)
        finally:
            self.finished.set()


def test_one_decoder_failure_does_not_stop_other_cameras() -> None:
    async def exercise() -> None:
        # Given: a running coordinator decoding a healthy camera
        coordinator = IngestCoordinator()
        healthy_binding = _new_binding()
        failing_binding = _new_binding()
        healthy_decoder = _ParkedDecoder(healthy_binding)
        failing_decoder = _FailingDecoder(failing_binding)
        coordinator.add(_new_parked_worker(healthy_binding, healthy_decoder))
        stop = anyio.Event()
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(partial(coordinator.run, stop_event=stop))
            with anyio.fail_after(2.0):
                await healthy_decoder.started.wait()

            # When: another camera's decoder raises while the coordinator runs
            coordinator.add(_new_parked_worker(failing_binding, failing_decoder))
            with anyio.fail_after(2.0):
                await failing_decoder.finished.wait()
            await anyio.sleep(0.05)

            # Then: the healthy camera keeps decoding and the failure is reported
            assert not healthy_decoder.finished.is_set()
            assert coordinator.failed_cameras == (failing_binding.camera_id,)
            stop.set()
        assert healthy_decoder.finished.is_set()

    anyio.run(exercise)
