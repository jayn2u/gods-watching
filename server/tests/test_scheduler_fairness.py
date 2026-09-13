from uuid import uuid4

from gods_watching.contracts.identifiers import CameraId
from gods_watching.ingest.scheduler import FairRoundRobinScheduler


def test_scheduler_admits_due_camera_while_other_camera_is_inflight() -> None:
    slow, healthy = (CameraId(uuid4()) for _ in range(2))
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    scheduler.register(slow)
    scheduler.register(healthy)

    assert scheduler.next_due(now_monotonic=0.0) == slow
    _ = scheduler.mark_dispatched(slow, now_monotonic=0.0)
    assert scheduler.next_due(now_monotonic=0.0) == healthy
    _ = scheduler.mark_dispatched(healthy, now_monotonic=0.0)
    scheduler.complete(healthy)

    assert scheduler.next_due(now_monotonic=0.2) == healthy


def test_scheduler_ignores_completion_from_previous_registration() -> None:
    camera_id = CameraId(uuid4())
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    scheduler.register(camera_id)

    assert scheduler.next_due(now_monotonic=0.0) == camera_id
    old_registration = scheduler.mark_dispatched(camera_id, now_monotonic=0.0)
    assert old_registration is not None
    scheduler.unregister(camera_id)
    scheduler.register(camera_id)

    assert scheduler.next_due(now_monotonic=0.0) == camera_id
    new_registration = scheduler.mark_dispatched(camera_id, now_monotonic=0.0)
    assert new_registration is not None
    assert new_registration != old_registration

    scheduler.complete(camera_id, old_registration)
    assert scheduler.in_flight_count == 1
    scheduler.complete(camera_id, new_registration)
    assert scheduler.in_flight_count == 0


def test_scheduler_deadline_does_not_accumulate_wake_lateness() -> None:
    # Given: one 5 Hz camera whose scheduler wakes 10 ms late on every turn
    camera_id = CameraId(uuid4())
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    scheduler.register(camera_id)

    # When: two late turns complete before the next phase-aligned deadline
    assert scheduler.next_due(now_monotonic=10.01) == camera_id
    registration = scheduler.mark_dispatched(camera_id, now_monotonic=10.01)
    scheduler.complete(camera_id, registration)
    assert scheduler.next_due(now_monotonic=10.21) == camera_id
    registration = scheduler.mark_dispatched(camera_id, now_monotonic=10.21)
    scheduler.complete(camera_id, registration)

    # Then: lateness is not added to every future period
    assert scheduler.next_due(now_monotonic=10.399) is None
    assert scheduler.next_due(now_monotonic=10.4) == camera_id


def test_scheduler_skips_missed_periods_without_catch_up_burst() -> None:
    # Given: one camera that was in flight across several 5 Hz deadlines
    camera_id = CameraId(uuid4())
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    scheduler.register(camera_id)
    registration = scheduler.mark_dispatched(camera_id, now_monotonic=1.0)

    # When: the detector completes late and the camera is dispatched once
    scheduler.complete(camera_id, registration)
    assert scheduler.next_due(now_monotonic=1.81) == camera_id
    registration = scheduler.mark_dispatched(camera_id, now_monotonic=1.81)
    scheduler.complete(camera_id, registration)

    # Then: missed deadlines are skipped and no immediate catch-up turn is due
    assert scheduler.next_due(now_monotonic=1.81) is None
    assert scheduler.next_due(now_monotonic=2.0) == camera_id


def test_scheduler_waits_until_earliest_eligible_camera_deadline() -> None:
    # Given: one 5 Hz camera whose next phase-aligned deadline is in the future
    camera_id = CameraId(uuid4())
    scheduler = FairRoundRobinScheduler(target_fps=5.0)
    scheduler.register(camera_id)
    registration = scheduler.mark_dispatched(camera_id, now_monotonic=10.0)
    scheduler.complete(camera_id, registration)

    # When: the scheduler is polled before that deadline
    wait_seconds = scheduler.seconds_until_next_due(now_monotonic=10.01)

    # Then: it waits for the due phase rather than a fixed polling quantum
    assert abs(wait_seconds - 0.19) < 0.000_001
