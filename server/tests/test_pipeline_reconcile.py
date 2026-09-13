from uuid import uuid4

from gods_watching.cameras import CameraGenerationId
from gods_watching.contracts.identifiers import CameraId, CameraSessionId
from gods_watching.pipeline_worker.reconcile import (
    DesiredCamera,
    RunningCamera,
    plan_reconcile,
)


def _desired(
    camera_id: CameraId | None = None,
    *,
    version: int = 1,
    threshold: float = 0.5,
    session_id: CameraSessionId | None = None,
    generation_id: CameraGenerationId | None = None,
) -> DesiredCamera:
    return DesiredCamera(
        camera_id=camera_id or CameraId(uuid4()),
        version=version,
        session_id=session_id or CameraSessionId(uuid4()),
        generation_id=generation_id or CameraGenerationId(uuid4()),
        source_url="rtsp://camera/test",
        threshold=threshold,
    )


def _running_from(desired: DesiredCamera) -> RunningCamera:
    return RunningCamera(
        camera_id=desired.camera_id,
        version=desired.version,
        session_id=desired.session_id,
        generation_id=desired.generation_id,
        threshold=desired.threshold,
    )


def test_unchanged_cameras_produce_an_empty_plan() -> None:
    # Given: the running worker already matches the committed camera session
    desired = _desired()

    # When: the worker process reconciles
    plan = plan_reconcile({desired.camera_id: desired}, {desired.camera_id: _running_from(desired)})

    # Then: no decoder or tracker is disturbed
    assert plan.stop == ()
    assert plan.start == ()


def test_new_detection_enabled_camera_is_started() -> None:
    # Given: the API committed a camera with an active detection session
    desired = _desired()

    # When: the worker has not seen it yet
    plan = plan_reconcile({desired.camera_id: desired}, {})

    # Then: exactly that camera starts
    assert plan.start == (desired,)
    assert plan.stop == ()


def test_camera_without_an_active_session_is_stopped() -> None:
    # Given: a running camera whose session the API ended (disabled or deleted)
    running = _running_from(_desired())

    # When: the committed state no longer lists it
    plan = plan_reconcile({}, {running.camera_id: running})

    # Then: its worker stops and nothing starts
    assert plan.stop == (running.camera_id,)
    assert plan.start == ()


def test_new_session_for_a_running_camera_replaces_its_worker() -> None:
    # Given: a source edit ended the old session and started a new generation
    old = _desired()
    edited = _desired(old.camera_id, version=old.version + 1)

    # When: the worker still runs the old generation
    plan = plan_reconcile({old.camera_id: edited}, {old.camera_id: _running_from(old)})

    # Then: the old worker stops before the replacement starts
    assert plan.stop == (old.camera_id,)
    assert plan.start == (edited,)


def test_threshold_edit_on_the_same_session_rebinds_the_worker() -> None:
    # Given: a threshold edit bumped the version but kept the same session
    old = _desired(threshold=0.5)
    edited = _desired(
        old.camera_id,
        version=old.version + 1,
        threshold=0.7,
        session_id=old.session_id,
        generation_id=old.generation_id,
    )

    # When: the worker still holds a binding for the previous version
    plan = plan_reconcile({old.camera_id: edited}, {old.camera_id: _running_from(old)})

    # Then: it is replaced, because publication rejects handoffs from a stale version
    assert plan.stop == (old.camera_id,)
    assert plan.start == (edited,)


def test_rename_only_version_bump_also_rebinds_the_worker() -> None:
    # Given: a rename bumped the version without changing session or threshold
    old = _desired()
    renamed = _desired(
        old.camera_id,
        version=old.version + 1,
        threshold=old.threshold,
        session_id=old.session_id,
        generation_id=old.generation_id,
    )

    # When: the worker reconciles
    plan = plan_reconcile({old.camera_id: renamed}, {old.camera_id: _running_from(old)})

    # Then: the worker is rebound to the committed version
    assert plan.stop == (old.camera_id,)
    assert plan.start == (renamed,)


def test_plan_orders_cameras_deterministically() -> None:
    # Given: several cameras to start and stop in one pass
    starting = [_desired() for _ in range(3)]
    stopping = [_running_from(_desired()) for _ in range(3)]

    # When: the worker reconciles
    plan = plan_reconcile(
        {camera.camera_id: camera for camera in starting},
        {camera.camera_id: camera for camera in stopping},
    )

    # Then: both lists follow camera id order so repeated passes act identically
    assert [str(camera.camera_id) for camera in plan.start] == sorted(
        str(camera.camera_id) for camera in starting
    )
    assert [str(camera_id) for camera_id in plan.stop] == sorted(
        str(camera.camera_id) for camera in stopping
    )
