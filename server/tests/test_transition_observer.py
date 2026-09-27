"""Transition evidence includes only completed phases and committed crops."""

from pathlib import Path
from typing import cast

import anyio
import pytest

from gods_watching.model_selection.registry import ClipModelPackage, load_clip_registry
from gods_watching.model_selection.transition_observer import PHASES, TransitionObserver
from gods_watching.pipeline_worker import app as worker_app
from gods_watching.pipeline_worker.app import _PipelineLifecycle, compose_one_shot_transition
from gods_watching.pipeline_worker.settings import PipelineWorkerSettings


def test_observer_marks_complete_only_after_all_phases(tmp_path: Path) -> None:
    registry = load_clip_registry(tmp_path)
    package: ClipModelPackage = registry.default
    observer = TransitionObserver()
    observer.begin(package, package)
    for phase in PHASES[:-1]:
        observer.start(phase)
        observer.end(phase)
    observer.finish()
    assert observer.measurement is not None
    assert not observer.measurement.complete
    assert observer.measurement.measured_fixed_seconds is None
    assert observer.measurement.completed_crops == 0
    observer.start(PHASES[-1])
    observer.end(PHASES[-1])
    observer.finish()
    assert observer.measurement.complete
    assert observer.measurement.measured_fixed_seconds is not None


def test_begin_discards_partial_previous_run(tmp_path: Path) -> None:
    package = load_clip_registry(tmp_path).default
    observer = TransitionObserver()
    observer.begin(package, package)
    observer.start("crop_embedding")
    observer.committed_crop(0.0)
    observer.begin(package, package)
    assert observer.measurement is not None
    assert observer.measurement.completed_crops == 0
    assert not observer.measurement.complete


def test_resumed_transition_cannot_be_complete(tmp_path: Path) -> None:
    package = load_clip_registry(tmp_path).default
    observer = TransitionObserver()
    observer.begin(package, package, fresh=False)
    for phase in PHASES:
        observer.start(phase)
        observer.end(phase)
    observer.finish()
    assert observer.measurement is not None
    assert not observer.measurement.complete


def test_clear_removes_previous_success(tmp_path: Path) -> None:
    package = load_clip_registry(tmp_path).default
    observer = TransitionObserver()
    observer.begin(package, package)
    for phase in PHASES:
        observer.start(phase)
        observer.end(phase)
    observer.finish()
    assert observer.measurement is not None
    assert observer.measurement.complete
    observer.clear()
    assert observer.measurement is None


def test_one_shot_composes_real_lifecycle_without_polling(tmp_path: Path) -> None:
    async def inspect() -> None:
        settings = PipelineWorkerSettings.model_construct(
            model_assets_root=tmp_path, triton_grpc_url="localhost:8001"
        )
        resources = [object() for _ in range(8)]
        async with anyio.create_task_group() as task_group:
            runner = compose_one_shot_transition(
                settings=settings,
                database=cast("object", resources[0]),
                storage=cast("object", resources[1]),
                crop_store=cast("object", resources[2]),
                detector_transport=cast("object", resources[3]),
                detector=cast("object", resources[4]),
                registry=load_clip_registry(tmp_path),
                prepared=cast("object", resources[5]),
                coordinator=cast("object", resources[6]),
                runtime=cast("object", resources[7]),
                task_group=task_group,
                quality_policy_path=tmp_path / "quality.json",
            )
            assert type(runner.lifecycle) is _PipelineLifecycle
            assert runner.selection.database is resources[0]
            assert runner.selection.preflight_assets_root == tmp_path
            assert runner.selection.preflight_crop_store is resources[2]
            assert runner.lifecycle.generation is None
            assert runner.runtime is resources[7]

    anyio.run(inspect)


def test_one_shot_joins_restarted_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []

    class Transport:
        async def __aexit__(self, *_args: object) -> None:
            events.append("transport_closed")

    class Selection:
        async def run_pending(self, **_kwargs: object) -> None:
            events.append("transition_done")

    class Lifecycle:
        async def stop_and_join(self) -> None:
            events.append("generation_joined")

    monkeypatch.setattr(worker_app, "TritonClipTransport", lambda *_args: Transport())
    runner = worker_app.OneShotTransition(
        cast("object", Selection()), cast("object", Lifecycle()),
        cast("object", object()), cast("object", object()), "localhost:8001",
    )
    async def run() -> None:
        await runner.run_pending(observer=TransitionObserver())

    anyio.run(run)
    assert events == ["transition_done", "generation_joined", "transport_closed"]
