from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast, final

import anyio
import pytest

from gods_watching.appearances import PublicationOutcome
from gods_watching.appearances.pipeline import PublicationStatsSnapshot
from gods_watching.model_selection.models import TransitionRecoveryError
from gods_watching.pipeline_worker import app as worker_app

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass(frozen=True, slots=True)
class _Acknowledgement:
    outcome: PublicationOutcome


@final
class _Consumer:
    def __init__(self, *, pending: int = 2) -> None:
        self.pending = pending
        self.first_drain_started = anyio.Event()
        self.release_first_drain = anyio.Event()
        self.drain_calls = 0

    @property
    def stats(self) -> PublicationStatsSnapshot:
        return PublicationStatsSnapshot(
            pending_embeddings=self.pending,
            last_searchable_latency_seconds=None,
        )

    async def drain_one(self) -> _Acknowledgement | None:
        if self.pending == 0:
            return None
        self.drain_calls += 1
        if self.drain_calls == 1:
            self.first_drain_started.set()
            await self.release_first_drain.wait()
        self.pending -= 1
        return _Acknowledgement(PublicationOutcome.PUBLISHED)


@final
class _GenerationStopState:
    def __init__(self) -> None:
        self.stop_event = anyio.Event()
        self.done_event = anyio.Event()
        self.consumer = _Consumer(pending=3)


@pytest.mark.anyio
async def test_transition_stop_refuses_restart_until_publication_drain_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker_app,
        "_PUBLICATION_SHUTDOWN_DRAIN_SECONDS",
        0.01,
        raising=False,
    )
    generation = _GenerationStopState()
    lifecycle_class = cast("type[object]", vars(worker_app)["_PipelineLifecycle"])
    lifecycle = object.__new__(lifecycle_class)
    vars(lifecycle)["_generation"] = generation
    stop_and_join = cast(
        "Callable[..., Awaitable[None]]",
        vars(lifecycle_class)["stop_and_join"],
    )

    async def finish_later() -> None:
        await anyio.sleep(0.03)
        generation.done_event.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(finish_later)
        with pytest.raises(TransitionRecoveryError, match="publication drain"):
            await stop_and_join(lifecycle)

    assert generation.stop_event.is_set()
    assert vars(lifecycle)["_generation"] is generation
    assert generation.consumer.stats.pending_embeddings == 3


@pytest.mark.anyio
async def test_completed_pipeline_stop_does_not_read_consumer_stats() -> None:
    generation = _GenerationStopState()
    generation.done_event.set()
    vars(generation)["consumer"] = object()
    lifecycle_class = cast("type[object]", vars(worker_app)["_PipelineLifecycle"])
    lifecycle = object.__new__(lifecycle_class)
    vars(lifecycle)["_generation"] = generation
    stop_and_join = cast(
        "Callable[..., Awaitable[None]]",
        vars(lifecycle_class)["stop_and_join"],
    )

    await stop_and_join(lifecycle)

    assert vars(lifecycle)["_generation"] is None


@pytest.mark.anyio
async def test_publication_drainer_finishes_queued_work_after_stop() -> None:
    consumer = _Consumer(pending=2)
    stop_event = anyio.Event()
    drainer_finished = anyio.Event()
    drain_publications = cast(
        "Callable[..., Awaitable[None]]",
        vars(worker_app)["_drain_publications"],
    )

    async def drain() -> None:
        await drain_publications(consumer, stop_event, 0.005)
        drainer_finished.set()

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(drain)
        await consumer.first_drain_started.wait()
        stop_event.set()
        consumer.release_first_drain.set()
        await drainer_finished.wait()

    assert consumer.pending == 0
    assert consumer.drain_calls == 2


@pytest.mark.anyio
async def test_publication_drainer_waits_for_pipeline_terminal_handoffs() -> None:
    consumer = _Consumer(pending=0)
    stop_event = anyio.Event()
    stop_event.set()
    pipeline_done = anyio.Event()
    drainer_done = anyio.Event()
    drain_publications = cast(
        "Callable[..., Awaitable[None]]",
        vars(worker_app)["_drain_publications"],
    )

    async def drain() -> None:
        await drain_publications(consumer, stop_event, 0.005, pipeline_done, drainer_done)

    async with anyio.create_task_group() as task_group:
        task_group.start_soon(drain)
        await anyio.sleep(0.02)
        assert not drainer_done.is_set()
        pipeline_done.set()
        with anyio.fail_after(1):
            await drainer_done.wait()


@pytest.mark.anyio
async def test_shutdown_drain_deadline_logs_remaining_publications(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        worker_app,
        "_PUBLICATION_SHUTDOWN_DRAIN_SECONDS",
        0.01,
        raising=False,
    )
    consumer = _Consumer(pending=3)
    pipeline_done = anyio.Event()
    pipeline_done.set()
    drainer_done = anyio.Event()
    wait_for_drain = cast(
        "Callable[..., Awaitable[bool]]",
        vars(worker_app)["_wait_for_shutdown_drain"],
    )

    with caplog.at_level("WARNING", logger="gods_watching.pipeline_worker.app"):
        drained = await wait_for_drain(pipeline_done, drainer_done, consumer)

    assert not drained
    assert "pending_publications=3" in caplog.text
    assert "pipeline_done=True" in caplog.text
    assert "drainer_done=False" in caplog.text
