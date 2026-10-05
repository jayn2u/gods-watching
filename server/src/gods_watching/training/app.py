"""Separate GPU supervisor application factory, intentionally outside the API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .settings import TrainingSettings
from .supervisor import SubprocessTrainingChildLauncher, TrainingSupervisor

if TYPE_CHECKING:
    import asyncio

    from gods_watching.storage import Database


def build_training_supervisor(
    database: Database,
    settings: TrainingSettings | None = None,
) -> TrainingSupervisor:
    """Compose one fixed runner child and the operator-configured training roots."""
    configured = settings or TrainingSettings()
    return TrainingSupervisor(
        database,
        configured,
        child_launcher=SubprocessTrainingChildLauncher(configured.training_root),
    )


async def run_training_supervisor(
    database: Database,
    *,
    stop_event: asyncio.Event,
    settings: TrainingSettings | None = None,
) -> None:
    """Serve durable requests until shutdown, without sharing the API process."""
    supervisor = build_training_supervisor(database, settings)
    await supervisor.run(stop_event)


__all__ = ["build_training_supervisor", "run_training_supervisor"]
