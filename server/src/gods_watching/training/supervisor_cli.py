"""Signal-aware entrypoint for the isolated CUDA training supervisor."""

from __future__ import annotations

import asyncio
import signal
import sys

from pydantic import ValidationError

from gods_watching.storage import Database

from .app import run_training_supervisor
from .settings import TrainingSettings


def _settings_error(error: ValidationError) -> str:
    fields = sorted({".".join(str(part) for part in item["loc"]) for item in error.errors()})
    return f"training supervisor settings are invalid: {', '.join(fields)}"


async def _serve(database: Database, settings: TrainingSettings) -> None:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop_event.set)
    try:
        await run_training_supervisor(database, stop_event=stop_event, settings=settings)
    finally:
        await database.close()


def main() -> int:
    """Validate settings without secrets in output, then serve until shutdown."""
    try:
        settings = TrainingSettings()
    except ValidationError as error:
        _ = sys.stderr.write(_settings_error(error) + "\n")
        return 2
    if settings.database_url is None:
        _ = sys.stderr.write("training supervisor requires GW_DATABASE_URL\n")
        return 2
    database = Database.connect(settings.database_url)
    asyncio.run(_serve(database, settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
