"""Run the Task 11 production pipeline against owned external services."""

import os
from pathlib import Path
from typing import Literal

import anyio

from gods_watching.inference.clip import TritonClipTransport
from gods_watching.storage import CropObjectStore, Database

from .task11_active_probe import run_active_probe
from .task11_errors import Task11ExecutionError
from .task11_models import Task11DriverEvidence
from .task11_stale_probe import run_stale_probe


def _env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value.strip():
        raise Task11ExecutionError(detail=f"required environment variable is missing: {name}")
    return value


async def _main() -> None:
    mode_value = _env("GW_TASK11_MODE")
    if mode_value not in ("appearance", "appearance-stale"):
        raise Task11ExecutionError(detail=f"unsupported Task 11 mode: {mode_value}")
    mode: Literal["appearance", "appearance-stale"] = mode_value
    database = Database.connect(_env("GW_TASK11_DATABASE_URL"))
    crop_store = CropObjectStore(Path(_env("GW_TASK11_CROP_ROOT")))
    triton_url = _env("GW_TASK11_TRITON_URL")
    try:
        active = await run_active_probe(
            database=database,
            crop_store=crop_store,
            triton_url=triton_url,
            rtsp_url=_env("GW_TASK11_RTSP_URL"),
        )
        stale = None
        if mode == "appearance-stale":
            async with TritonClipTransport(triton_url) as transport:
                stale = await run_stale_probe(
                    database=database,
                    crop_store=crop_store,
                    transport=transport,
                )
        evidence = Task11DriverEvidence(mode=mode, active=active, stale=stale)
        output = Path(_env("GW_TASK11_OUTPUT"))
        _ = output.write_text(evidence.model_dump_json(indent=2) + "\n", encoding="utf-8")
    finally:
        await database.close()


if __name__ == "__main__":
    anyio.run(_main)
