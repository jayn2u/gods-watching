"""Fail-closed boundary for a future disposable model-transition rehearsal.

This module deliberately does not offer a Docker provisioner until deployment
identity and runtime checks can be implemented. No caller can receive a stack
from the current boundary.
"""

from __future__ import annotations

# ruff: noqa: EM101
import json
import os
import stat
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class RehearsalStackError(ValueError):
    """A stable refusal code for an unverified rehearsal resource."""


@dataclass(slots=True)
class RehearsalInputs:
    """Explicit offline inputs; none defaults to a production resource."""

    database_dump: Path
    crop_snapshot: Path
    assets: Path
    gpu_uuid: str
    target_model_id: str


@dataclass(frozen=True, slots=True)
class RehearsalStack:
    """The verified handles required by a rehearsal runner."""

    database_url: str
    crops_root: Path
    triton_url: str
    gpu_uuid: str
    postgres_container_id: str
    triton_container_id: str


def _directory(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and os.access(path, os.R_OK | os.X_OK)
    except OSError:
        return False


def validate_rehearsal_inputs(inputs: RehearsalInputs) -> None:  # noqa: C901
    """Reject absent, aliased, or inconsistent inputs before Docker access."""
    try:
        dump_mode = inputs.database_dump.lstat().st_mode
    except OSError as error:
        raise RehearsalStackError("database_dump_unavailable") from error
    if not stat.S_ISREG(dump_mode) or not os.access(inputs.database_dump, os.R_OK):
        raise RehearsalStackError("database_dump_unavailable")
    if not _directory(inputs.crop_snapshot):
        raise RehearsalStackError("crop_snapshot_unavailable")
    if not _directory(inputs.assets):
        raise RehearsalStackError("assets_unavailable")
    live_crops = os.environ.get("GW_CROPS_ROOT")
    if live_crops and inputs.crop_snapshot.resolve() == Path(live_crops).resolve():
        raise RehearsalStackError("live_crop_alias")
    live_assets = os.environ.get("GW_MODELS_ROOT")
    if live_assets and inputs.assets.resolve() == Path(live_assets).resolve():
        raise RehearsalStackError("live_assets_alias")
    if not inputs.gpu_uuid.startswith("GPU-") or not inputs.target_model_id.strip():
        raise RehearsalStackError("invalid_rehearsal_identity")
    manifest = inputs.assets / "prepared-manifest.json"
    try:
        if not stat.S_ISREG(manifest.lstat().st_mode):
            raise RehearsalStackError("preparation_manifest_unavailable")
        prepared = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RehearsalStackError("preparation_manifest_unavailable") from error
    if not isinstance(prepared, dict) or prepared.get("cuda_device_uuid") != inputs.gpu_uuid:
        raise RehearsalStackError("gpu_uuid_mismatch")


@asynccontextmanager
async def isolated_rehearsal_stack(inputs: RehearsalInputs) -> AsyncIterator[RehearsalStack]:
    """Refuse until a deployment-attested, egress-isolated provisioner exists."""
    validate_rehearsal_inputs(inputs)
    raise RehearsalStackError("provisioner_unavailable")
    yield  # pragma: no cover
