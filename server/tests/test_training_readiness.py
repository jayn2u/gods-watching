from __future__ import annotations

import asyncio
import json
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, cast

import pytest
from PIL import Image

from gods_watching.contracts.training import TrainingConfig, TrainingSupervisorStatus
from gods_watching.training.dataset import DatasetManifest, DatasetSplitCounts, validate_cuhk
from gods_watching.training.readiness import read_supervisor_status, write_supervisor_status
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.service import (
    TrainingDatasetUnavailableError,
    TrainingService,
)
from gods_watching.training.settings import TrainingSettings

if TYPE_CHECKING:
    from pathlib import Path

    from gods_watching.storage import Database


def _manifest(root: Path) -> DatasetManifest:
    return DatasetManifest(
        dataset_id="cuhk-pedes",
        fingerprint="a" * 64,
        protocol="cuhk-pedes-original-splits-v1",
        root=root,
        samples=(),
        split_counts={
            "train": DatasetSplitCounts(images=2, captions=2, identities=2),
            "val": DatasetSplitCounts(images=1, captions=1, identities=1),
            "test": DatasetSplitCounts(images=1, captions=1, identities=1),
        },
    )


def test_supervisor_status_is_bound_to_source_dataset_and_freshness(tmp_path: Path) -> None:
    repository = TrainingRepository()
    observed_at = datetime.now(UTC)
    status = TrainingSupervisorStatus(
        state="ready",
        observed_at=observed_at,
        source_fingerprint=repository.source_fingerprint,
        dataset_fingerprint="a" * 64,
    )
    write_supervisor_status(tmp_path, status)

    assert (
        read_supervisor_status(
            tmp_path,
            source_fingerprint=repository.source_fingerprint,
            dataset_fingerprint="a" * 64,
            now=observed_at,
        ).state
        == "ready"
    )
    assert (
        read_supervisor_status(
            tmp_path,
            source_fingerprint="b" * 64,
            dataset_fingerprint="a" * 64,
            now=observed_at,
        ).reason
        == "training_supervisor_source_changed"
    )
    assert (
        read_supervisor_status(
            tmp_path,
            source_fingerprint=repository.source_fingerprint,
            dataset_fingerprint="c" * 64,
            now=observed_at,
        ).reason
        == "training_supervisor_dataset_changed"
    )
    assert (
        read_supervisor_status(
            tmp_path,
            source_fingerprint=repository.source_fingerprint,
            dataset_fingerprint="a" * 64,
            now=observed_at + timedelta(seconds=16),
        ).reason
        == "training_supervisor_status_stale"
    )


def test_naive_supervisor_heartbeat_is_unavailable_instead_of_raising(tmp_path: Path) -> None:
    repository = TrainingRepository()
    status_path = tmp_path / "supervisor-status.json"
    _ = status_path.write_text(
        json.dumps(
            {
                "state": "ready",
                "observed_at": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "source_fingerprint": repository.source_fingerprint,
                "dataset_fingerprint": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    status = read_supervisor_status(
        tmp_path,
        source_fingerprint=repository.source_fingerprint,
        dataset_fingerprint="a" * 64,
    )

    assert status.state == "unavailable"
    assert status.reason == "training_supervisor_unavailable"


def test_api_reports_typed_supervisor_validation_without_waiting_for_full_scan(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        dataset_root = tmp_path / "dataset"
        training_root = tmp_path / "runs"
        manifest = _manifest(dataset_root)
        repository = TrainingRepository()
        write_supervisor_status(
            training_root,
            TrainingSupervisorStatus(
                state="validating",
                reason="dataset_validating",
                observed_at=datetime.now(UTC),
                source_fingerprint=repository.source_fingerprint,
            ),
        )
        validator_started = threading.Event()
        validator_release = threading.Event()
        validator_calls = 0

        def slow_validator(_root: Path) -> DatasetManifest:
            nonlocal validator_calls
            validator_calls += 1
            validator_started.set()
            assert validator_release.wait(timeout=3)
            return manifest

        service = TrainingService(
            cast("Database", object()),
            TrainingSettings(dataset_root=dataset_root, training_root=training_root),
            dataset_validator=slow_validator,
        )

        status = await service.datasets()
        assert status.registered
        assert not status.valid
        assert status.reason == "dataset_validating"
        assert status.supervisor.state == "validating"
        assert status.supervisor.source_fingerprint == repository.source_fingerprint

        assert await asyncio.to_thread(validator_started.wait, 1)
        joined_warmup = asyncio.create_task(service.warm_dataset())
        assert validator_calls == 1
        validator_release.set()
        await joined_warmup
        assert validator_calls == 1

    asyncio.run(scenario())


def test_api_reports_missing_dataset_as_unavailable_without_settings_crash(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        service = TrainingService(
            cast("Database", object()),
            TrainingSettings(dataset_root=None, training_root=tmp_path),
        )

        status = await service.datasets()

        assert not status.registered
        assert not status.valid
        assert status.reason == "dataset_not_configured"
        assert status.supervisor.state == "unavailable"

    asyncio.run(scenario())


def test_ready_dataset_change_returns_validating_and_rewarms_off_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "dataset"
        records = [
            {"split": "train", "id": 1, "file_path": "train/a.png", "captions": ["one"]},
            {"split": "train", "id": 2, "file_path": "train/b.png", "captions": ["two"]},
            {"split": "val", "id": 3, "file_path": "val/a.png", "captions": ["val"]},
            {"split": "test", "id": 4, "file_path": "test/a.png", "captions": ["test"]},
        ]
        for row in records:
            image_path = root / "imgs" / str(row["file_path"])
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (4, 3), (16, 32, 48)).save(image_path, format="PNG")
        _ = (root / "reid_raw.json").write_text(json.dumps(records), encoding="utf-8")
        prior_fingerprint = (await asyncio.to_thread(validate_cuhk, root)).fingerprint

        validation_started = threading.Event()
        validation_release = threading.Event()
        validation_calls = 0

        def validator(dataset_root: Path) -> DatasetManifest:
            nonlocal validation_calls
            validation_calls += 1
            if validation_calls == 2:
                validation_started.set()
                assert validation_release.wait(timeout=3)
            return validate_cuhk(dataset_root)

        service = TrainingService(
            cast("Database", object()),
            TrainingSettings(dataset_root=root, training_root=tmp_path / "runs"),
            dataset_validator=validator,
        )
        await service.warm_dataset()

        changed_image = root / "imgs" / "train/a.png"
        Image.new("RGB", (4, 3), (128, 32, 48)).save(changed_image, format="PNG")
        dataset_request = asyncio.create_task(service.datasets())
        try:
            refreshed = await asyncio.wait_for(asyncio.shield(dataset_request), timeout=0.5)
        except TimeoutError:
            assert await asyncio.to_thread(validation_started.wait, 1)
            validation_release.set()
            _ = await dataset_request
            pytest.fail("changed source triggered a full validation in the status request")

        assert not refreshed.valid
        assert refreshed.reason == "dataset_validating"
        assert refreshed.snapshot is None
        assert await asyncio.to_thread(validation_started.wait, 1)
        assert validation_calls == 2
        with pytest.raises(TrainingDatasetUnavailableError):
            _ = await service.preflight(TrainingConfig())

        validation_release.set()
        await service.warm_dataset()
        validated_again = await service.datasets()
        assert validated_again.valid
        assert validated_again.snapshot is not None
        assert validated_again.snapshot.fingerprint != prior_fingerprint
        assert validation_calls == 2

    asyncio.run(scenario())
