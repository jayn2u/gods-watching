from __future__ import annotations

import shutil
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest
from server.tests.test_model_package_import import package as valid_export
from sqlalchemy import delete

from gods_watching.contracts.training import TrainingConfig
from gods_watching.model_selection.registry import B16_REVISION, DEFAULT_CLIP_MODEL_ID
from gods_watching.storage import Database
from gods_watching.training.evaluation import (
    EvaluationBinding,
    RetrievalScores,
    TrainingEvaluationReport,
)
from gods_watching.training.models import TrainingExecutionSlot, TrainingJob, TrainingPhase
from gods_watching.training.publishing import publish_candidate
from gods_watching.training.repository import TrainingRepository
from gods_watching.training.settings import TrainingSettings
from gods_watching.training.supervisor import TrainingSupervisor

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from uuid import UUID

    from gods_watching.training.publishing import TrainingJobIdentity

_DATASET = {
    "dataset_id": "cuhk-pedes",
    "fingerprint": "a" * 64,
    "protocol": "cuhk-pedes-original-splits-v1",
    "split_counts": {
        "train": {"images": 2, "captions": 4, "identities": 2},
        "val": {"images": 1, "captions": 2, "identities": 1},
        "test": {"images": 1, "captions": 2, "identities": 1},
    },
    "image_count": 4,
    "caption_count": 8,
    "identity_count": 4,
}
_EXPORT_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)


def _evaluation(job: TrainingJob) -> TrainingEvaluationReport:
    return TrainingEvaluationReport(
        binding=EvaluationBinding(
            dataset_sha256=job.dataset_fingerprint,
            dataset_split="test",
            protocol="cuhk-pedes-original-splits-v1",
            source_fingerprint=job.source_fingerprint,
            baseline_model_id=DEFAULT_CLIP_MODEL_ID,
            baseline_revision=B16_REVISION,
            baseline_package_sha256="c" * 64,
            evaluation_code_revision="d" * 64,
        ),
        baseline=RetrievalScores(0.2, 0.4, 0.6),
        candidate=RetrievalScores(0.3, 0.5, 0.7),
        best_validation_epoch=1,
    )


def _exporter(source: Path) -> Callable[[Path, Path, Path, dict[str, object]], None]:
    def export(
        _checkpoint: Path,
        destination: Path,
        _model_root: Path,
        _model_state: dict[str, object],
    ) -> None:
        for name in _EXPORT_FILES:
            _ = shutil.copyfile(source / name, destination / name)

    return export


@pytest.mark.anyio
async def test_publish_retry_after_import_before_candidate_record_is_stable(  # noqa: PLR0915
    database_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = Database.connect(database_url)
    repository = TrainingRepository()
    best_checkpoint = tmp_path / "best.pt"
    _ = best_checkpoint.write_bytes(b"validation-best-checkpoint")
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    source = valid_export(fixture_root)
    assets_root = tmp_path / "assets"

    def load_checkpoint(_path: Path, _identity: Mapping[str, object]) -> dict[str, object]:
        return {
            "identity": {},
            "model": {},
            "training": {"best_epoch": 1},
        }

    monkeypatch.setattr(
        "gods_watching.training.publishing.load_checkpoint_verified",
        load_checkpoint,
    )
    monkeypatch.setattr(
        "gods_watching.training.publishing._export_candidate",
        _exporter(source),
    )
    supervisor = TrainingSupervisor(
        database,
        TrainingSettings(training_root=tmp_path),
        child_launcher=None,
        repository=repository,
    )
    job_id: UUID | None = None
    try:
        async with database.transaction() as session:
            job = await repository.create(session, uuid4(), TrainingConfig(), _DATASET)
            job = await repository.claim_job_owner(
                session,
                job.id,
                expected_generation=job.owner_generation,
            )
            job_id = job.id
            first_generation = job.owner_generation
            assert await repository.set_child_identity(
                session,
                job.id,
                owner_generation=first_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_training_started(
                session,
                job.id,
                owner_generation=first_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.report_training_progress(
                session,
                job.id,
                owner_generation=first_generation,
                pid=2_147_483_647,
                start_time=1,
                epoch=1,
                step=2,
                checkpoint_path=best_checkpoint,
                best_metric=0.5,
            )
            assert await repository.mark_engine_staging(
                session,
                job.id,
                owner_generation=first_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            assert await repository.mark_publishing(
                session,
                job.id,
                owner_generation=first_generation,
                pid=2_147_483_647,
                start_time=1,
            )
            publication_job = await repository.get_job(session, job.id)
            assert publication_job is not None

        report = _evaluation(publication_job)
        first = publish_candidate(
            cast("TrainingJobIdentity", cast("object", publication_job)),
            best_checkpoint,
            report,
            assets_root,
        )

        async with database.transaction() as session:
            await supervisor._recover_orphan_slot(session)  # pyright: ignore[reportPrivateUsage]  # noqa: SLF001

        async with database.transaction() as session:
            interrupted = await repository.get_job(session, job.id)
            slot = await session.get(TrainingExecutionSlot, True)
            assert interrupted is not None
            assert interrupted.phase == TrainingPhase.INTERRUPTED.value
            assert interrupted.candidate_model_id is None
            assert interrupted.evaluation_report is None
            assert slot is not None
            assert slot.active_job_id is None
            resumed = await repository.resume_interrupted(
                session,
                job.id,
                expected_generation=first_generation,
            )
            assert resumed.owner_generation == first_generation + 1

        retry_pid = 12346
        retry_start_time = 67891
        async with database.transaction() as session:
            assert await repository.set_child_identity(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
            )
            assert await repository.mark_training_started(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
            )
            assert await repository.report_training_progress(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
                epoch=1,
                step=2,
                checkpoint_path=best_checkpoint,
                best_metric=0.5,
            )
            assert await repository.mark_engine_staging(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
            )
            assert await repository.mark_publishing(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
            )
            retry_job = await repository.get_job(session, job.id)
            assert retry_job is not None

        retry = publish_candidate(
            cast("TrainingJobIdentity", cast("object", retry_job)),
            best_checkpoint,
            report,
            assets_root,
        )
        assert retry == first

        async with database.transaction() as session:
            assert await repository.record_candidate_publication(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
                candidate_model_id=retry.model_id,
                candidate_revision=retry.revision,
                evaluation=retry.evaluation,
            )
            finished = await repository.finish_child_exit(
                session,
                job.id,
                owner_generation=resumed.owner_generation,
                pid=retry_pid,
                start_time=retry_start_time,
                exit_code=0,
                now=datetime.now(UTC),
            )
            slot = await session.get(TrainingExecutionSlot, True)
            assert finished is not None
            assert finished.phase == TrainingPhase.SUCCEEDED.value
            assert finished.candidate_model_id == first.model_id
            assert finished.candidate_revision == first.revision
            assert finished.evaluation_report == first.evaluation.model_dump(mode="json")
            assert slot is not None
            assert slot.active_job_id is None
    finally:
        async with database.transaction() as session:
            slot = await session.get(TrainingExecutionSlot, True)
            if slot is not None:
                slot.active_job_id = None
            if job_id is not None:
                _ = await session.execute(delete(TrainingJob).where(TrainingJob.id == job_id))
        await database.close()
