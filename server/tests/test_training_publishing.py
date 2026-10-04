from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast
from uuid import UUID

import pytest
from server.tests.test_model_package_import import package as valid_export

from gods_watching.model_selection.registry import B16_REVISION
from gods_watching.training.evaluation import (
    EvaluationBinding,
    RetrievalScores,
    TrainingEvaluationReport,
)
from gods_watching.training.publishing import (
    CandidateIdentity,
    TrainingPublicationError,
    publish_candidate,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class _Job:
    id: UUID
    config_snapshot: dict[str, object]
    dataset_fingerprint: str
    source_fingerprint: str
    candidate_model_id: str | None = None


def _job() -> _Job:
    return _Job(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        config_snapshot={"epochs": 2, "micro_batch_size": 2},
        dataset_fingerprint="a" * 64,
        source_fingerprint="b" * 64,
    )


def _report() -> TrainingEvaluationReport:
    return TrainingEvaluationReport(
        binding=EvaluationBinding(
            dataset_sha256="a" * 64,
            dataset_split="test",
            protocol="cuhk-pedes-original-splits-v1",
            source_fingerprint="b" * 64,
            baseline_model_id="openai/clip-vit-base-patch16",
            baseline_revision=B16_REVISION,
            baseline_package_sha256="f" * 64,
            evaluation_code_revision="d" * 64,
            metric_definition="cuhk-pedes-text-to-image-macro-recall-at-1-primary-r5-r10-detail-v2",
        ),
        baseline=RetrievalScores(recall_at_1=0.2, recall_at_5=0.4, recall_at_10=0.6),
        candidate=RetrievalScores(recall_at_1=0.3, recall_at_5=0.5, recall_at_10=0.7),
        best_validation_epoch=1,
    )


def _payload_exporter(
    template_root: Path,
    *,
    omit: str | None = None,
) -> Callable[[Path, Path, Path, dict[str, object]], None]:
    template_root.mkdir(parents=True, exist_ok=True)
    payload = valid_export(template_root)
    core_files = {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    }

    def export(
        _checkpoint: Path,
        destination: Path,
        _model_root: Path,
        _model_state: dict[str, object],
    ) -> None:
        for name in core_files - {omit} if omit is not None else core_files:
            _ = shutil.copyfile(payload / name, destination / name)

    return export


def test_publication_refuses_an_incomplete_model_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _ = checkpoint.write_bytes(b"best-validation-checkpoint")
    assets = tmp_path / "assets"
    template = tmp_path / "template"
    exporter = _payload_exporter(template, omit="model.safetensors")
    def load_checkpoint(_path: Path, _identity: Mapping[str, object]) -> dict[str, object]:
        return {
            "identity": _job().config_snapshot,
            "model": {},
            "training": {"best_epoch": 1},
        }

    monkeypatch.setattr(
        "gods_watching.training.publishing.load_checkpoint_verified",
        load_checkpoint,
    )
    monkeypatch.setattr("gods_watching.training.publishing._export_candidate", exporter)

    with pytest.raises(TrainingPublicationError, match="incomplete"):
        _ = publish_candidate(_job(), checkpoint, _report(), assets)

    assert not (assets / "imported").exists()


def test_publication_retry_is_idempotent_and_report_binds_exported_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "best.pt"
    _ = checkpoint.write_bytes(b"best-validation-checkpoint")
    assets = tmp_path / "assets"
    template = tmp_path / "template"
    def load_checkpoint(_path: Path, _identity: Mapping[str, object]) -> dict[str, object]:
        return {
            "identity": _job().config_snapshot,
            "model": {},
            "training": {"best_epoch": 1},
        }

    monkeypatch.setattr(
        "gods_watching.training.publishing.load_checkpoint_verified",
        load_checkpoint,
    )
    monkeypatch.setattr(
        "gods_watching.training.publishing._export_candidate",
        _payload_exporter(template),
    )

    first = publish_candidate(_job(), checkpoint, _report(), assets)
    second = publish_candidate(_job(), checkpoint, _report(), assets)

    assert isinstance(first, CandidateIdentity)
    assert second == first
    assert first.model_id == "local/cuhk-pedes-12345678123456781234567812345678"
    package_root = assets / "imported" / first.package_sha256
    manifest_object = cast(
        "object",
        json.loads((package_root / "manifest.json").read_text(encoding="utf-8")),
    )
    assert isinstance(manifest_object, dict)
    manifest = cast("dict[str, object]", manifest_object)
    report_name = manifest.get("cuhk_report")
    assert isinstance(report_name, str)
    report_bytes = (package_root / report_name).read_bytes()
    report_object = cast("object", json.loads(report_bytes))
    assert isinstance(report_object, dict)
    report = cast("dict[str, object]", report_object)
    weights_sha256 = hashlib.sha256((package_root / "model.safetensors").read_bytes()).hexdigest()
    assert report["candidate_weights_sha256"] == weights_sha256
    assert report["dataset_split"] == "test"
    assert report["baseline_score"] == 0.2
    assert report["candidate_score"] == 0.3
    assert report["baseline_text_to_image"] == {
        "recall_at_1": 0.2,
        "recall_at_5": 0.4,
        "recall_at_10": 0.6,
    }
    assert report["candidate_text_to_image"] == {
        "recall_at_1": 0.3,
        "recall_at_5": 0.5,
        "recall_at_10": 0.7,
    }
    assert first.evaluation.dataset_sha256 == "a" * 64
    assert first.evaluation.best_validation_epoch == 1
    assert first.evaluation.package_sha256 == first.package_sha256
    assert len(list((assets / "imported").glob("*/manifest.json"))) == 1
