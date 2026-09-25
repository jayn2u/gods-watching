"""Fail-closed eligibility checks for imported CLIP packages."""

# ruff: noqa: SLF001

import hashlib
import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from gods_watching.model_selection.assets import PreparedModelStatus
from gods_watching.model_selection.imported_manifest import ImportedClipManifest, ImportedFile
from gods_watching.model_selection.quality import QualityEvidence, QualityPolicy, assess_quality
from gods_watching.model_selection.registry import (
    DEFAULT_CLIP_MODEL,
    ClipModelPackage,
    ClipModelRegistry,
)
from gods_watching.model_selection.service import ModelSelectionService

if TYPE_CHECKING:
    from gods_watching.storage import Database


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _fixture() -> tuple[ImportedClipManifest, QualityEvidence, QualityPolicy]:
    report = {
        "dataset_split": "test",
        "dataset_sha256": "a" * 64,
        "protocol": "identity-disjoint-v1",
        "source_checkpoint": DEFAULT_CLIP_MODEL.model_id,
        "source_checkpoint_revision": DEFAULT_CLIP_MODEL.revision,
        "candidate_weights_sha256": "f" * 64,
        "evaluation_code_revision": "cuhk-eval-v1",
        "metric_definition": "Recall@1",
        "baseline_score": 0.5,
        "candidate_score": 0.6,
    }
    report_json = json.dumps(report)
    manifest = ImportedClipManifest(
        model_id="test/fine-tuned",
        revision="b" * 64,
        display_name="candidate",
        base_model_id=DEFAULT_CLIP_MODEL.model_id,
        dimension=512,
        files=(
            ImportedFile(
                "cuhk.json", len(report_json), hashlib.sha256(report_json.encode()).hexdigest()
            ),
            ImportedFile("model.safetensors", 1, "f" * 64),
        ),
        package_sha256="b" * 64,
        cuhk_report="cuhk.json",
    )
    policy = QualityPolicy(
        product_cases_sha256="c" * 64,
        evaluator_revision="d" * 64,
        baseline_model_id=DEFAULT_CLIP_MODEL.model_id,
        baseline_revision=DEFAULT_CLIP_MODEL.revision,
        cuhk_dataset_sha256="a" * 64,
        cuhk_protocol="identity-disjoint-v1",
        cuhk_evaluation_code_revision="cuhk-eval-v1",
        cuhk_metric_definition="Recall@1",
    )
    evidence = QualityEvidence(
        package_sha256=manifest.package_sha256,
        baseline_model_id=policy.baseline_model_id,
        baseline_revision=policy.baseline_revision,
        cuhk_report_json=report_json,
        product_cases_sha256=policy.product_cases_sha256,
        text_baseline_recall_at_5=0.7,
        text_candidate_recall_at_5=0.8,
        image_baseline_recall_at_5=0.7,
        image_candidate_recall_at_5=0.8,
        evaluator_revision=policy.evaluator_revision,
        created_at="2026-09-25T00:00:00+00:00",
    )
    return manifest, evidence, policy


def test_complete_evidence_passes() -> None:
    assert assess_quality(*_fixture()).passed


@pytest.mark.parametrize(
    "field", ["candidate_weights_sha256", "metric_definition", "source_checkpoint_revision"]
)
def test_cuhk_report_candidate_and_protocol_binding(field: str) -> None:
    manifest, evidence, policy = _fixture()
    report = json.loads(evidence.cuhk_report_json)
    report[field] = "other"
    report_json = json.dumps(report)
    evidence = replace(evidence, cuhk_report_json=report_json)
    manifest = replace(
        manifest,
        files=(
            ImportedFile(
                "cuhk.json", len(report_json), hashlib.sha256(report_json.encode()).hexdigest()
            ),
            manifest.files[1],
        ),
    )
    assert not assess_quality(manifest, evidence, policy).passed


@pytest.mark.parametrize(
    "change",
    [
        "train",
        "package",
        "cases",
        "baseline",
        "evaluator",
        "cuhk_dataset",
        "cuhk_protocol",
        "cuhk_evaluator",
        "text",
        "image",
    ],
)
def test_stale_or_failing_evidence_rejected(change: str) -> None:
    manifest, evidence, policy = _fixture()
    if change == "train":
        report = json.loads(evidence.cuhk_report_json)
        report["dataset_split"] = "train"
        evidence = replace(evidence, cuhk_report_json=json.dumps(report))
        manifest = replace(
            manifest,
            files=(
                ImportedFile(
                    "cuhk.json",
                    len(evidence.cuhk_report_json),
                    hashlib.sha256(evidence.cuhk_report_json.encode()).hexdigest(),
                ),
            ),
        )
    elif change == "package":
        manifest = replace(manifest, package_sha256="e" * 64)
    elif change == "cases":
        evidence = replace(evidence, product_cases_sha256="e" * 64)
    elif change == "baseline":
        evidence = replace(evidence, baseline_revision="e" * 64)
    elif change == "evaluator":
        evidence = replace(evidence, evaluator_revision="e" * 64)
    elif change == "cuhk_dataset":
        policy = replace(policy, cuhk_dataset_sha256="e" * 64)
    elif change == "cuhk_protocol":
        policy = replace(policy, cuhk_protocol="changed")
    elif change == "cuhk_evaluator":
        policy = replace(policy, cuhk_evaluation_code_revision="changed")
    elif change == "text":
        evidence = replace(evidence, text_candidate_recall_at_5=0.79)
    else:
        evidence = replace(evidence, image_candidate_recall_at_5=0.7)
    assert not assess_quality(manifest, evidence, policy).passed


def test_policy_rejects_missing_trusted_fields() -> None:
    assert QualityPolicy.parse({}) is None


def test_evidence_rejects_nonfinite_and_naive_time() -> None:
    manifest, evidence, policy = _fixture()
    raw = asdict(evidence)
    raw["text_candidate_recall_at_5"] = float("nan")
    assert QualityEvidence.parse(raw) is None
    assert not assess_quality(
        manifest, replace(evidence, text_candidate_recall_at_5=float("nan")), policy
    ).passed
    raw["text_candidate_recall_at_5"] = 0.8
    raw["created_at"] = "2026-09-25T00:00:00"
    assert QualityEvidence.parse(raw) is None


def test_product_cases_require_exact_fixed_ids_and_crop_bytes(tmp_path: Path) -> None:
    cases_path = Path(__file__).resolve().parents[2] / "qa/clip/evaluate_product_cases.py"
    validate_cases = runpy.run_path(str(cases_path))["_cases"]
    appearances = []
    for index in range(40):
        crop = tmp_path / f"crop-{index}.jpg"
        crop.write_bytes(f"real-crop-{index}".encode())
        appearances.append(
            {
                "id": f"appearance-{index}",
                "crop": crop.name,
                "scene": f"scene-{index % 2}",
                "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
            }
        )
    image_queries = []
    for index in range(20):
        crop = tmp_path / f"query-{index}.jpg"
        crop.write_bytes(f"held-out-crop-{index}".encode())
        image_queries.append(
            {
                "id": f"image-{index}",
                "crop": crop.name,
                "sha256": hashlib.sha256(crop.read_bytes()).hexdigest(),
                "relevant_ids": [f"appearance-{index + 20}"],
            }
        )
    cases = {
        "appearances": appearances,
        "text_queries": [
            {
                "id": f"text-{index}",
                "text": "person in blue",
                "relevant_ids": [f"appearance-{index}"],
            }
            for index in range(20)
        ],
        "image_queries": image_queries,
    }
    path = tmp_path / "retrieval-cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert validate_cases(path, digest) == cases
    cases["text_queries"][1]["id"] = cases["text_queries"][0]["id"]
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_cases(path, digest)
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(path, hashlib.sha256(path.read_bytes()).hexdigest())
    (tmp_path / "crop-0.jpg").write_bytes(b"altered")
    with pytest.raises(ValueError, match="crop hash mismatch"):
        validate_cases(path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_baseline_snapshot_must_match_trusted_lock(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "qa/clip/evaluate_product_cases.py"
    verify_baseline = runpy.run_path(str(script))["_verify_baseline"]
    _, _, policy = _fixture()
    baseline = tmp_path / "clip"
    baseline.mkdir()
    (baseline / "config.json").write_bytes(b"trusted config")
    (baseline / "pytorch_model.bin").write_bytes(b"trusted weights")
    (baseline / "gods-watching-model.json").write_text(
        json.dumps(
            {
                "model_id": policy.baseline_model_id,
                "revision": policy.baseline_revision,
                "dimension": DEFAULT_CLIP_MODEL.dimension,
                "processor": DEFAULT_CLIP_MODEL.processor,
                "runtime": DEFAULT_CLIP_MODEL.runtime,
            }
        ),
        encoding="utf-8",
    )
    files = [
        {
            "path": f"clip/{name}",
            "sha256": hashlib.sha256((baseline / name).read_bytes()).hexdigest(),
            "size": (baseline / name).stat().st_size,
        }
        for name in ("config.json", "pytorch_model.bin")
    ]
    lock_path = tmp_path / "models.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "container": {
                    "base_image": "test",
                    "base_digest": "sha256:test",
                    "built_image": "test",
                    "python_abi": "cp312",
                },
                "models": [
                    {
                        "model_id": policy.baseline_model_id,
                        "revision": policy.baseline_revision,
                        "license": "MIT",
                        "source": "test",
                        "files": files,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    verify_baseline(baseline, policy, lock_path)
    (baseline / "config.json").write_bytes(b"altered processor config")
    with pytest.raises(ValueError, match="baseline"):
        verify_baseline(baseline, policy, lock_path)
    (baseline / "config.json").write_bytes(b"trusted config")
    (baseline / "pytorch_model.bin").write_bytes(b"altered weights")
    with pytest.raises(ValueError, match="baseline"):
        verify_baseline(baseline, policy, lock_path)
    (baseline / "pytorch_model.bin").write_bytes(b"trusted weights")
    wrong_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    wrong_lock["models"][0]["revision"] = "untrusted-revision"
    lock_path.write_text(json.dumps(wrong_lock), encoding="utf-8")
    with pytest.raises(ValueError, match="baseline identity"):
        verify_baseline(baseline, policy, lock_path)
    wrong = tmp_path / "different" / "clip"
    wrong.mkdir(parents=True)
    for name in ("config.json", "pytorch_model.bin"):
        (wrong / name).write_bytes((baseline / name).read_bytes())
    with pytest.raises(ValueError, match="baseline"):
        verify_baseline(tmp_path / "different", policy, lock_path)


def test_malformed_cuhk_unicode_fails_closed() -> None:
    manifest, evidence, policy = _fixture()
    malformed = replace(evidence, cuhk_report_json="\ud800")
    assert not assess_quality(manifest, malformed, policy).passed


@pytest.mark.anyio
async def test_catalog_survives_malformed_imported_report(tmp_path: Path) -> None:
    manifest, evidence, policy = _fixture()
    imported_root = tmp_path / "imported"
    package_dir = imported_root / manifest.package_sha256
    package_dir.mkdir(parents=True)
    (package_dir / "manifest.json").write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    evidence_root = tmp_path / "quality-evidence"
    evidence_root.mkdir()
    (evidence_root / f"{manifest.package_sha256}.json").write_text(
        json.dumps(asdict(replace(evidence, cuhk_report_json="\ud800"))), encoding="utf-8"
    )
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(asdict(policy)), encoding="utf-8")
    imported = ClipModelPackage(
        model_id=manifest.model_id,
        revision=manifest.revision,
        snapshot_path=Path("/models/imported") / manifest.package_sha256,
        dimension=512,
        processor="CLIPProcessor",
        runtime="transformers",
    )

    class Prepared:
        def status(self, package: ClipModelPackage) -> PreparedModelStatus:
            del package
            return PreparedModelStatus(prepared=True)

    service = ModelSelectionService(
        database=cast("Database", object()),
        registry=ClipModelRegistry((DEFAULT_CLIP_MODEL, imported)),
        prepared=Prepared(),
        imported_assets_root=imported_root,
        quality_policy_path=policy_path,
        quality_evidence_root=evidence_root,
    )
    catalog = await service._response(DEFAULT_CLIP_MODEL.model_id, None)
    assert catalog.models[0].quality_passed
    assert not catalog.models[1].quality_passed
