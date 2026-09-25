"""Fail-closed eligibility checks for imported CLIP packages."""

import hashlib
import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from gods_watching.model_selection.imported_manifest import ImportedClipManifest, ImportedFile
from gods_watching.model_selection.quality import QualityEvidence, QualityPolicy, assess_quality
from gods_watching.model_selection.registry import DEFAULT_CLIP_MODEL


def _fixture() -> tuple[ImportedClipManifest, QualityEvidence, QualityPolicy]:
    report = {
        "dataset_split": "test",
        "dataset_sha256": "a" * 64,
        "protocol": "identity-disjoint-v1",
        "source_checkpoint": DEFAULT_CLIP_MODEL.model_id,
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
