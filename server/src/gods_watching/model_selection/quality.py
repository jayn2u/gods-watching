"""Independent, fail-closed eligibility evidence for imported CLIP packages."""

# ruff: noqa: TC001, TC003, C901, PLR0911, PLR0912, FBT003, PLR2004

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .imported_manifest import ImportedClipManifest
from .registry import BUILTIN_CLIP_MODELS

_HASH = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Deployment-owned expected inputs, independent of candidate evidence."""

    product_cases_sha256: str
    evaluator_revision: str
    baseline_model_id: str
    baseline_revision: str
    cuhk_dataset_sha256: str
    cuhk_protocol: str
    cuhk_evaluation_code_revision: str
    cuhk_metric_definition: str

    @classmethod
    def parse(cls, raw: object) -> QualityPolicy | None:
        """Accept only a complete, pinned policy."""
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            return None
        if not all(isinstance(value, str) and value.strip() for value in raw.values()):
            return None
        if not _HASH.fullmatch(raw["product_cases_sha256"]) or not _HASH.fullmatch(
            raw["cuhk_dataset_sha256"]
        ):
            return None
        if not _HASH.fullmatch(raw["evaluator_revision"]):
            return None
        if not any(
            model.model_id == raw["baseline_model_id"]
            and model.revision == raw["baseline_revision"]
            for model in BUILTIN_CLIP_MODELS
        ):
            return None
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class QualityEvidence:
    """Evaluation record stored outside immutable model weights."""

    package_sha256: str
    baseline_model_id: str
    baseline_revision: str
    cuhk_report_json: str
    product_cases_sha256: str
    text_baseline_recall_at_5: float
    text_candidate_recall_at_5: float
    image_baseline_recall_at_5: float
    image_candidate_recall_at_5: float
    evaluator_revision: str
    created_at: str

    @classmethod
    def parse(cls, raw: object) -> QualityEvidence | None:
        """Reject incomplete, extra, nonfinite, or malformed evidence."""
        if not isinstance(raw, dict) or set(raw) != set(cls.__dataclass_fields__):
            return None
        for key in ("package_sha256", "product_cases_sha256"):
            if not isinstance(raw[key], str) or not _HASH.fullmatch(raw[key]):
                return None
        for key in ("baseline_model_id", "baseline_revision", "evaluator_revision"):
            if not isinstance(raw[key], str) or not raw[key].strip():
                return None
        if not isinstance(raw["cuhk_report_json"], str):
            return None
        try:
            raw["cuhk_report_json"].encode("utf-8")
        except UnicodeEncodeError:
            return None
        for key in (
            "text_baseline_recall_at_5",
            "text_candidate_recall_at_5",
            "image_baseline_recall_at_5",
            "image_candidate_recall_at_5",
        ):
            score = raw[key]
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                return None
        try:
            stamp = datetime.fromisoformat(raw["created_at"])
        except (TypeError, ValueError):
            return None
        if stamp.tzinfo is None:
            return None
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class QualityStatus:
    """Public eligibility result with a bounded reason."""

    passed: bool
    reason: str | None


def assess_quality(
    manifest: ImportedClipManifest,
    evidence: QualityEvidence | None,
    policy: QualityPolicy | None,
) -> QualityStatus:
    """Evaluate immutable report, trusted input bindings, and independent scores."""
    if policy is None:
        return QualityStatus(False, "trusted quality policy missing")
    if evidence is None:
        return QualityStatus(False, "product quality evidence missing")
    if (
        QualityPolicy.parse(asdict(policy)) is None
        or QualityEvidence.parse(asdict(evidence)) is None
    ):
        return QualityStatus(False, "quality policy or evidence invalid")
    if evidence.package_sha256 != manifest.package_sha256:
        return QualityStatus(False, "candidate package hash mismatch")
    if (evidence.baseline_model_id, evidence.baseline_revision) != (
        policy.baseline_model_id,
        policy.baseline_revision,
    ):
        return QualityStatus(False, "baseline identity mismatch")
    if evidence.product_cases_sha256 != policy.product_cases_sha256:
        return QualityStatus(False, "product cases hash mismatch")
    if evidence.evaluator_revision != policy.evaluator_revision:
        return QualityStatus(False, "evaluator revision mismatch")
    report_file = next((item for item in manifest.files if item.path == manifest.cuhk_report), None)
    if report_file is None:
        return QualityStatus(False, "CUHK report missing from package")
    report_bytes = evidence.cuhk_report_json.encode("utf-8")
    if (
        len(report_bytes) != report_file.size
        or hashlib.sha256(report_bytes).hexdigest() != report_file.sha256
    ):
        return QualityStatus(False, "CUHK report hash mismatch")
    try:
        report = json.loads(evidence.cuhk_report_json)
    except ValueError:
        return QualityStatus(False, "CUHK report invalid")
    if not isinstance(report, dict):
        return QualityStatus(False, "CUHK report invalid")
    if report.get("dataset_split") != "test":
        return QualityStatus(False, "CUHK held-out test result required")
    if report.get("dataset_sha256") != policy.cuhk_dataset_sha256:
        return QualityStatus(False, "CUHK dataset hash mismatch")
    if report.get("protocol") != policy.cuhk_protocol:
        return QualityStatus(False, "CUHK protocol mismatch")
    if report.get("evaluation_code_revision") != policy.cuhk_evaluation_code_revision:
        return QualityStatus(False, "CUHK evaluator revision mismatch")
    for key in ("protocol", "source_checkpoint", "evaluation_code_revision", "metric_definition"):
        if not isinstance(report.get(key), str) or not report[key].strip():
            return QualityStatus(False, "CUHK report incomplete")
    if report["source_checkpoint"] != policy.baseline_model_id:
        return QualityStatus(False, "CUHK source checkpoint mismatch")
    if report.get("source_checkpoint_revision") != policy.baseline_revision:
        return QualityStatus(False, "CUHK source checkpoint revision mismatch")
    if report.get("metric_definition") != policy.cuhk_metric_definition:
        return QualityStatus(False, "CUHK metric definition mismatch")
    weight = next((item for item in manifest.files if item.path == "model.safetensors"), None)
    if weight is None or report.get("candidate_weights_sha256") != weight.sha256:
        return QualityStatus(False, "CUHK candidate weights mismatch")
    baseline, candidate = report.get("baseline_score"), report.get("candidate_score")
    if any(
        type(score) not in (int, float) or not math.isfinite(score)
        for score in (baseline, candidate)
    ):
        return QualityStatus(False, "CUHK score invalid")
    if candidate <= baseline:
        return QualityStatus(False, "CUHK held-out score did not improve")
    for modality in ("text", "image"):
        original = getattr(evidence, f"{modality}_baseline_recall_at_5")
        improved = getattr(evidence, f"{modality}_candidate_recall_at_5")
        if improved < 0.8 or improved <= original:
            return QualityStatus(False, f"product {modality} Recall@5 below eligibility threshold")
    return QualityStatus(True, None)


def load_quality_policy(path: Path) -> QualityPolicy | None:
    """Read a deployment-owned immutable policy, failing closed."""
    try:
        return QualityPolicy.parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError):
        return None


def load_quality_evidence(path: Path) -> QualityEvidence | None:
    """Read a per-package evaluation record, failing closed."""
    try:
        return QualityEvidence.parse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, ValueError):
        return None
