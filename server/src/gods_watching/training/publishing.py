"""Strict export and idempotent publication of one trained CLIP candidate."""

# ruff: noqa: EM101, TC001, TC003, D107

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from gods_watching.contracts.training import (
    TrainingEvaluationSummary,
    TrainingRetrievalScores,
)
from gods_watching.model_selection.imported_manifest import ClipPackageImportError
from gods_watching.model_selection.importer import import_clip_package
from gods_watching.model_selection.registry import B16_REVISION, DEFAULT_CLIP_MODEL_ID
from gods_watching.training.calibration import builtin_base_load_options
from gods_watching.training.checkpoints import load_checkpoint_verified
from gods_watching.training.evaluation import TrainingEvaluationReport

if TYPE_CHECKING:
    from uuid import UUID

_PUBLISHED_PAYLOAD_FILES = frozenset(
    {
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    }
)
_REPORT_FILENAME = "cuhk-report.json"
_PACKAGE_FILENAME = "package.json"
_CLIP_DIMENSION = 512


class TrainingPublicationError(RuntimeError):
    """A candidate export was incomplete or did not fit the local importer contract."""

    code: str

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Content-addressed publication identity and its safe API evaluation summary."""

    model_id: str
    revision: str
    package_sha256: str
    candidate_weights_sha256: str
    evaluation: TrainingEvaluationSummary


class _SavePretrainedModel(Protocol):
    def load_state_dict(self, state: Mapping[str, object]) -> None: ...

    def save_pretrained(
        self,
        save_directory: Path,
        *,
        safe_serialization: bool,
    ) -> None: ...


class _SavePretrainedProcessor(Protocol):
    def save_pretrained(self, save_directory: Path) -> None: ...


class _ModelLoader(Protocol):
    def from_pretrained(
        self,
        model_path: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
        **options: object,
    ) -> _SavePretrainedModel: ...


class _ProcessorLoader(Protocol):
    def from_pretrained(
        self,
        model_path: str,
        *,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> _SavePretrainedProcessor: ...


class _TransformersApi(Protocol):
    CLIPModel: _ModelLoader
    CLIPProcessor: _ProcessorLoader


class TrainingJobIdentity(Protocol):
    """Immutable job fields required to bind and name one publication."""

    @property
    def id(self) -> UUID: ...

    @property
    def config_snapshot(self) -> dict[str, object]: ...

    @property
    def dataset_fingerprint(self) -> str: ...

    @property
    def source_fingerprint(self) -> str: ...

    @property
    def candidate_model_id(self) -> str | None: ...


def publish_candidate(
    job: TrainingJobIdentity,
    best_checkpoint: Path,
    report: TrainingEvaluationReport,
    assets_root: Path,
) -> CandidateIdentity:
    """Export best-validation weights, strictly import the package, and return stable IDs."""
    _validate_publication_binding(job, report)
    checkpoint_identity = {
        "config_snapshot": job.config_snapshot,
        "dataset_fingerprint": job.dataset_fingerprint,
        "source_fingerprint": job.source_fingerprint,
    }
    try:
        checkpoint = load_checkpoint_verified(best_checkpoint, checkpoint_identity)
    except Exception as error:
        raise TrainingPublicationError("best_checkpoint_invalid") from error
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise TrainingPublicationError("best_checkpoint_model_missing")
    raw_training = checkpoint.get("training")
    if not isinstance(raw_training, dict):
        raise TrainingPublicationError("best_checkpoint_epoch_mismatch")
    training = cast("dict[str, object]", raw_training)
    if training.get("best_epoch") != report.best_validation_epoch:
        raise TrainingPublicationError("best_checkpoint_epoch_mismatch")

    assets_root = Path(assets_root)
    try:
        assets_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        _require_directory(assets_root)
        with tempfile.TemporaryDirectory(prefix=".cuhk-candidate-", dir=assets_root) as temporary:
            stage = Path(temporary)
            _export_candidate(
                best_checkpoint,
                stage,
                Path(os.environ.get("GW_TRAINING_MODEL_ROOT", "/models/clip")),
                cast("dict[str, object]", model_state),
            )
            weights_path = _validate_export(stage)
            weights_sha256 = _sha256(weights_path)
            report_value = report.package_report(weights_sha256)
            report_path = stage / _REPORT_FILENAME
            _write_durable(
                report_path,
                json.dumps(
                    report_value,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            files = _file_records(stage)
            candidate_model_id = _candidate_model_id(job)
            package_value = {
                "model_id": candidate_model_id,
                "display_name": f"CUHK-PEDES CLIP {job.id.hex[:8]}",
                "base_model_id": DEFAULT_CLIP_MODEL_ID,
                "dimension": _CLIP_DIMENSION,
                "files": files,
                "cuhk_report": _REPORT_FILENAME,
            }
            _write_durable(
                stage / _PACKAGE_FILENAME,
                json.dumps(
                    package_value,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
            )
            try:
                manifest = import_clip_package(stage, assets_root)
            except ClipPackageImportError as error:
                raise TrainingPublicationError(error.code) from error
    except TrainingPublicationError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise TrainingPublicationError("candidate_publication_failed") from error

    summary = _api_evaluation_summary(
        report,
        candidate_weights_sha256=weights_sha256,
        package_sha256=manifest.package_sha256,
    )
    return CandidateIdentity(
        model_id=manifest.model_id,
        revision=manifest.revision,
        package_sha256=manifest.package_sha256,
        candidate_weights_sha256=weights_sha256,
        evaluation=summary,
    )


def _export_candidate(
    best_checkpoint: Path,
    destination: Path,
    model_root: Path,
    model_state: dict[str, object],
) -> None:
    """Export a full base-architecture CLIP checkpoint without pickle extras."""
    transformers = cast(
        "_TransformersApi",
        cast("object", importlib.import_module("transformers")),
    )
    _ = best_checkpoint
    load_options = cast(
        "dict[str, object]",
        cast("object", builtin_base_load_options(model_root).model_dump()),
    )
    model = transformers.CLIPModel.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
        **load_options,
    )
    model.load_state_dict(model_state)
    processor = transformers.CLIPProcessor.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)


def _validate_export(stage: Path) -> Path:
    """Prune save_pretrained extras and reject missing, linked, or special payload files."""
    for item in stage.iterdir():
        if item.is_symlink():
            raise TrainingPublicationError("candidate_export_contains_link")
        if item.name in _PUBLISHED_PAYLOAD_FILES:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    for name in _PUBLISHED_PAYLOAD_FILES:
        path = stage / name
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            raise TrainingPublicationError("candidate_export_incomplete") from error
        if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
            raise TrainingPublicationError("candidate_export_incomplete")
    return stage / "model.safetensors"


def _file_records(stage: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    file_names: set[str] = set()
    for path in sorted(stage.iterdir(), key=lambda item: item.name):
        if path.name == _PACKAGE_FILENAME:
            continue
        if path.name not in _PUBLISHED_PAYLOAD_FILES | {_REPORT_FILENAME}:
            raise TrainingPublicationError("candidate_export_contains_unlisted_file")
        file_names.add(path.name)
        records.append(
            {
                "path": path.name,
                "size": path.stat(follow_symlinks=False).st_size,
                "sha256": _sha256(path),
            }
        )
    expected = set(_PUBLISHED_PAYLOAD_FILES) | {_REPORT_FILENAME}
    if file_names != expected:
        raise TrainingPublicationError("candidate_export_incomplete")
    return records


def _validate_publication_binding(
    job: TrainingJobIdentity,
    report: TrainingEvaluationReport,
) -> None:
    binding = report.binding
    if (
        binding.dataset_sha256 != job.dataset_fingerprint
        or binding.source_fingerprint != job.source_fingerprint
        or binding.baseline_model_id != DEFAULT_CLIP_MODEL_ID
        or binding.baseline_revision != B16_REVISION
        or binding.dataset_split != "test"
    ):
        raise TrainingPublicationError("evaluation_provenance_mismatch")


def _candidate_model_id(job: TrainingJobIdentity) -> str:
    model_id = job.candidate_model_id or f"local/cuhk-pedes-{job.id.hex}"
    if model_id != f"local/cuhk-pedes-{job.id.hex}":
        raise TrainingPublicationError("candidate_identity_mismatch")
    return model_id


def _api_evaluation_summary(
    report: TrainingEvaluationReport,
    *,
    candidate_weights_sha256: str,
    package_sha256: str,
) -> TrainingEvaluationSummary:
    return TrainingEvaluationSummary(
        dataset_sha256=report.binding.dataset_sha256,
        dataset_split=report.binding.dataset_split,
        protocol=report.binding.protocol,
        baseline_model_id=report.binding.baseline_model_id,
        baseline_revision=report.binding.baseline_revision,
        baseline_package_sha256=report.binding.baseline_package_sha256,
        training_source_fingerprint=report.binding.source_fingerprint,
        evaluation_code_revision=report.binding.evaluation_code_revision,
        metric_definition=report.binding.metric_definition,
        best_validation_epoch=report.best_validation_epoch,
        baseline=TrainingRetrievalScores.model_validate(asdict(report.baseline)),
        candidate=TrainingRetrievalScores.model_validate(asdict(report.candidate)),
        candidate_weights_sha256=candidate_weights_sha256,
        package_sha256=package_sha256,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_durable(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "wb") as output:
        _ = output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _require_directory(path: Path) -> None:
    mode = path.lstat().st_mode
    if path.is_symlink() or not stat.S_ISDIR(mode):
        raise TrainingPublicationError("candidate_asset_root_invalid")


__all__ = ["CandidateIdentity", "TrainingPublicationError", "publish_candidate"]
