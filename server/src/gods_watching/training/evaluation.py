"""Deterministic CUHK-PEDES test metrics and bound baseline-cache records."""

# Keep these CPU-side contracts importable without Torch; worker imports are lazy.
# ruff: noqa: TRY003, EM101, EM102, TC001, PLC0415, C901, PLR0911, PLR0912, PLR0915, PLR2004, D105

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard, cast

from gods_watching.model_selection.registry import B16_REVISION, DEFAULT_CLIP_MODEL_ID
from gods_watching.training.checkpoints import load_checkpoint_verified
from gods_watching.training.dataset import TrainingSample
from gods_watching.training.retrieval import (
    EvaluationCacheError,
    EvaluationCancelledError,
    RetrievalEmbeddings,
    RetrievalScores,
    evaluate_retrieval,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from gods_watching.training.engine import (
        TrainingCancellation,
        TrainingPaths,
        TrainingRunSnapshot,
    )

_CACHE_SCHEMA_VERSION = 1
_REPORT_METRIC_DEFINITION = "cuhk-pedes-text-to-image-macro-recall-at-1-primary-r5-r10-detail-v2"
_SHA256_LENGTH = 64
_EVALUATION_SOURCE_FILES = (
    "evaluation.py",
    "engine_data.py",
    "dataset.py",
    "metrics.py",
    "memory.py",
    "retrieval.py",
    "torch_backend.py",
    "determinism.py",
    "checkpoints.py",
    "calibration.py",
)
_BASELINE_PACKAGE_IDENTITY_FILES = {
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}


@dataclass(frozen=True, slots=True)
class EvaluationBinding:
    """Immutable identities that make one baseline cache safe to reuse."""

    dataset_sha256: str
    dataset_split: Literal["test"]
    protocol: str
    source_fingerprint: str
    baseline_model_id: str
    baseline_revision: str
    baseline_package_sha256: str
    evaluation_code_revision: str
    metric_definition: str = _REPORT_METRIC_DEFINITION

    def __post_init__(self) -> None:
        """Require held-out test and content identities for every cache record."""
        hash_fields = (
            ("dataset_sha256", self.dataset_sha256),
            ("source_fingerprint", self.source_fingerprint),
            ("baseline_package_sha256", self.baseline_package_sha256),
            ("evaluation_code_revision", self.evaluation_code_revision),
        )
        for field_name, value in hash_fields:
            if (
                not _is_sha256_digest(value)
                or len(value) != _SHA256_LENGTH
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise EvaluationCacheError(f"{field_name} must be a lowercase SHA-256 digest")
        if self.dataset_split != "test":
            raise EvaluationCacheError("final retrieval evaluation must use the test split")
        if not self.protocol or not self.baseline_model_id or not self.baseline_revision:
            raise EvaluationCacheError("evaluation provenance is incomplete")
        if self.metric_definition != _REPORT_METRIC_DEFINITION:
            raise EvaluationCacheError("unsupported CUHK-PEDES metric definition")


@dataclass(frozen=True, slots=True)
class TrainingEvaluationReport:
    """Safe package/API evaluation report without captions or host paths."""

    binding: EvaluationBinding
    baseline: RetrievalScores
    candidate: RetrievalScores
    best_validation_epoch: int

    def __post_init__(self) -> None:
        if type(self.best_validation_epoch) is not int or self.best_validation_epoch < 1:
            raise EvaluationCacheError("best validation epoch must be positive")

    def package_report(self, candidate_weights_sha256: str) -> dict[str, object]:
        """Create the exact importer report fields plus safe provenance detail."""
        if len(candidate_weights_sha256) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in candidate_weights_sha256
        ):
            raise EvaluationCacheError("candidate weights hash is malformed")
        return {
            "dataset_split": self.binding.dataset_split,
            "dataset_sha256": self.binding.dataset_sha256,
            "protocol": self.binding.protocol,
            "source_checkpoint": self.binding.baseline_model_id,
            "source_checkpoint_revision": self.binding.baseline_revision,
            "baseline_package_sha256": self.binding.baseline_package_sha256,
            "candidate_weights_sha256": candidate_weights_sha256,
            "evaluation_code_revision": self.binding.evaluation_code_revision,
            "metric_definition": self.binding.metric_definition,
            "baseline_score": self.baseline.recall_at_1,
            "candidate_score": self.candidate.recall_at_1,
            "baseline_text_to_image": asdict(self.baseline),
            "candidate_text_to_image": asdict(self.candidate),
            "training_source_fingerprint": self.binding.source_fingerprint,
            "best_validation_epoch": self.best_validation_epoch,
        }


def require_test_samples(samples: Sequence[TrainingSample]) -> tuple[TrainingSample, ...]:
    """Reject empty or non-test inputs before final evaluation can emit a report."""
    if not samples or any(sample.split != "test" for sample in samples):
        raise EvaluationCacheError("final evaluation requires only CUHK-PEDES test split samples")
    return tuple(samples)


def evaluate_best_checkpoint(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
    best_checkpoint: Path,
    *,
    cancellation: TrainingCancellation,
) -> TrainingEvaluationReport:
    """Compare locked baseline and validation-selected weights on native test rows."""
    from gods_watching.training.engine_data import (
        embed_cuhk_samples,
        load_clip_evaluation_components,
    )

    if cancellation.is_set():
        raise EvaluationCancelledError("final evaluation cancelled")
    checkpoint = load_checkpoint_verified(best_checkpoint, snapshot.checkpoint_identity())
    raw_training = checkpoint.get("training")
    if not isinstance(raw_training, dict):
        raise EvaluationCacheError("best checkpoint selection metadata is missing")
    training = cast("dict[str, object]", raw_training)
    best_epoch = training.get("best_epoch")
    if type(best_epoch) is not int or best_epoch < 1:
        raise EvaluationCacheError("best validation epoch is missing")
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise EvaluationCacheError("best checkpoint model state is missing")

    components = load_clip_evaluation_components(snapshot, paths)
    test_samples = require_test_samples(components.test_samples)
    baseline_package_sha256 = verify_pinned_baseline_package(
        paths.model_root,
        paths.model_lock_path,
    )
    binding = EvaluationBinding(
        dataset_sha256=snapshot.dataset_fingerprint,
        dataset_split="test",
        protocol=components.dataset_protocol,
        source_fingerprint=snapshot.source_fingerprint,
        baseline_model_id=DEFAULT_CLIP_MODEL_ID,
        baseline_revision=B16_REVISION,
        baseline_package_sha256=baseline_package_sha256,
        evaluation_code_revision=evaluation_code_revision(),
    )
    baseline_cache = paths.run_directory / "baseline-evaluation.json"
    baseline_scores = load_baseline_cache(baseline_cache, binding)
    if baseline_scores is None:
        baseline_embeddings = embed_cuhk_samples(
            components.model,
            components.processor,
            test_samples,
            cancellation=cancellation,
        )
        baseline_scores = evaluate_retrieval(
            baseline_embeddings.image_embeddings,
            baseline_embeddings.text_embeddings,
            baseline_embeddings.image_ids,
            baseline_embeddings.text_ids,
        )
        if cancellation.is_set():
            raise EvaluationCancelledError("final evaluation cancelled")
        save_baseline_cache(baseline_cache, binding, baseline_scores)

    if cancellation.is_set():
        raise EvaluationCancelledError("final evaluation cancelled")
    components.model.load_state_dict(cast("Mapping[str, object]", model_state))
    candidate_embeddings = embed_cuhk_samples(
        components.model,
        components.processor,
        test_samples,
        cancellation=cancellation,
    )
    candidate_scores = evaluate_retrieval(
        candidate_embeddings.image_embeddings,
        candidate_embeddings.text_embeddings,
        candidate_embeddings.image_ids,
        candidate_embeddings.text_ids,
    )
    if cancellation.is_set():
        raise EvaluationCancelledError("final evaluation cancelled")
    return TrainingEvaluationReport(
        binding=binding,
        baseline=baseline_scores,
        candidate=candidate_scores,
        best_validation_epoch=best_epoch,
    )


def evaluation_code_revision() -> str:
    """Hash the full local evaluator dependency bundle used for held-out scores."""
    source_root = Path(__file__).parent
    sources = {name: (source_root / name).read_bytes() for name in _EVALUATION_SOURCE_FILES}
    return _evaluator_source_revision(sources)


def verify_pinned_baseline_package(model_root: Path, lock_path: Path) -> str:
    """Hash and verify the exact locked B/16 weights and processor before loading."""
    from gods_watching.model_selection.assets import IDENTITY_MARKER_NAME
    from gods_watching.setup.models import load_models_lock

    try:
        root = Path(model_root)
        if root.is_symlink() or not root.is_dir():
            raise EvaluationCacheError("pinned baseline package is unavailable")
        lock = load_models_lock(Path(lock_path))
        model = next(item for item in lock.models if item.model_id == DEFAULT_CLIP_MODEL_ID)
    except (OSError, StopIteration, ValueError, TypeError) as error:
        raise EvaluationCacheError("pinned baseline lock is unavailable") from error
    if model.revision != B16_REVISION:
        raise EvaluationCacheError("pinned baseline revision does not match the evaluator")

    files = tuple(sorted(model.files, key=lambda item: item.path.as_posix()))
    expected_names: set[str] = set()
    package_digest = hashlib.sha256()
    for locked_file in files:
        locked_path = locked_file.path
        if (
            locked_path.is_absolute()
            or len(locked_path.parts) != 2
            or locked_path.parts[0] != root.name
            or ".." in locked_path.parts
        ):
            raise EvaluationCacheError("pinned baseline lock paths do not match the model root")
        name = locked_path.name
        expected_names.add(name)
        candidate = root / name
        file_descriptor = -1
        try:
            file_descriptor = os.open(
                candidate,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            )
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise EvaluationCacheError("pinned baseline package contains an unsafe file")
            if metadata.st_size != locked_file.size:
                raise EvaluationCacheError("pinned baseline file size does not match its lock")
            file_digest = hashlib.sha256()
            with os.fdopen(file_descriptor, "rb") as source:
                file_descriptor = -1
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    file_digest.update(chunk)
        except OSError as error:
            raise EvaluationCacheError("pinned baseline package file is unavailable") from error
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)
        if file_digest.hexdigest() != locked_file.sha256:
            raise EvaluationCacheError("pinned baseline file digest does not match its lock")
        package_digest.update(name.encode("utf-8"))
        package_digest.update(b"\0")
        package_digest.update(bytes.fromhex(locked_file.sha256))
        package_digest.update(b"\0")

    try:
        actual_names = {path.name for path in root.iterdir()}
    except OSError as error:
        raise EvaluationCacheError("pinned baseline package listing is unavailable") from error
    if expected_names != _BASELINE_PACKAGE_IDENTITY_FILES or actual_names != expected_names | {
        IDENTITY_MARKER_NAME
    }:
        raise EvaluationCacheError("pinned baseline package contents do not match the lock")
    marker_path = root / IDENTITY_MARKER_NAME
    try:
        if marker_path.is_symlink() or not marker_path.is_file():
            raise EvaluationCacheError("pinned baseline identity marker is unavailable")
        marker = _parse_json_object(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvaluationCacheError("pinned baseline identity marker is invalid") from error
    if marker is None:
        raise EvaluationCacheError("pinned baseline identity marker is invalid")
    if marker != {
        "model_id": DEFAULT_CLIP_MODEL_ID,
        "revision": B16_REVISION,
        "dimension": 512,
        "processor": "CLIPProcessor",
        "runtime": "transformers",
    }:
        raise EvaluationCacheError("pinned baseline identity marker does not match the lock")
    return package_digest.hexdigest()


def _evaluator_source_revision(sources: Mapping[str, bytes]) -> str:
    """Return a deterministic digest that changes with any local evaluator source."""
    digest = hashlib.sha256()
    for name in sorted(sources):
        content = sources[name]
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _is_sha256_digest(value: object) -> TypeGuard[str]:
    """Return whether a dynamic value is a lowercase SHA-256 string."""
    return isinstance(value, str)


def _string_keyed_object(decoded: object) -> dict[str, object] | None:
    """Narrow a decoded value to a string-keyed mapping."""
    if not isinstance(decoded, dict):
        return None
    entries = cast("dict[object, object]", decoded)
    if not all(isinstance(key, str) for key in entries):
        return None
    return cast("dict[str, object]", entries)


def _parse_json_object(payload: str) -> dict[str, object] | None:
    """Narrow decoded JSON to a string-keyed mapping at the trust boundary."""
    return _string_keyed_object(cast("object", json.loads(payload)))


def load_baseline_cache(path: Path, binding: EvaluationBinding) -> RetrievalScores | None:
    """Return cached baseline scores only when every immutable binding matches."""
    try:
        if stat.S_ISLNK(path.lstat().st_mode) or not stat.S_ISREG(path.lstat().st_mode):
            return None
        value = _parse_json_object(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if value is None or set(value) != {"schema_version", "binding", "scores"}:
        return None
    if value.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return None
    raw_binding = value.get("binding")
    parsed_binding = _string_keyed_object(raw_binding)
    if parsed_binding is None or parsed_binding != asdict(binding):
        return None
    raw_scores = value.get("scores")
    parsed_scores = _string_keyed_object(raw_scores)
    if parsed_scores is None or set(parsed_scores) != {
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
    }:
        return None
    try:
        return RetrievalScores(**cast("dict[str, float]", parsed_scores))
    except (TypeError, EvaluationCacheError):
        return None


def save_baseline_cache(path: Path, binding: EvaluationBinding, scores: RetrievalScores) -> None:
    """Atomically persist a small provenance-bound baseline score record."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    payload = json.dumps(
        {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "binding": asdict(binding),
            "scores": asdict(scores),
        },
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            _ = stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _ = temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            _ = os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


__all__ = [
    "EvaluationBinding",
    "EvaluationCacheError",
    "EvaluationCancelledError",
    "RetrievalEmbeddings",
    "RetrievalScores",
    "TrainingEvaluationReport",
    "evaluate_best_checkpoint",
    "evaluate_retrieval",
    "evaluation_code_revision",
    "load_baseline_cache",
    "require_test_samples",
    "save_baseline_cache",
]
