from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, cast
from uuid import uuid4

import pytest

from gods_watching.contracts.training import TrainingConfig
from gods_watching.model_selection.registry import B16_REVISION, DEFAULT_CLIP_MODEL_ID
from gods_watching.setup.models import LockedContainer, LockedModel, LockedModelFile, ModelsLock
from gods_watching.training import engine_data, evaluation
from gods_watching.training.dataset import DatasetSplit, TrainingSample
from gods_watching.training.engine import TrainingPaths, TrainingRunSnapshot
from gods_watching.training.engine_data import ClipEvaluationComponents
from gods_watching.training.evaluation import (
    EvaluationBinding,
    EvaluationCacheError,
    RetrievalEmbeddings,
    RetrievalScores,
    evaluate_best_checkpoint,
    evaluate_retrieval,
    evaluation_code_revision,
    load_baseline_cache,
    require_test_samples,
    save_baseline_cache,
    verify_pinned_baseline_package,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gods_watching.training.dataset import DatasetManifest
    from gods_watching.training.engine import TrainingCancellation
    from gods_watching.training.engine_api import TrainingModel
    from gods_watching.training.engine_data import (
        _ClipProcessor,  # pyright: ignore[reportPrivateUsage]
        _ClipValidationModel,  # pyright: ignore[reportPrivateUsage]
    )

_FINGERPRINT = "a" * 64


def _binding() -> EvaluationBinding:
    return EvaluationBinding(
        dataset_sha256=_FINGERPRINT,
        dataset_split="test",
        protocol="cuhk-pedes-original-splits-v1",
        source_fingerprint="b" * 64,
        baseline_model_id="openai/clip-vit-base-patch16",
        baseline_revision="c" * 40,
        baseline_package_sha256="f" * 64,
        evaluation_code_revision="d" * 64,
        metric_definition="cuhk-pedes-text-to-image-macro-recall-at-1-primary-r5-r10-detail-v2",
    )


def _sample(split: DatasetSplit) -> TrainingSample:
    return TrainingSample(
        split=split,
        relative_path=f"{split}/image.jpg",
        image_path=Path(f"/{split}/image.jpg"),
        person_id=1,
        captions=("person with a blue coat",),
        image_sha256="e" * 64,
        width=224,
        height=224,
    )


def _object_boundary(value: object) -> object:
    """Forget mock implementation details before narrowing to a runtime protocol."""
    return value


def test_retrieval_scores_count_all_same_person_gallery_matches() -> None:
    scores = evaluate_retrieval(
        image_embeddings=[
            [1.0, 0.0],
            [0.99, 0.1],
            [0.98, 0.2],
            [0.0, 1.0],
        ],
        text_embeddings=[[1.0, 0.0], [0.0, 1.0]],
        image_ids=[2, 1, 1, 3],
        text_ids=[1, 3],
    )

    assert scores.recall_at_1 == 0.5
    assert scores.recall_at_5 == 1.0
    assert scores.recall_at_10 == 1.0


def test_exact_ties_keep_ascending_gallery_order() -> None:
    scores = evaluate_retrieval(
        image_embeddings=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        text_embeddings=[[1.0, 0.0]],
        image_ids=[11, 22, 33],
        text_ids=[22],
    )

    assert scores.recall_at_1 == 0.0
    assert scores.recall_at_5 == 1.0
    assert scores.recall_at_10 == 1.0


def test_final_evaluation_rejects_validation_samples() -> None:
    with pytest.raises(EvaluationCacheError, match="test split"):
        _ = require_test_samples((_sample("val"),))


def test_baseline_cache_is_bound_to_dataset_and_evaluator(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    binding = _binding()
    scores = RetrievalScores(recall_at_1=0.5, recall_at_5=0.75, recall_at_10=1.0)
    save_baseline_cache(path, binding, scores)

    assert load_baseline_cache(path, binding) == scores
    assert load_baseline_cache(path, replace(binding, dataset_sha256="f" * 64)) is None
    assert load_baseline_cache(path, replace(binding, evaluation_code_revision="e" * 64)) is None


def test_evaluator_revision_binds_embedding_scoring_and_checkpoint_sources() -> None:
    from gods_watching.training.evaluation import (  # noqa: PLC0415
        _evaluator_source_revision,  # pyright: ignore[reportPrivateUsage]
    )

    sources = {
        "evaluation.py": b"test split and report policy",
        "engine_data.py": b"image/text preprocessing and embedding",
        "dataset.py": b"original split validation",
        "metrics.py": b"macro R@1/5/10 scoring",
        "memory.py": b"image shape and text truncation policy",
        "torch_backend.py": b"tensor normalization behavior",
        "checkpoints.py": b"best validation checkpoint validation",
        "calibration.py": b"pinned baseline loader options",
    }
    revision = _evaluator_source_revision(sources)
    assert len(revision) == 64
    assert _evaluator_source_revision(sources) == revision
    for source_name in (
        "engine_data.py",
        "memory.py",
        "metrics.py",
        "torch_backend.py",
    ):
        changed = dict(sources)
        changed[source_name] += b" changed"
        assert _evaluator_source_revision(changed) != revision
    assert len(evaluation_code_revision()) == 64


def test_baseline_cache_and_report_binding_reject_same_path_asset_mutation(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "models" / "clip"
    model_root.mkdir(parents=True)
    filenames = (
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )
    file_records: list[LockedModelFile] = []
    for name in filenames:
        content = f"pinned:{name}".encode()
        _ = (model_root / name).write_bytes(content)
        file_records.append(
            LockedModelFile(
                path=Path("clip") / name,
                sha256=hashlib.sha256(content).hexdigest(),
                size=len(content),
            )
        )
    _ = (model_root / "gods-watching-model.json").write_text(
        json.dumps(
            {
                "model_id": DEFAULT_CLIP_MODEL_ID,
                "revision": B16_REVISION,
                "dimension": 512,
                "processor": "CLIPProcessor",
                "runtime": "transformers",
            }
        ),
        encoding="utf-8",
    )
    lock_path = tmp_path / "models.lock.json"
    lock = ModelsLock(
        schema_version="1",
        container=LockedContainer(
            base_image="test",
            base_digest="sha256:test",
            built_image="test",
            python_abi="cp312",
        ),
        models=(
            LockedModel(
                model_id=DEFAULT_CLIP_MODEL_ID,
                revision=B16_REVISION,
                license="MIT",
                source="test",
                files=tuple(file_records),
            ),
        ),
    )
    _ = lock_path.write_text(lock.model_dump_json(), encoding="utf-8")

    package_identity = verify_pinned_baseline_package(model_root, lock_path)
    binding = replace(_binding(), baseline_package_sha256=package_identity)
    cache_path = tmp_path / "baseline-cache.json"
    scores = RetrievalScores(recall_at_1=0.4, recall_at_5=0.7, recall_at_10=0.9)
    save_baseline_cache(cache_path, binding, scores)
    assert load_baseline_cache(cache_path, binding) == scores

    changed_name = "tokenizer_config.json"
    changed_file = model_root / changed_name
    changed_bytes = b"mutated processor file"
    _ = changed_file.write_bytes(changed_bytes)
    with pytest.raises(EvaluationCacheError, match="does not match its lock"):
        _ = verify_pinned_baseline_package(model_root, lock_path)

    original_model = lock.models[0]
    changed_files = tuple(
        LockedModelFile(
            path=item.path,
            sha256=hashlib.sha256(changed_bytes).hexdigest(),
            size=len(changed_bytes),
        )
        if item.path == Path("clip") / changed_name
        else item
        for item in original_model.files
    )
    updated_model = original_model.model_copy(update={"files": changed_files})
    updated_lock = lock.model_copy(update={"models": (updated_model,)})
    _ = lock_path.write_text(updated_lock.model_dump_json(), encoding="utf-8")
    changed_identity = verify_pinned_baseline_package(model_root, lock_path)
    assert changed_identity != package_identity
    assert load_baseline_cache(
        cache_path,
        replace(binding, baseline_package_sha256=changed_identity),
    ) is None


def test_final_evaluation_uses_test_split_and_validation_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_samples = (_sample("test"),)
    class FakeModel:
        selected_state: dict[str, str] | None = None

        def load_state_dict(self, state: Mapping[str, object]) -> None:
            self.selected_state = {
                key: value for key, value in state.items() if isinstance(value, str)
            }

    model = FakeModel()
    model_object = _object_boundary(model)
    processor_object: object = object()
    components = ClipEvaluationComponents(
        model=cast("_ClipValidationModel", model_object),
        processor=cast("_ClipProcessor", processor_object),
        test_samples=test_samples,
        dataset_protocol="cuhk-pedes-original-splits-v1",
    )

    def load_components(
        _snapshot: TrainingRunSnapshot,
        _paths: TrainingPaths,
    ) -> ClipEvaluationComponents:
        return components

    monkeypatch.setattr(
        engine_data,
        "load_clip_evaluation_components",
        load_components,
        raising=False,
    )
    observed_splits: list[tuple[str, ...]] = []

    def embed(
        _model: TrainingModel,
        _processor: _ClipProcessor,
        samples: tuple[TrainingSample, ...],
        *,
        cancellation: TrainingCancellation | None = None,
    ) -> RetrievalEmbeddings:
        _ = cancellation
        observed_splits.append(tuple(sample.split for sample in samples))
        return RetrievalEmbeddings(
            image_embeddings=[[1.0, 0.0], [0.0, 1.0]],
            text_embeddings=[[1.0, 0.0], [0.0, 1.0]],
            image_ids=(1, 2),
            text_ids=(1, 2),
        )

    monkeypatch.setattr(engine_data, "embed_cuhk_samples", embed, raising=False)
    selected_state = {"text_encoder": "best-validation-epoch"}

    def load_checkpoint(
        _path: Path,
        _identity: Mapping[str, object],
    ) -> dict[str, object]:
        return {
            "training": {"best_epoch": 3},
            "model": selected_state,
        }

    def verify_baseline(_root: Path, _lock_path: Path) -> str:
        return "f" * 64

    monkeypatch.setattr(
        evaluation,
        "load_checkpoint_verified",
        load_checkpoint,
        raising=False,
    )
    monkeypatch.setattr(evaluation, "verify_pinned_baseline_package", verify_baseline)
    snapshot = TrainingRunSnapshot(
        job_id=uuid4(),
        owner_generation=2,
        config=TrainingConfig(epochs=4),
        dataset_fingerprint=_FINGERPRINT,
        source_fingerprint="b" * 64,
    )
    paths = TrainingPaths(
        run_directory=tmp_path,
        dataset_root=tmp_path / "dataset",
        model_root=Path("/models/clip"),
    )

    report = evaluate_best_checkpoint(
        snapshot,
        paths,
        tmp_path / "best.pt",
        cancellation=threading.Event(),
    )

    assert observed_splits == [("test",), ("test",)]
    assert model.selected_state == selected_state
    assert report.best_validation_epoch == 3


def test_evaluation_loader_selects_original_test_split_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = _sample("val")
    test_sample = _sample("test")

    class _Manifest:
        fingerprint: str = _FINGERPRINT
        protocol: str = "cuhk-pedes-original-splits-v1"

        def samples_for(self, split: str) -> tuple[TrainingSample, ...]:
            return (test_sample,) if split == "test" else (validation,)

    model = object()
    processor = object()

    def validate(_root: Path) -> DatasetManifest:
        return cast("DatasetManifest", cast("object", _Manifest()))

    monkeypatch.setattr(engine_data, "validate_cuhk", validate)

    def load_model(
        _root: Path,
        *,
        gradient_checkpointing: bool,
    ) -> tuple[_ClipValidationModel, _ClipProcessor]:
        _ = gradient_checkpointing
        return (
            cast("_ClipValidationModel", model),
            cast("_ClipProcessor", processor),
        )

    monkeypatch.setattr(
        engine_data,
        "_load_local_clip",
        load_model,
    )
    snapshot = TrainingRunSnapshot(
        job_id=uuid4(),
        owner_generation=1,
        config=TrainingConfig(),
        dataset_fingerprint=_FINGERPRINT,
        source_fingerprint="b" * 64,
    )
    paths = TrainingPaths(tmp_path, tmp_path / "dataset", Path("/models/clip"))

    components = engine_data.load_clip_evaluation_components(snapshot, paths)

    assert components.model is model
    assert components.processor is processor
    assert components.test_samples == (test_sample,)
    assert components.dataset_protocol == "cuhk-pedes-original-splits-v1"
