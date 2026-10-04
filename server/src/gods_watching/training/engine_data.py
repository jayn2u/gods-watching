"""Local CUHK image batching and CLIP validation data adapters."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import hashlib
import importlib
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from PIL import Image

from gods_watching.training.calibration import builtin_base_load_options
from gods_watching.training.dataset import DatasetValidationError, TrainingSample, validate_cuhk
from gods_watching.training.engine import TrainingBatch, TrainingComponents, TrainingEngineError
from gods_watching.training.engine_api import TrainingModel
from gods_watching.training.memory import SUPPORTED_IMAGE_SIZE, SUPPORTED_TEXT_MAX_LENGTH
from gods_watching.training.retrieval import (
    EvaluationCancelledError,
    RetrievalEmbeddings,
    evaluate_retrieval,
)
from gods_watching.training.sampler import IdentityBatchSampler

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from gods_watching.training.engine import (
        TrainingCancellation,
        TrainingPaths,
        TrainingRunSnapshot,
    )
    from gods_watching.training.engine_api import TrainingTensor
    from gods_watching.training.sampler import SampleSelection

_EVALUATION_IMAGE_BATCH_SIZE = 16
_EVALUATION_TEXT_BATCH_SIZE = 32


class _ClipValidationModel(TrainingModel, Protocol):
    """CLIP feature APIs used only for the validation split."""

    def gradient_checkpointing_enable(self) -> None: ...

    def get_image_features(self, *, pixel_values: TrainingTensor) -> TrainingTensor: ...

    def get_text_features(
        self,
        *,
        input_ids: TrainingTensor,
        attention_mask: TrainingTensor,
    ) -> TrainingTensor: ...


class _ClipProcessor(Protocol):
    """Pinned Transformers processor inputs and tensor output surface."""

    def __call__(  # noqa: PLR0913
        self,
        *,
        images: Sequence[Image.Image] | None = None,
        text: Sequence[str] | None = None,
        return_tensors: str,
        padding: bool = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> Mapping[str, TrainingTensor]: ...


class _ClipProcessorLoader(Protocol):
    """Transformers processor loader."""

    def from_pretrained(self, model_path: str, *, local_files_only: bool) -> _ClipProcessor: ...


class _ClipModelLoader(Protocol):
    """Transformers model loader."""

    def from_pretrained(
        self,
        model_path: str,
        *,
        local_files_only: bool,
        **options: object,
    ) -> _ClipValidationModel: ...


class _TransformersApi(Protocol):
    """Narrow dynamic surface for the pinned Transformers CLIP classes."""

    CLIPModel: _ClipModelLoader
    CLIPProcessor: _ClipProcessorLoader


@dataclass(frozen=True, slots=True)
class ClipEvaluationComponents:
    """Local base CLIP pair and only the validated held-out test rows."""

    model: _ClipValidationModel
    processor: _ClipProcessor
    test_samples: tuple[TrainingSample, ...]
    dataset_protocol: str


def load_clip_components(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
) -> TrainingComponents:
    """Load only the locked local CLIP package and hash-check every training image."""
    manifest = validate_cuhk(paths.dataset_root)
    if manifest.fingerprint != snapshot.dataset_fingerprint:
        raise DatasetValidationError("dataset changed between admission and worker startup")
    model, processor = _load_local_clip(
        paths.model_root,
        gradient_checkpointing=snapshot.config.gradient_checkpointing,
    )
    train_samples = manifest.samples_for("train")
    validation_samples = manifest.samples_for("val")

    def sampler(epoch: int) -> IdentityBatchSampler:
        return IdentityBatchSampler(
            train_samples,
            snapshot.config.micro_batch_size,
            snapshot.config.seed,
            epoch,
        )

    def encode(selections: tuple[SampleSelection, ...]) -> TrainingBatch:
        images: list[Image.Image] = []
        captions: list[str] = []
        for selection in selections:
            sample = selection.sample
            images.append(read_verified_image(sample))
            captions.append(selection.caption)
        encoded = processor(
            images=images,
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SUPPORTED_TEXT_MAX_LENGTH,
        )
        pixel_values = encoded["pixel_values"]
        if tuple(pixel_values.shape[-2:]) != (SUPPORTED_IMAGE_SIZE, SUPPORTED_IMAGE_SIZE):
            raise TrainingEngineError("processor produced an unsupported image size")
        return TrainingBatch(model_inputs=dict(encoded))

    def count_batches(epoch: int) -> int:
        return len(sampler(epoch))

    def train_batches(epoch: int) -> Iterable[TrainingBatch]:
        for selections in sampler(epoch):
            yield encode(selections)

    def validate(model_to_validate: TrainingModel, _epoch: int) -> float:
        return clip_validation_recall(model_to_validate, processor, validation_samples)

    return TrainingComponents(
        model=model,
        train_batch_count=count_batches,
        iter_train_batches=train_batches,
        validation_recall_at_1=validate,
    )


def _load_local_clip(
    model_root: Path,
    *,
    gradient_checkpointing: bool,
) -> tuple[_ClipValidationModel, _ClipProcessor]:
    if model_root.resolve(strict=True) != Path("/models/clip"):
        raise TrainingEngineError(
            "training may load weights only from the pinned /models/clip mount"
        )
    transformers = cast(
        "_TransformersApi",
        cast("object", importlib.import_module("transformers")),
    )
    processor = transformers.CLIPProcessor.from_pretrained(
        str(model_root),
        local_files_only=True,
    )
    load_options = cast(
        "dict[str, object]",
        cast("object", builtin_base_load_options(model_root).model_dump()),
    )
    model = transformers.CLIPModel.from_pretrained(
        str(model_root),
        local_files_only=True,
        **load_options,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    return model, processor


def load_clip_evaluation_components(
    snapshot: TrainingRunSnapshot,
    paths: TrainingPaths,
) -> ClipEvaluationComponents:
    """Load the pinned baseline model and only native CUHK-PEDES test samples."""
    manifest = validate_cuhk(paths.dataset_root)
    if manifest.fingerprint != snapshot.dataset_fingerprint:
        raise DatasetValidationError("dataset changed between admission and final evaluation")
    model, processor = _load_local_clip(paths.model_root, gradient_checkpointing=False)
    return ClipEvaluationComponents(
        model=model,
        processor=processor,
        test_samples=manifest.samples_for("test"),
        dataset_protocol=manifest.protocol,
    )


def read_verified_image(sample: TrainingSample) -> Image.Image:
    """Read through an O_NOFOLLOW descriptor and verify bytes against admission hash."""
    file_descriptor = os.open(
        sample.image_path,
        os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
    )
    with os.fdopen(file_descriptor, "rb") as source:
        content = source.read()
    if hashlib.sha256(content).hexdigest() != sample.image_sha256:
        raise DatasetValidationError("dataset image changed after admission")
    with Image.open(BytesIO(content)) as image:
        return image.convert("RGB").copy()


def clip_validation_recall(
    model: TrainingModel,
    processor: _ClipProcessor,
    samples: tuple[TrainingSample, ...],
) -> float:
    """Embed validation images/captions and score macro text-to-image R@1."""
    embeddings = embed_cuhk_samples(model, processor, samples)
    return evaluate_retrieval(
        embeddings.image_embeddings,
        embeddings.text_embeddings,
        embeddings.image_ids,
        embeddings.text_ids,
    ).recall_at_1


def embed_cuhk_samples(
    model: TrainingModel,
    processor: _ClipProcessor,
    samples: tuple[TrainingSample, ...],
    *,
    cancellation: TrainingCancellation | None = None,
) -> RetrievalEmbeddings:
    """Embed one CUHK split into CPU vectors using bounded CUDA batches."""
    from gods_watching.training.torch_backend import create_torch_training_backend  # noqa: PLC0415

    backend = create_torch_training_backend()
    if not backend.cuda_available():
        raise TrainingEngineError("CUDA is required for final CUHK-PEDES evaluation")
    device = backend.create_device("cuda:0")
    clip_model = cast("_ClipValidationModel", model)
    clip_model = cast("_ClipValidationModel", clip_model.to(device))
    clip_model.eval()
    image_embeddings: list[TrainingTensor] = []
    image_identity: list[int] = []
    text_embeddings: list[TrainingTensor] = []
    text_identity: list[int] = []
    with backend.no_grad():
        for start in range(0, len(samples), _EVALUATION_IMAGE_BATCH_SIZE):
            if cancellation is not None and cancellation.is_set():
                raise EvaluationCancelledError
            chunk = samples[start : start + _EVALUATION_IMAGE_BATCH_SIZE]
            images = [read_verified_image(sample) for sample in chunk]
            encoded = processor(images=images, return_tensors="pt")
            embeddings = backend.normalize(
                clip_model.get_image_features(
                    pixel_values=encoded["pixel_values"].to(device)
                )
            )
            image_embeddings.append(embeddings.cpu())
            image_identity.extend(sample.person_id for sample in chunk)
            text_rows = [
                (sample.person_id, caption) for sample in chunk for caption in sample.captions
            ]
            for text_start in range(0, len(text_rows), _EVALUATION_TEXT_BATCH_SIZE):
                if cancellation is not None and cancellation.is_set():
                    raise EvaluationCancelledError
                text_chunk = text_rows[
                    text_start : text_start + _EVALUATION_TEXT_BATCH_SIZE
                ]
                text_encoded = processor(
                    text=[caption for _person_id, caption in text_chunk],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=SUPPORTED_TEXT_MAX_LENGTH,
                )
                encoded_text = backend.normalize(
                    clip_model.get_text_features(
                        input_ids=text_encoded["input_ids"].to(device),
                        attention_mask=text_encoded["attention_mask"].to(device),
                    )
                )
                text_embeddings.append(encoded_text.cpu())
                text_identity.extend(person_id for person_id, _caption in text_chunk)
    return RetrievalEmbeddings(
        image_embeddings=backend.concatenate(image_embeddings),
        text_embeddings=backend.concatenate(text_embeddings),
        image_ids=tuple(image_identity),
        text_ids=tuple(text_identity),
    )


__all__ = ["clip_validation_recall", "load_clip_components", "read_verified_image"]
