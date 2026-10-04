from pathlib import Path

from gods_watching.training.dataset import TrainingSample
from gods_watching.training.sampler import IdentityBatchSampler


def _samples() -> tuple[TrainingSample, ...]:
    return (
        TrainingSample(
            split="train",
            relative_path="train/person-1-a.jpg",
            image_path=Path("/dataset/imgs/train/person-1-a.jpg"),
            person_id=1,
            captions=("one A", "one B"),
            image_sha256="a" * 64,
            width=10,
            height=20,
        ),
        TrainingSample(
            split="train",
            relative_path="train/person-1-b.jpg",
            image_path=Path("/dataset/imgs/train/person-1-b.jpg"),
            person_id=1,
            captions=("one C", "one D"),
            image_sha256="b" * 64,
            width=10,
            height=20,
        ),
        TrainingSample(
            split="train",
            relative_path="train/person-2-a.jpg",
            image_path=Path("/dataset/imgs/train/person-2-a.jpg"),
            person_id=2,
            captions=("two A", "two B"),
            image_sha256="c" * 64,
            width=10,
            height=20,
        ),
        TrainingSample(
            split="train",
            relative_path="train/person-3-a.jpg",
            image_path=Path("/dataset/imgs/train/person-3-a.jpg"),
            person_id=3,
            captions=("three A", "three B"),
            image_sha256="d" * 64,
            width=10,
            height=20,
        ),
        TrainingSample(
            split="train",
            relative_path="train/person-4-a.jpg",
            image_path=Path("/dataset/imgs/train/person-4-a.jpg"),
            person_id=4,
            captions=("four A", "four B"),
            image_sha256="e" * 64,
            width=10,
            height=20,
        ),
    )


def test_sampler_has_distinct_identity_and_reproducible_epochs() -> None:
    samples = _samples()
    first = IdentityBatchSampler(samples, batch_size=3, seed=42, epoch=0)
    repeated = IdentityBatchSampler(samples, batch_size=3, seed=42, epoch=0)
    next_epoch = IdentityBatchSampler(samples, batch_size=3, seed=42, epoch=1)

    first_batches = tuple(
        tuple((item.sample.relative_path, item.caption) for item in b) for b in first
    )
    repeated_batches = tuple(
        tuple((item.sample.relative_path, item.caption) for item in b) for b in repeated
    )
    next_batches = tuple(
        tuple((item.sample.relative_path, item.caption) for item in b) for b in next_epoch
    )

    assert first_batches == repeated_batches
    assert first_batches != next_batches
    assert all(len({item.sample.person_id for item in batch}) == len(batch) for batch in first)
    emitted = [item.sample.relative_path for batch in first for item in batch]
    assert len(emitted) + first.remainder_count == len(samples)
    assert len(emitted) == len(set(emitted))
    assert all(item.caption in item.sample.captions for batch in first for item in batch)
