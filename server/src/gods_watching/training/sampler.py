"""Seeded micro-batches that never place the same person twice in one batch."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from gods_watching.training.dataset import TrainingSample

_MIN_BATCH_SIZE = 2
_MAX_BATCH_SIZE = 128


@dataclass(frozen=True, slots=True)
class SampleSelection:
    """One image paired with a seeded choice of its person description."""

    sample: TrainingSample
    caption: str
    caption_index: int


class IdentityBatchSampler:
    """Build deterministic image micro-batches with distinct person identities.

    Each image is consumed at most once per epoch. The last partial batch is
    emitted when it has at least two distinct identities. Remaining images from
    a single identity are counted in ``remainder_count`` because they cannot
    form a valid contrastive batch. Gradient accumulation is outside this
    sampler; it never enlarges a micro-batch's negative pool.
    """

    batch_size: int
    seed: int
    epoch: int
    _batches: tuple[tuple[SampleSelection, ...], ...]
    remainder_count: int

    def __init__(  # noqa: C901
        self,
        samples: Sequence[TrainingSample],
        batch_size: int,
        seed: int,
        epoch: int,
    ) -> None:
        """Shuffle one epoch of train images and record any unbatchable tail."""
        if type(batch_size) is not int or not _MIN_BATCH_SIZE <= batch_size <= _MAX_BATCH_SIZE:
            raise ValueError("batch size must be an integer between 2 and 128")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if type(epoch) is not int or epoch < 0:
            raise ValueError("epoch must be a non-negative integer")

        buckets: dict[int, list[TrainingSample]] = {}
        for sample in samples:
            if sample.split != "train":
                raise ValueError("identity batch sampler accepts train samples only")
            if not sample.captions:
                raise ValueError("training sample must contain at least one caption")
            buckets.setdefault(sample.person_id, []).append(sample)
        if len(buckets) < _MIN_BATCH_SIZE:
            raise ValueError("training samples must contain at least two identities")

        rng = random.Random((seed << 32) ^ epoch)  # noqa: S311
        for bucket in buckets.values():
            rng.shuffle(bucket)
        active = list(buckets)
        rng.shuffle(active)
        active_ids = deque(active)
        batches: list[tuple[SampleSelection, ...]] = []

        while len(active_ids) >= _MIN_BATCH_SIZE:
            selected_count = min(batch_size, len(active_ids))
            batch: list[SampleSelection] = []
            for _ in range(selected_count):
                person_id = active_ids.popleft()
                bucket = buckets[person_id]
                sample = bucket.pop()
                caption_index = rng.randrange(len(sample.captions))
                batch.append(
                    SampleSelection(
                        sample=sample,
                        caption=sample.captions[caption_index],
                        caption_index=caption_index,
                    )
                )
                if bucket:
                    active_ids.append(person_id)
            batches.append(tuple(batch))

        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self._batches = tuple(batches)
        self.remainder_count = sum(len(bucket) for bucket in buckets.values())

    def __iter__(self) -> Iterator[tuple[SampleSelection, ...]]:
        """Yield each deterministic micro-batch for this epoch."""
        return iter(self._batches)

    def __len__(self) -> int:
        """Return the number of valid micro-batches in this epoch."""
        return len(self._batches)


__all__ = ["IdentityBatchSampler", "SampleSelection"]
