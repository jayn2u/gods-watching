"""Shared CPU-only retrieval metric and result contracts for CLIP evaluation."""

# ruff: noqa: TRY003, EM101

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gods_watching.training.metrics import text_to_image_recall_at_k

if TYPE_CHECKING:
    from collections.abc import Sequence


class EvaluationCacheError(ValueError):
    """A retrieval result or final evaluation input violated its fixed contract."""


class EvaluationCancelledError(EvaluationCacheError):
    """Final evaluation stopped at a safe batch boundary after operator cancellation."""


@dataclass(frozen=True, slots=True)
class RetrievalScores:
    """Macro text-query recall against the identity-relevant image gallery."""

    recall_at_1: float
    recall_at_5: float
    recall_at_10: float

    def __post_init__(self) -> None:
        """Reject invalid scores before they reach a package report or API."""
        for value in (self.recall_at_1, self.recall_at_5, self.recall_at_10):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise EvaluationCacheError("retrieval score must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class RetrievalEmbeddings:
    """CPU-resident gallery/query embeddings and their person identities."""

    image_embeddings: object
    text_embeddings: object
    image_ids: tuple[int, ...]
    text_ids: tuple[int, ...]


def evaluate_retrieval(
    image_embeddings: object,
    text_embeddings: object,
    image_ids: Sequence[int],
    text_ids: Sequence[int],
) -> RetrievalScores:
    """Score macro text queries using same-person relevance and stable tie order."""
    return RetrievalScores(
        recall_at_1=text_to_image_recall_at_k(
            image_embeddings,
            text_embeddings,
            image_ids,
            text_ids,
            k=1,
        ),
        recall_at_5=text_to_image_recall_at_k(
            image_embeddings,
            text_embeddings,
            image_ids,
            text_ids,
            k=5,
        ),
        recall_at_10=text_to_image_recall_at_k(
            image_embeddings,
            text_embeddings,
            image_ids,
            text_ids,
            k=10,
        ),
    )


__all__ = [
    "EvaluationCacheError",
    "EvaluationCancelledError",
    "RetrievalEmbeddings",
    "RetrievalScores",
    "evaluate_retrieval",
]
