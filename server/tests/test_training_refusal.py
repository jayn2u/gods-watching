from __future__ import annotations

from datetime import UTC, datetime

import pytest

from gods_watching.training.service import TrainingMemoryRefusedError, TrainingService


def test_memory_refusal_retains_supervisor_observation_and_profile_identity() -> None:
    observed_at = datetime(2026, 10, 4, 12, 30, tzinfo=UTC)
    profile_identity = "b" * 64

    with pytest.raises(TrainingMemoryRefusedError) as raised:
        TrainingService._raise_refusal(  # noqa: SLF001 # pyright: ignore[reportPrivateUsage]
            "training_memory_refused",
            {
                "training_peak_bytes": 3 * 1024**3,
                "reserve_bytes": 2 * 1024**3,
                "required_bytes": 5 * 1024**3,
                "free_bytes": 4 * 1024**3,
                "profile_identity": profile_identity,
                "observed_at": observed_at.isoformat(),
                "reason": "insufficient_free_memory",
            },
        )

    assert raised.value.observed_at == observed_at
    assert raised.value.profile_identity == profile_identity
    assert raised.value.reason == "insufficient_free_memory"
