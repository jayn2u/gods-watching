import pytest
from qa.training import evaluate_product


def test_training_product_entrypoint_delegates_to_existing_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def evaluator_main() -> int:
        nonlocal calls
        calls += 1
        return 7

    monkeypatch.setattr(evaluate_product, "_evaluator_main", lambda: evaluator_main)

    assert evaluate_product.main() == 7
    assert calls == 1
