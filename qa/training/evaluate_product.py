"""Delegate product eligibility evaluation to the single policy-owning CLI."""

# This standalone QA executable preserves the existing product evaluator contract.
# ruff: noqa: INP001, TRY003, EM101

from __future__ import annotations

import runpy
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable


def _evaluator_main() -> Callable[[], int]:
    """Load the established evaluator's main function without copying its policy."""
    evaluator = Path(__file__).resolve().parents[1] / "clip" / "evaluate_product_cases.py"
    module = runpy.run_path(str(evaluator))
    main = module.get("main")
    if not callable(main):
        raise TypeError("product quality evaluator entrypoint is unavailable")
    return cast("Callable[[], int]", main)


def main() -> int:
    """Run the established product-cases evaluator with its original CLI args."""
    return _evaluator_main()()


if __name__ == "__main__":
    raise SystemExit(main())
