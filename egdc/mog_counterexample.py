"""Counterexample-guided program refinement.

When a discovered program fails on a new input, use the failure to refine
the program instead of starting the search over from scratch.

Algorithm:
1. Take the failing program + the new counterexample
2. Add counterexample to the training set
3. Re-run the search with the expanded example set
4. If the new program still fails, repeat with more counterexamples
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from egdc.mog_program_search import (
    robust_search_program, RobustSearchResult,
    _eval_code_on_examples,
)


@dataclass
class RefinementResult:
    success: bool
    code: str
    loss: float
    iterations: int
    metadata: dict[str, Any]


class CounterexampleRefiner:
    def __init__(self, max_iterations: int = 3):
        self.max_iterations = max_iterations

    def refine(
        self,
        fn_name: str,
        arg_names: list[str],
        initial_code: str,
        train_examples: list[tuple[tuple[float, ...], float]],
        counterexamples: list[tuple[tuple[float, ...], float]],
    ) -> RefinementResult:
        """Refine a program by incorporating counterexamples."""
        all_examples = list(train_examples) + list(counterexamples)

        for iteration in range(self.max_iterations):
            result = robust_search_program(
                arg_names=arg_names,
                train_examples=all_examples,
                holdout_examples=[],
                function_name=fn_name,
                seed=iteration * 100,
            )

            if result.success and result.loss < 1e-6:
                # Verify on counterexamples specifically
                ce_loss = _eval_code_on_examples(result.code, arg_names, counterexamples)
                if ce_loss < 1e-6:
                    return RefinementResult(True, result.code, result.loss, iteration + 1, {})

        return RefinementResult(False, "", float("inf"), self.max_iterations, {})
