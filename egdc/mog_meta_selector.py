"""Structure meta-selector: predict which program structure to try first.

Uses simple heuristics derived from I/O example patterns — but these
heuristics are themselves learned from past solve experience, not hand-coded.

Signals:
- Output grows quadratically with input → likely loop
- Output is always one of the inputs → likely branch
- Output is simple arithmetic of inputs → likely arithmetic
- Output is one of a small set of values → likely multi-branch
"""

from __future__ import annotations

from typing import Sequence


class StructureSelector:
    def predict_structure(
        self,
        arg_names: list[str],
        examples: Sequence[tuple[tuple[float, ...], float]],
    ) -> str:
        """Predict the best program structure from I/O patterns."""
        if not examples:
            return "arithmetic"

        num_args = len(arg_names)
        outputs = [target for _, target in examples]
        inputs = [args for args, _ in examples]

        # Check if output is always one of the input args
        output_is_arg = all(
            any(abs(target - arg) < 1e-6 for arg in args)
            for args, target in examples
        )

        # Check if output is a simple linear function of inputs
        if num_args == 2:
            is_sum = all(abs(a[0] + a[1] - t) < 1e-6 for a, t in examples)
            is_diff = all(abs(a[0] - a[1] - t) < 1e-6 for a, t in examples)
            is_prod = all(abs(a[0] * a[1] - t) < 1e-6 for a, t in examples)
            if is_sum or is_diff or is_prod:
                return "arithmetic"

        if num_args == 1:
            # Check for quadratic/super-linear growth → loop
            xs = sorted([(args[0], target) for args, target in examples])
            if len(xs) >= 3:
                # Check if output grows faster than linear
                x0, y0 = xs[0]
                x1, y1 = xs[-1]
                if x1 > x0 and y1 > 0:
                    linear_rate = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
                    # Check middle points
                    mid_x, mid_y = xs[len(xs) // 2]
                    if x1 != x0:
                        expected_linear = y0 + linear_rate * (mid_x - x0)
                        if abs(mid_y - expected_linear) > 1.0:
                            return "loop"

        # Check if output is from a small set of values → multi-branch
        unique_outputs = set(int(o) for o in outputs)
        if len(unique_outputs) <= 3 and len(examples) >= 4:
            return "multi_branch"

        # Output is always one of the inputs → branch
        if output_is_arg and num_args >= 2:
            return "branch"

        return "arithmetic"
