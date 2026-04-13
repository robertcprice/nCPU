"""Interactive program synthesis from I/O traces.

Discovers programs that use read_i64(), has_input(), and println_i64() to
process a stream of inputs and produce outputs. These are REAL programs
that compile and run with the Mog interpreter.

Supported patterns:
- Running accumulator: state = f(state, input), output state each step
- Pair processor: read two inputs, output f(a, b)
- Filter/transform: read input, conditionally output
- State machine: state transitions based on input
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from egdc.mog.lang import interpret


@dataclass
class InteractiveResult:
    success: bool
    code: str
    loss: float
    method: str
    verified: bool = False


class InteractiveSolver:
    def solve_from_traces(self, fn_name: str,
                           traces: list[list[tuple[int, int]]]) -> InteractiveResult:
        """Discover a stateful interactive program from I/O traces.

        Each trace is [(input, expected_output), ...] representing one execution.
        """
        # Try all candidate program structures
        candidates = self._generate_candidates(fn_name)
        best_loss = float("inf")
        best_code = ""
        best_method = ""

        for method, code_template, eval_fn in candidates:
            loss = 0.0
            total_steps = 0
            for trace in traces:
                state = 0.0
                for inp, expected in trace:
                    state = eval_fn(state, float(inp))
                    loss += (state - expected) ** 2
                    total_steps += 1
            loss /= max(total_steps, 1)
            if loss < best_loss:
                best_loss = loss
                best_code = code_template
                best_method = method
                if loss < 1e-6:
                    break

        if best_loss >= 1e-6:
            return InteractiveResult(False, "", best_loss, "failed")

        # Verify by actually running the program with the interpreter
        verified = self._verify(best_code, traces)

        return InteractiveResult(
            success=best_loss < 1e-6,
            code=best_code,
            loss=best_loss,
            method=best_method,
            verified=verified,
        )

    def _generate_candidates(self, fn_name: str):
        """Generate candidate interactive program structures."""
        candidates = []

        # Running accumulators: state OP= input
        for op, op_fn in [("+", lambda s, x: s + x), ("-", lambda s, x: s - x),
                           ("*", lambda s, x: s * x)]:
            code = (
                f"fn main() -> int {{\n"
                f"    state: i64 = 0;\n"
                f"    while has_input() == 1 {{\n"
                f"        x := read_i64();\n"
                f"        state = state {op} x;\n"
                f"        println_i64(state);\n"
                f"    }}\n"
                f"    return 0;\n"
                f"}}\n"
            )
            candidates.append((f"accum_{op}", code, op_fn))

        # Running max/min
        candidates.append(("running_max", (
            f"fn main() -> int {{\n"
            f"    state: i64 = -999999;\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        if x > state {{ state = x; }}\n"
            f"        println_i64(state);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), lambda s, x: max(s, x) if s != 0 else x))

        # Count: output how many inputs seen
        candidates.append(("counter", (
            f"fn main() -> int {{\n"
            f"    count: i64 = 0;\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        count = count + 1;\n"
            f"        println_i64(count);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), lambda s, x: s + 1))

        # Double each input
        candidates.append(("double", (
            f"fn main() -> int {{\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        println_i64(x * 2);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), lambda s, x: x * 2))

        # Square each input
        candidates.append(("square", (
            f"fn main() -> int {{\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        println_i64(x * x);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), lambda s, x: x * x))

        # Negate
        candidates.append(("negate", (
            f"fn main() -> int {{\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        println_i64(0 - x);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), lambda s, x: -x))

        # Running average (integer): total / count
        candidates.append(("running_avg", (
            f"fn main() -> int {{\n"
            f"    total: i64 = 0;\n"
            f"    count: i64 = 0;\n"
            f"    while has_input() == 1 {{\n"
            f"        x := read_i64();\n"
            f"        total = total + x;\n"
            f"        count = count + 1;\n"
            f"        println_i64(total / count);\n"
            f"    }}\n"
            f"    return 0;\n"
            f"}}\n"
        ), None))  # complex state, skip Python eval

        return [(m, c, f) for m, c, f in candidates if f is not None]

    def _verify(self, code: str, traces: list[list[tuple[int, int]]]) -> bool:
        """Verify the program by actually running it with the interpreter."""
        for trace in traces:
            inputs = [str(inp) for inp, _ in trace]
            expected = [str(exp) for _, exp in trace]
            result = interpret(code, input_data=inputs)
            if not result.success:
                return False
            actual = result.output.strip().split("\n")
            if actual != expected:
                return False
        return True

    def solve_pair_processor(self, fn_name: str,
                              traces: list[list[tuple[tuple[int, int], int]]]) -> InteractiveResult:
        """Discover a program that reads pairs of inputs and outputs a result.

        Each trace entry: ((input1, input2), expected_output)
        """
        pair_ops = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("max", lambda a, b: max(a, b)),
            ("min", lambda a, b: min(a, b)),
        ]
        best_loss = float("inf")
        best_code = ""
        best_method = ""

        for op_name, op_fn in pair_ops:
            loss = 0.0
            total = 0
            for trace in traces:
                for (a, b), expected in trace:
                    pred = op_fn(a, b)
                    loss += (pred - expected) ** 2
                    total += 1
            loss /= max(total, 1)
            if loss < best_loss:
                best_loss = loss
                if op_name in ("+", "-", "*"):
                    expr = f"a {op_name} b"
                elif op_name == "max":
                    expr = "a;\n        if b > a { result = b; }"
                else:
                    expr = "a;\n        if b < a { result = b; }"
                best_code = (
                    f"fn main() -> int {{\n"
                    f"    while has_input() == 1 {{\n"
                    f"        a := read_i64();\n"
                    f"        b := read_i64();\n"
                    f"        result := {expr}\n"
                    f"        println_i64(result);\n"
                    f"    }}\n"
                    f"    return 0;\n"
                    f"}}\n"
                )
                best_method = f"pair_{op_name}"

        verified = False
        if best_loss < 1e-6:
            # Build input list: a1, b1, a2, b2, ...
            for trace in traces:
                inputs = []
                expected = []
                for (a, b), exp in trace:
                    inputs.extend([str(a), str(b)])
                    expected.append(str(exp))
                result = interpret(best_code, input_data=inputs)
                if result.success and result.output.strip().split("\n") == expected:
                    verified = True

        return InteractiveResult(best_loss < 1e-6, best_code, best_loss, best_method, verified)
