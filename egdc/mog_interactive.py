"""Interactive program synthesis from I/O traces.

An I/O trace is a sequence of (input, expected_output) pairs that represent
a stateful computation — like a running sum, a counter, or a state machine.

The solver discovers a program that maintains state and processes inputs
sequentially to produce the correct outputs.

Program structure:
    state: i64 = init;
    for each input x:
        state = f(state, x);
        output state;
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from egdc.mog_program_search import _py_eval_expr


@dataclass
class InteractiveResult:
    success: bool
    code: str
    loss: float
    method: str


class InteractiveSolver:
    def solve_from_traces(self, fn_name: str,
                           traces: list[list[tuple[int, int]]]) -> InteractiveResult:
        """Discover a stateful program from I/O traces.

        Each trace is a list of (input, expected_output) pairs representing
        sequential processing with hidden state.
        """
        # Search over: state update function f(state, input) -> new_state
        # where output = new_state

        INITS = [0, 1]
        OPS = ["+", "-", "*"]

        # Candidate state update expressions in terms of (state, x)
        update_candidates = [
            ("state + x", lambda s, x: s + x),
            ("state - x", lambda s, x: s - x),
            ("state * x", lambda s, x: s * x),
            ("state + 1", lambda s, x: s + 1),
            ("state - 1", lambda s, x: s - 1),
            ("x", lambda s, x: x),
            ("state + x * 2", lambda s, x: s + x * 2),
            ("state + x * x", lambda s, x: s + x * x),
        ]

        best_loss = float("inf")
        best_code = ""

        for init in INITS:
            for update_str, update_fn in update_candidates:
                loss = 0.0
                for trace in traces:
                    state = float(init)
                    for inp, expected in trace:
                        state = update_fn(state, float(inp))
                        loss += (state - expected) ** 2
                loss /= sum(len(t) for t in traces)
                if loss < best_loss:
                    best_loss = loss
                    # Generate Mog code for the interactive program
                    best_code = (
                        f"fn {fn_name}(inputs: [i64]) -> [i64] {{\n"
                        f"    state: i64 = {init};\n"
                        f"    results: [i64] = [];\n"
                        f"    for x in inputs {{\n"
                        f"        state = {update_str};\n"
                        f"        results = results.push(state);\n"
                        f"    }}\n"
                        f"    return results;\n"
                        f"}}\n"
                    )
                    if loss < 1e-6:
                        return InteractiveResult(True, best_code, loss, "trace_search")

        # If array-returning function doesn't work in Mog, generate a while-loop version
        if best_loss < 1e-6:
            return InteractiveResult(True, best_code, best_loss, "trace_search")

        # Also try a simpler version: just the state update function
        for init in INITS:
            for update_str, update_fn in update_candidates:
                loss = 0.0
                for trace in traces:
                    state = float(init)
                    for inp, expected in trace:
                        state = update_fn(state, float(inp))
                        loss += (state - expected) ** 2
                loss /= sum(len(t) for t in traces)
                if loss < best_loss:
                    best_loss = loss
                    best_code = (
                        f"fn {fn_name}_step(state: i64, x: i64) -> i64 {{\n"
                        f"    return {update_str};\n"
                        f"}}\n\n"
                        f"fn {fn_name}(inputs: [i64]) -> i64 {{\n"
                        f"    state: i64 = {init};\n"
                        f"    for x in inputs {{\n"
                        f"        state = {fn_name}_step(state, x);\n"
                        f"    }}\n"
                        f"    return state;\n"
                        f"}}\n"
                    )

        success = best_loss < 1e-6
        return InteractiveResult(success, best_code, best_loss, "trace_search")
