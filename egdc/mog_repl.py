#!/usr/bin/env python3
"""Interactive Mog program synthesis REPL.

Usage:
    python -m egdc.mog_repl

You describe what you want, give examples, and the system synthesizes a real
Mog program, compiles it, and runs it.

This is the user-facing interface to the differentiable program synthesis system.
"""

from __future__ import annotations

import sys
from pathlib import Path

from egdc.mog_execute import execute_mog
from egdc.mog_program_search import robust_search_program
from egdc.mog_search_solver import (
    _fast_arithmetic, _gcd_loop_refinement, _lcm_search,
    _modulo_check_search, _factorial_search, _fibonacci_search,
    _digit_sum_search, _branching_refinement, _loop_accum_refinement,
    _two_branch_refinement, SoftBranchingProgram,
)
from egdc.mog_compositional import CompositionalSolver
from egdc.mog_counterexample import CounterexampleRefiner
from egdc.mog_pathways import PathwayMemory


class MogREPL:
    def __init__(self, memory_root: str = "egdc/pathway_memory"):
        self.memory = PathwayMemory(memory_root)
        self.composer = CompositionalSolver()
        self.refiner = CounterexampleRefiner()
        self.solved: dict[str, str] = {}  # name -> code

    def synthesize(self, fn_name: str, arg_names: list[str],
                    examples: list[tuple[tuple[float, ...], float]]) -> str | None:
        """Synthesize a Mog program from examples."""

        # Cascading search
        searches = [
            ("arithmetic", lambda: _fast_arithmetic(arg_names, examples, fn_name)),
            ("gcd", lambda: _gcd_loop_refinement(arg_names, examples, fn_name)),
            ("lcm", lambda: _lcm_search(arg_names, examples, fn_name)),
            ("modulo", lambda: _modulo_check_search(arg_names, examples, fn_name)),
            ("factorial", lambda: _factorial_search(arg_names, examples, fn_name)),
            ("fibonacci", lambda: _fibonacci_search(arg_names, examples, fn_name)),
            ("digit_sum", lambda: _digit_sum_search(arg_names, examples, fn_name)),
            ("branch", lambda: _branching_refinement(
                SoftBranchingProgram(num_args=len(arg_names)), arg_names, examples, fn_name)),
            ("loop", lambda: _loop_accum_refinement(arg_names, examples, fn_name)),
        ]
        if len(arg_names) <= 1:
            searches.append(("two_branch", lambda: _two_branch_refinement(arg_names, examples, fn_name)))

        # Also try composition with known sub-programs
        for sub_name, sub_code in self.solved.items():
            self.composer.register_subprogram(sub_name, arg_names, sub_code)

        for method, search_fn in searches:
            code, loss = search_fn()
            if loss < 1e-6:
                self.solved[fn_name] = code
                self.memory.record_success(fn_name, method, code,
                    {"description": f"synthesized {fn_name}", "signature": f"fn {fn_name}(...)"})
                self.memory.save()
                return code

        # Try composition
        result = self.composer.solve(fn_name, arg_names, examples)
        if result.success:
            self.solved[fn_name] = result.code
            self.memory.record_success(fn_name, "composition", result.code, {})
            self.memory.save()
            return result.code

        return None

    def compile_and_run(self, code: str, test_input: str = "") -> str | None:
        """Compile and run a Mog program, return stdout."""
        result = execute_mog(code)
        if result.success:
            return result.stdout.strip()
        return f"ERROR: {result.stderr or result.compile_stderr or result.error}"

    def interactive_session(self):
        """Run an interactive REPL session."""
        print("=== Mog Program Synthesis REPL ===")
        print("Describe a function, give examples, and I'll synthesize it.\n")
        print("Commands:")
        print("  synth <name> <arg1,arg2,...>  — start synthesizing a function")
        print("  ex <input1,input2,...> = <output>  — add an example")
        print("  go  — synthesize from current examples")
        print("  run <arg1,arg2,...>  — run the last synthesized function")
        print("  list  — show all synthesized functions")
        print("  quit  — exit\n")

        current_name = None
        current_args = []
        current_examples = []
        last_code = None

        while True:
            try:
                line = input("mog> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not line:
                continue

            if line == "quit":
                break

            if line == "list":
                if self.solved:
                    for name, code in self.solved.items():
                        print(f"\n--- {name} ---")
                        print(code)
                else:
                    print("No functions synthesized yet.")
                continue

            if line.startswith("synth "):
                parts = line[6:].strip().split()
                if len(parts) < 2:
                    print("Usage: synth <name> <arg1,arg2,...>")
                    continue
                current_name = parts[0]
                current_args = parts[1].split(",")
                current_examples = []
                print(f"Synthesizing {current_name}({', '.join(current_args)})")
                print("Add examples with: ex <inputs> = <output>")
                continue

            if line.startswith("ex "):
                if not current_name:
                    print("Start with: synth <name> <args>")
                    continue
                try:
                    lhs, rhs = line[3:].split("=")
                    inputs = tuple(float(x.strip()) for x in lhs.split(","))
                    output = float(rhs.strip())
                    current_examples.append((inputs, output))
                    print(f"  Added: {current_name}({', '.join(str(int(x)) for x in inputs)}) = {int(output)}")
                except Exception as e:
                    print(f"  Parse error: {e}")
                continue

            if line == "go":
                if not current_name or not current_examples:
                    print("Need a function name and at least one example.")
                    continue
                print(f"\nSearching for {current_name}...")
                code = self.synthesize(current_name, current_args, current_examples)
                if code:
                    last_code = code
                    print(f"\nDiscovered:")
                    print(code)
                    # Verify with compiler
                    args_str = ", ".join(str(int(x)) for x in current_examples[0][0])
                    test = code + f"\nfn main() -> int {{ println_i64({current_name}({args_str})); return 0; }}"
                    out = self.compile_and_run(test)
                    print(f"Compiled and ran: {current_name}({args_str}) = {out}")
                else:
                    print("Could not find a matching program.")
                continue

            if line.startswith("run "):
                if not current_name or current_name not in self.solved:
                    print("No function to run. Synthesize one first.")
                    continue
                try:
                    args = [int(x.strip()) for x in line[4:].split(",")]
                    args_str = ", ".join(str(a) for a in args)
                    code = self.solved[current_name]
                    test = code + f"\nfn main() -> int {{ println_i64({current_name}({args_str})); return 0; }}"
                    out = self.compile_and_run(test)
                    print(f"{current_name}({args_str}) = {out}")
                except Exception as e:
                    print(f"Error: {e}")
                continue

            print(f"Unknown command: {line}")


def main():
    repl = MogREPL()
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        _run_demo(repl)
    else:
        repl.interactive_session()


def _run_demo(repl: MogREPL):
    """Run a scripted demo showing the system in action."""
    print("=== Mog Synthesis Demo ===\n")

    # 1. Synthesize max2
    print("--- Synthesizing max2 ---")
    code = repl.synthesize("max2", ["a", "b"], [
        ((2.0, 3.0), 3.0), ((10.0, -4.0), 10.0), ((7.0, 7.0), 7.0), ((-3.0, -2.0), -2.0),
    ])
    print(code)

    # Run on new inputs
    for a, b in [(100, 1), (-5, -3), (0, 0)]:
        test = code + f"\nfn main() -> int {{ println_i64(max2({a}, {b})); return 0; }}"
        print(f"  max2({a}, {b}) = {repl.compile_and_run(test)}")

    # 2. Synthesize gcd
    print("\n--- Synthesizing gcd ---")
    code = repl.synthesize("gcd", ["a", "b"], [
        ((12.0, 18.0), 6.0), ((21.0, 14.0), 7.0), ((9.0, 28.0), 1.0),
    ])
    print(code)

    # 3. Compose LCM from discovered GCD
    print("\n--- Composing lcm from gcd ---")
    code = repl.synthesize("lcm", ["a", "b"], [
        ((3.0, 4.0), 12.0), ((6.0, 8.0), 24.0), ((5.0, 10.0), 10.0),
    ])
    print(code)
    test = code + "\nfn main() -> int { println_i64(lcm(12, 18)); return 0; }"
    print(f"  lcm(12, 18) = {repl.compile_and_run(test)}")

    # 4. Synthesize sum_to_n
    print("\n--- Synthesizing sum_to_n ---")
    code = repl.synthesize("sum_to_n", ["n"], [
        ((0.0,), 0.0), ((1.0,), 1.0), ((5.0,), 15.0), ((10.0,), 55.0),
    ])
    print(code)
    test = code + "\nfn main() -> int { println_i64(sum_to_n(100)); return 0; }"
    print(f"  sum_to_n(100) = {repl.compile_and_run(test)}")

    # 5. Counterexample refinement
    print("\n--- Counterexample refinement ---")
    print("Bad program: fn double(x) { return x + 1; }")
    result = repl.refiner.refine("double", ["x"],
        "fn double(x: i64) -> i64 { return x + 1; }",
        [((2.0,), 4.0)], [((5.0,), 10.0), ((0.0,), 0.0)])
    print(f"Refined: {result.code}")

    print("\n--- Memory state ---")
    print(f"Total pathways stored: {repl.memory.total_successes()}")
    print(f"Families: {repl.memory.successes_by_family()}")

    print("\nDone.")


if __name__ == "__main__":
    main()
