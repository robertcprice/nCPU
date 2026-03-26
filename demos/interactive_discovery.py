#!/usr/bin/env python3
"""Interactive Program Discovery REPL.

Type input-output examples, watch the differentiable CPU discover a program
in real-time. Then test it on new inputs interactively.

This is the killer demo: you describe WHAT you want a program to do,
and gradient descent figures out HOW.

Run: python demos/interactive_discovery.py
"""

from __future__ import annotations

import readline  # enables arrow key history in input()
import sys
import time

import torch

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    Instruction,
    SoftProgram,
    OPCODES,
    _OP,
)
from ncpu.differentiable.program_synthesis import ProgramSynthesizer, SynthesisSpec


BANNER = r"""
    ╔══════════════════════════════════════════════════════════╗
    ║          nCPU Interactive Program Discovery              ║
    ║                                                          ║
    ║  Describe what a program should do with examples.        ║
    ║  Gradient descent discovers the program for you.         ║
    ║                                                          ║
    ║  No search. No enumeration. Just backpropagation         ║
    ║  through a differentiable CPU.                           ║
    ╚══════════════════════════════════════════════════════════╝
"""

HELP = """
Commands:
  example <inputs> -> <outputs>    Add an input-output example
    e.g.:  example 3, 5 -> 8       (R0=3, R1=5, expect R2=8)
    e.g.:  example 7 -> 49         (R0=7, expect R1=49)

  synthesize [iters] [slots]       Discover a program from your examples
    e.g.:  synthesize              (default: 3000 iters, 6 slots)
    e.g.:  synthesize 5000 8       (5000 iters, 8 slots)

  test <inputs>                    Test the discovered program on new inputs
    e.g.:  test 4, 6               (run with R0=4, R1=6)

  show                             Show current examples
  clear                            Clear all examples
  preset <name>                    Load a preset example set
    Presets: add, mul, fib, square, double, dot, cube

  help                             Show this help
  quit                             Exit

Tips:
  - Start simple: 5-10 examples of a single operation
  - Inputs map to R0, R1, R2, ... (left to right)
  - Outputs map to the next registers after inputs
  - More examples = better generalization
"""


class InteractiveDiscovery:
    """Interactive REPL for program discovery."""

    def __init__(self):
        self.examples: list[tuple[dict[int, float], dict[int, float]]] = []
        self.engine = DifferentiableEngine()
        self.last_program: SoftProgram | None = None
        self.last_result = None
        self.num_input_regs = 0
        self.num_output_regs = 0

    def parse_example(self, line: str):
        """Parse 'example 3, 5 -> 8' into ({0:3, 1:5}, {2:8})."""
        parts = line.split("->")
        if len(parts) != 2:
            print("  Error: use format 'example <inputs> -> <outputs>'")
            return None

        inputs_str = parts[0].strip()
        outputs_str = parts[1].strip()

        try:
            input_vals = [float(x.strip()) for x in inputs_str.split(",")]
            output_vals = [float(x.strip()) for x in outputs_str.split(",")]
        except ValueError:
            print("  Error: values must be numbers")
            return None

        n_in = len(input_vals)
        n_out = len(output_vals)

        # Auto-detect register mapping
        if not self.examples:
            self.num_input_regs = n_in
            self.num_output_regs = n_out
        elif n_in != self.num_input_regs or n_out != self.num_output_regs:
            print(
                f"  Error: expected {self.num_input_regs} inputs and "
                f"{self.num_output_regs} outputs (got {n_in}, {n_out})"
            )
            return None

        inputs = {i: v for i, v in enumerate(input_vals)}
        outputs = {self.num_input_regs + i: v for i, v in enumerate(output_vals)}
        return inputs, outputs

    def add_example(self, line: str):
        result = self.parse_example(line)
        if result is None:
            return
        inputs, outputs = result
        self.examples.append((inputs, outputs))
        in_str = ", ".join(f"R{i}={v:.0f}" for i, v in inputs.items())
        out_str = ", ".join(f"R{i}={v:.0f}" for i, v in outputs.items())
        print(f"  Added: [{in_str}] -> [{out_str}]  ({len(self.examples)} examples)")

    def show_examples(self):
        if not self.examples:
            print("  No examples yet. Use 'example' to add some.")
            return
        print(f"  {len(self.examples)} examples:")
        for i, (inp, out) in enumerate(self.examples):
            in_str = ", ".join(f"R{k}={v:.0f}" for k, v in inp.items())
            out_str = ", ".join(f"R{k}={v:.0f}" for k, v in out.items())
            print(f"    {i + 1:3d}. [{in_str}] -> [{out_str}]")

    def load_preset(self, name: str):
        """Load a preset example set."""
        import random
        random.seed(42)

        self.examples.clear()
        presets = {
            "add": (2, 1, lambda a, b: [a + b]),
            "mul": (2, 1, lambda a, b: [a * b]),
            "square": (1, 1, lambda a: [a * a]),
            "double": (1, 1, lambda a: [a * 2]),
            "cube": (1, 1, lambda a: [a * a * a]),
            "fib": (2, 2, lambda a, b: [b, a + b]),
            "dot": (4, 1, lambda a, b, c, d: [a * c + b * d]),
        }

        if name not in presets:
            print(f"  Unknown preset. Available: {', '.join(presets.keys())}")
            return

        n_in, n_out, fn = presets[name]
        self.num_input_regs = n_in
        self.num_output_regs = n_out

        for _ in range(20):
            args = [random.randint(1, 12) for _ in range(n_in)]
            inputs = {i: float(v) for i, v in enumerate(args)}
            results = fn(*args)
            outputs = {n_in + i: float(v) for i, v in enumerate(results)}
            self.examples.append((inputs, outputs))

        print(f"  Loaded preset '{name}': {len(self.examples)} examples")
        self.show_examples()

    def synthesize(self, max_iters: int = 3000, max_len: int = 6):
        """Run program synthesis on current examples."""
        if len(self.examples) < 3:
            print("  Need at least 3 examples. Add more with 'example'.")
            return

        spec = SynthesisSpec(self.examples)
        synth = ProgramSynthesizer(max_program_len=max_len, lr=0.02)

        print(f"\n  Synthesizing from {len(self.examples)} examples...")
        print(f"  Program slots: {max_len}, Max iterations: {max_iters}")
        print(f"  {'─' * 60}")

        t0 = time.time()
        result = synth.synthesize(
            spec,
            max_iters=max_iters,
            verbose=True,
            print_every=max(max_iters // 6, 100),
            skip_bitwise=True,
            max_exec_steps=max_len + 2,
            initial_temperature=2.0,
            final_temperature=0.1,
        )
        elapsed = time.time() - t0

        self.last_program = result.program
        self.last_result = result

        print(f"  {'─' * 60}")
        print(f"\n  Discovered program:")
        for line in result.program_text.split("\n"):
            print(f"    {line}")
        print(f"\n  Accuracy: {result.accuracy:.0%}")
        print(f"  Loss: {result.loss_history[-1]:.4f}")
        print(f"  Time: {elapsed:.1f}s")

        if result.accuracy == 1.0:
            print("\n  Program is ready! Use 'test <inputs>' to try it.")
        else:
            print(
                f"\n  Partial match ({result.accuracy:.0%}). "
                f"Try more examples or 'synthesize {max_iters * 2} {max_len + 2}'"
            )

    def test_program(self, line: str):
        """Test the discovered program on new inputs."""
        if self.last_program is None:
            print("  No program yet. Run 'synthesize' first.")
            return

        try:
            vals = [float(x.strip()) for x in line.split(",")]
        except ValueError:
            print("  Error: values must be numbers (e.g., 'test 3, 5')")
            return

        inputs = {i: v for i, v in enumerate(vals)}

        with torch.no_grad():
            result = self.engine.execute_soft(
                self.last_program,
                inputs,
                temperature=0.1,
                max_steps=16,
                skip_bitwise=True,
            )

        in_str = ", ".join(f"R{i}={v:.0f}" for i, v in inputs.items())
        out_regs = range(self.num_input_regs, self.num_input_regs + self.num_output_regs)
        out_str = ", ".join(
            f"R{i}={result.registers[i].item():.2f}" for i in out_regs
        )
        print(f"  Input:  [{in_str}]")
        print(f"  Output: [{out_str}]")

    def run(self):
        """Main REPL loop."""
        print(BANNER)
        print("  Type 'help' for commands, 'preset add' to get started quickly.\n")

        while True:
            try:
                line = input("ncpu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Goodbye!")
                break

            if not line:
                continue

            cmd = line.split()[0].lower()
            rest = line[len(cmd):].strip()

            if cmd in ("quit", "exit", "q"):
                print("  Goodbye!")
                break
            elif cmd == "help":
                print(HELP)
            elif cmd == "example":
                self.add_example(rest)
            elif cmd == "show":
                self.show_examples()
            elif cmd == "clear":
                self.examples.clear()
                self.last_program = None
                self.num_input_regs = 0
                self.num_output_regs = 0
                print("  Cleared all examples.")
            elif cmd == "preset":
                self.load_preset(rest)
            elif cmd == "synthesize" or cmd == "synth" or cmd == "s":
                parts = rest.split()
                iters = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 3000
                slots = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 6
                self.synthesize(max_iters=iters, max_len=slots)
            elif cmd == "test" or cmd == "t":
                self.test_program(rest)
            else:
                # Maybe they typed an example without the 'example' prefix
                if "->" in line:
                    self.add_example(line)
                else:
                    print(f"  Unknown command: {cmd}. Type 'help' for commands.")


def main():
    # If running non-interactively (piped input), do a quick demo
    if not sys.stdin.isatty():
        print("Running non-interactive demo...")
        repl = InteractiveDiscovery()
        repl.load_preset("add")
        repl.synthesize(max_iters=1500, max_len=4)
        repl.test_program("15, 25")
        repl.test_program("100, 200")
        return

    repl = InteractiveDiscovery()
    repl.run()


if __name__ == "__main__":
    main()
