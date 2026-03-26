"""Differentiable program synthesis: discover programs from specifications.

This is the most novel capability of nCPU. Programs are represented as
continuous parameters, executed through a differentiable CPU, and optimized
via backpropagation to match desired input-output behavior.

The key insight: because every component of the CPU --- ALU, register file,
program counter, branch logic --- is differentiable, we can define a loss
function over program *behavior* and use gradient descent to search the
space of all possible programs.

Key techniques:
  - Gumbel-softmax for discrete instruction choices (temperature annealing)
  - Soft register addressing via attention weights
  - Multi-example loss for generalization beyond memorization
  - Temperature annealing schedule (exploration -> exploitation)
  - Length regularization to prefer shorter, simpler programs
  - Gradient clipping for training stability

Example usage:
    >>> spec = make_addition_spec(n_examples=20)
    >>> synthesizer = ProgramSynthesizer()
    >>> result = synthesizer.synthesize(spec, verbose=True)
    >>> print(result.program_text)
       0: ADD R2, R0, R1
       1: HALT
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    OPCODES,
    NUM_OPCODES,
)


# ---------------------------------------------------------------------------
# Specification and result types
# ---------------------------------------------------------------------------

@dataclass
class SynthesisSpec:
    """Specification for program synthesis: input-output examples.

    Each example is a pair (input_registers, expected_output_registers).
    Registers are indexed by integer, values are floats.

    Example:
        ({0: 5.0, 1: 3.0}, {2: 8.0})
        means: given R0=5, R1=3, the program should produce R2=8.
    """

    examples: list[tuple[dict[int, float], dict[int, float]]]

    def __post_init__(self) -> None:
        if not self.examples:
            raise ValueError("SynthesisSpec requires at least one example.")
        for i, (inputs, targets) in enumerate(self.examples):
            if not targets:
                raise ValueError(
                    f"Example {i} has no target registers; nothing to optimize."
                )

    @property
    def num_examples(self) -> int:
        return len(self.examples)

    @property
    def input_registers(self) -> set[int]:
        """All register indices used as inputs across examples."""
        regs: set[int] = set()
        for inputs, _ in self.examples:
            regs.update(inputs.keys())
        return regs

    @property
    def output_registers(self) -> set[int]:
        """All register indices used as targets across examples."""
        regs: set[int] = set()
        for _, targets in self.examples:
            regs.update(targets.keys())
        return regs


@dataclass
class SynthesisResult:
    """Result of program synthesis via gradient descent."""

    program: SoftProgram
    discrete_program: list  # list[Instruction] after extraction
    program_text: str       # Pretty-printed discovered program
    loss_history: list[float]
    converged: bool
    steps: int
    accuracy: float         # Fraction of examples solved (within tolerance)
    discretization_gap: Optional[float] = None  # soft_accuracy - discrete_accuracy
    verification: Optional[dict] = None  # Full verify_discrete output


# ---------------------------------------------------------------------------
# Program Synthesizer
# ---------------------------------------------------------------------------

class ProgramSynthesizer:
    """Discover programs from input-output specifications via gradient descent.

    This is the central novelty of nCPU: represent a program as continuous
    parameters, execute through a differentiable CPU, and optimize via
    backpropagation to match desired behavior.

    The search operates in a continuous relaxation of program space:
      - Opcodes are probability distributions over the instruction set
      - Register operands are attention weights over the register file
      - Immediates are unbounded real-valued parameters
      - Branch targets are distributions over instruction positions

    Temperature annealing drives the continuous solution toward a discrete
    program: high temperature enables gradient-based exploration across the
    full instruction space, low temperature sharpens choices toward a single
    instruction per slot.

    Regularization encourages shorter programs by penalizing non-trivial
    instructions at later positions, biasing the search toward the simplest
    program that satisfies the specification.
    """

    def __init__(
        self,
        engine: Optional[DifferentiableEngine] = None,
        max_program_len: int = 12,
        num_registers: int = 8,
        lr: float = 0.01,
    ):
        """Initialize the synthesizer.

        Args:
            engine: Differentiable execution engine (created if None).
            max_program_len: Maximum number of instruction slots in the
                synthesized program.
            num_registers: Number of architectural registers available.
            lr: Learning rate for Adam optimizer.
        """
        self.engine = engine or DifferentiableEngine(num_registers=num_registers)
        self.max_program_len = max_program_len
        self.num_registers = num_registers
        self.lr = lr

    def synthesize(
        self,
        spec: SynthesisSpec,
        max_iters: int = 5000,
        initial_temperature: float = 2.0,
        final_temperature: float = 0.1,
        tolerance: float = 1e-3,
        length_penalty: float = 0.001,
        verbose: bool = False,
        print_every: int = 500,
        skip_bitwise: bool = True,
        max_exec_steps: int = 16,
    ) -> SynthesisResult:
        """Synthesize a program that satisfies the specification.

        The optimization proceeds in three phases driven by temperature:

        1. **Exploration** (high temperature): Gumbel-softmax is soft, allowing
           gradients to flow through all possible instruction choices. The
           optimizer explores the full program space.

        2. **Refinement** (medium temperature): Distributions sharpen around
           promising instruction choices. The program structure crystallizes.

        3. **Discretization** (low temperature): Gumbel-softmax approaches
           hard one-hot vectors. The continuous program converges to a single
           discrete instruction sequence.

        Args:
            spec: Input-output specification to satisfy.
            max_iters: Maximum optimization steps.
            initial_temperature: Starting Gumbel-softmax temperature.
            final_temperature: Final temperature (should be small, ~0.1).
            tolerance: Loss threshold for early stopping.
            length_penalty: Coefficient for program length regularization.
                Higher values prefer shorter programs.
            verbose: Print progress during optimization.
            print_every: Steps between progress reports (when verbose).

        Returns:
            SynthesisResult with the discovered program, convergence info,
            and accuracy on the specification.
        """
        program = SoftProgram(
            max_length=self.max_program_len,
            num_registers=self.num_registers,
        )

        optimizer = torch.optim.Adam(program.parameters(), lr=self.lr)
        loss_history: list[float] = []
        final_loss = float("inf")
        step = -1

        for step in range(max_iters):
            # Temperature annealing: exponential decay from initial to final.
            # At step 0 -> initial_temperature; at step max_iters-1 -> final_temperature.
            progress = step / max(max_iters - 1, 1)
            temperature = initial_temperature * (
                final_temperature / initial_temperature
            ) ** progress

            optimizer.zero_grad()

            # Batched execution: all examples execute in parallel through a
            # single tensor computation, eliminating the Python loop.
            batch_inputs = [inputs for inputs, _ in spec.examples]
            batch_results = self.engine.execute_soft_batched(
                program, batch_inputs, max_steps=max_exec_steps,
                temperature=temperature, skip_bitwise=skip_bitwise,
            )

            # Accumulate loss across all examples (full gradient flow).
            total_loss = torch.tensor(0.0)
            for (inputs, targets), result in zip(spec.examples, batch_results):
                for reg_idx, target_val in targets.items():
                    predicted = result.registers[reg_idx]
                    total_loss = total_loss + (predicted - target_val) ** 2

            # Length regularization: penalize non-trivial instructions at later
            # positions. This encourages the optimizer to concentrate useful work
            # in the first few instruction slots and fill the rest with NOP/HALT.
            if length_penalty > 0:
                opcode_probs = F.softmax(program.opcode_logits, dim=-1)
                # Probability of doing something useful (not NOP or HALT)
                nop_prob = opcode_probs[:, OPCODES["NOP"]]
                halt_prob = opcode_probs[:, OPCODES["HALT"]]
                useful_prob = 1.0 - nop_prob - halt_prob
                # Weight by position: later instructions penalized more heavily
                position_weights = torch.arange(
                    self.max_program_len, dtype=torch.float32
                )
                total_loss = total_loss + length_penalty * (
                    useful_prob * position_weights
                ).sum()

            # Normalize by number of examples for consistent gradients
            # regardless of specification size
            total_loss = total_loss / len(spec.examples)

            total_loss.backward()

            # Gradient clipping prevents instability from Gumbel-softmax
            # noise, especially at low temperatures where gradients spike
            torch.nn.utils.clip_grad_norm_(program.parameters(), 5.0)

            optimizer.step()

            final_loss = total_loss.item()
            loss_history.append(final_loss)

            if verbose and step % print_every == 0:
                acc = self._evaluate_accuracy(
                    program, spec, temperature,
                    skip_bitwise=skip_bitwise, max_exec_steps=max_exec_steps,
                )
                print(
                    f"Step {step:5d} | "
                    f"Loss: {final_loss:.6f} | "
                    f"Temp: {temperature:.3f} | "
                    f"Acc: {acc:.1%}"
                )
                if step % (print_every * 2) == 0:
                    print(program.format_program())

            # Early stopping when loss is negligible
            if final_loss < tolerance:
                break

        # Final evaluation at low temperature (near-discrete)
        accuracy = self._evaluate_accuracy(
            program, spec, final_temperature,
            skip_bitwise=skip_bitwise, max_exec_steps=max_exec_steps,
        )

        result = SynthesisResult(
            program=program,
            discrete_program=program.extract_discrete_program(),
            program_text=program.format_program(),
            loss_history=loss_history,
            converged=final_loss < tolerance,
            steps=step + 1,
            accuracy=accuracy,
        )

        # Discrete verification: run the extracted hard program through
        # execute_fixed and measure the discretization gap.
        verification = self.verify_discrete(result, spec)
        result.discretization_gap = verification["discretization_gap"]
        result.verification = verification

        return result

    def _evaluate_accuracy(
        self,
        program: SoftProgram,
        spec: SynthesisSpec,
        temperature: float,
        threshold: float = 0.5,
        skip_bitwise: bool = True,
        max_exec_steps: int = 16,
    ) -> float:
        """Evaluate what fraction of examples the program solves.

        An example is considered solved if every target register is within
        ``threshold`` of the expected value.

        Args:
            program: The soft program to evaluate.
            spec: Specification with input-output examples.
            temperature: Gumbel-softmax temperature for evaluation.
            threshold: Maximum absolute error to count as correct.
            skip_bitwise: Skip expensive bitwise ops.
            max_exec_steps: Maximum execution steps.

        Returns:
            Fraction of examples correctly solved (0.0 to 1.0).
        """
        correct = 0
        with torch.no_grad():
            batch_inputs = [inputs for inputs, _ in spec.examples]
            batch_results = self.engine.execute_soft_batched(
                program, batch_inputs, max_steps=max_exec_steps,
                temperature=temperature, skip_bitwise=skip_bitwise,
            )
            for (inputs, targets), result in zip(spec.examples, batch_results):
                all_close = True
                for reg_idx, target_val in targets.items():
                    if abs(result.registers[reg_idx].item() - target_val) > threshold:
                        all_close = False
                        break
                if all_close:
                    correct += 1
        return correct / len(spec.examples)

    def verify_discrete(
        self,
        result: SynthesisResult,
        spec: SynthesisSpec,
        threshold: float = 0.5,
        max_exec_steps: int = 64,
    ) -> dict:
        """Verify the extracted discrete program against the spec.

        After synthesis produces a continuous (soft) program, the most likely
        discrete program is extracted via argmax over logits. This method
        runs that discrete program through execute_fixed (hard PC, hard
        register indexing) and compares against the specification.

        The "discretization gap" measures how much accuracy is lost going
        from the soft execution (which blends instruction probabilities)
        to the hard discrete program. A small gap indicates the synthesis
        successfully converged to a crisp discrete solution; a large gap
        suggests the soft program is exploiting blending in ways that do
        not survive discretization.

        Args:
            result: SynthesisResult from a completed synthesis run.
            spec: The specification the program was synthesized against.
            threshold: Maximum absolute error per register to count as
                correct (same semantics as _evaluate_accuracy).
            max_exec_steps: Maximum execution steps for execute_fixed.

        Returns:
            Dictionary with:
                soft_accuracy: float -- accuracy of the soft program
                discrete_accuracy: float -- accuracy of the discrete program
                discretization_gap: float -- soft_accuracy - discrete_accuracy
                per_example: list of tuples
                    (inputs, targets, predicted, correct: bool)
        """
        discrete_instructions = result.discrete_program
        fixed_program = FixedProgram(discrete_instructions)

        per_example = []
        correct = 0

        with torch.no_grad():
            for inputs, targets in spec.examples:
                exec_result = self.engine.execute_fixed(
                    fixed_program, inputs, max_steps=max_exec_steps
                )

                predicted = {}
                all_close = True
                for reg_idx, target_val in targets.items():
                    pred_val = exec_result.registers[reg_idx].item()
                    predicted[reg_idx] = pred_val
                    if abs(pred_val - target_val) > threshold:
                        all_close = False

                if all_close:
                    correct += 1

                per_example.append((inputs, targets, predicted, all_close))

        discrete_accuracy = correct / len(spec.examples)
        soft_accuracy = result.accuracy

        return {
            "soft_accuracy": soft_accuracy,
            "discrete_accuracy": discrete_accuracy,
            "discretization_gap": soft_accuracy - discrete_accuracy,
            "per_example": per_example,
        }


# ---------------------------------------------------------------------------
# Specification factories
# ---------------------------------------------------------------------------

def make_addition_spec(n_examples: int = 20) -> SynthesisSpec:
    """Create a spec: discover a program that adds R0 + R1 -> R2.

    This is the simplest synthesis task. The expected solution is a single
    ADD instruction followed by HALT.
    """
    examples = []
    for _ in range(n_examples):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        examples.append(({0: float(a), 1: float(b)}, {2: float(a + b)}))
    return SynthesisSpec(examples)


def make_multiply_spec(n_examples: int = 20) -> SynthesisSpec:
    """Create a spec: discover a program that multiplies R0 * R1 -> R2.

    Similar difficulty to addition -- a single MUL instruction suffices.
    Tests that the synthesizer can distinguish ADD from MUL by generalizing
    across multiple examples.
    """
    examples = []
    for _ in range(n_examples):
        a = random.randint(0, 15)
        b = random.randint(0, 15)
        examples.append(({0: float(a), 1: float(b)}, {2: float(a * b)}))
    return SynthesisSpec(examples)


def make_max_spec(n_examples: int = 20) -> SynthesisSpec:
    """Create a spec: discover a program that computes max(R0, R1) -> R2.

    This is significantly harder than addition or multiplication because it
    requires comparison and conditional behavior. The expected solution
    involves CMP, a conditional branch, and two possible MOV paths.

    Note: the differentiable execution engine handles branching via soft
    blending, so the synthesizer may discover an approximate solution that
    works well at low temperature but uses soft attention rather than a
    crisp branch.
    """
    examples = []
    for _ in range(n_examples):
        a = random.randint(0, 50)
        b = random.randint(0, 50)
        examples.append(({0: float(a), 1: float(b)}, {2: float(max(a, b))}))
    return SynthesisSpec(examples)


def make_polynomial_spec(n_examples: int = 20) -> SynthesisSpec:
    """Create a spec: discover a program that computes R0^2 + R0 -> R2.

    This requires multiple instructions: MUL R2, R0, R0 then ADD R2, R2, R0
    (or equivalent). Tests multi-instruction synthesis where intermediate
    values must be computed and combined.
    """
    examples = []
    for _ in range(n_examples):
        x = random.randint(0, 20)
        examples.append(({0: float(x)}, {2: float(x * x + x)}))
    return SynthesisSpec(examples)


# ---------------------------------------------------------------------------
# Demo functions
# ---------------------------------------------------------------------------

def demo_synthesize_addition() -> SynthesisResult:
    """Demo: gradient descent discovers ADD R2, R0, R1; HALT.

    This demonstrates the core capability: starting from random continuous
    parameters, the optimizer converges on the correct discrete program
    for addition purely from input-output examples.
    """
    print("=" * 64)
    print("PROGRAM SYNTHESIS: Addition (R0 + R1 -> R2)")
    print("=" * 64)
    print()

    random.seed(42)
    torch.manual_seed(42)

    spec = make_addition_spec(n_examples=30)
    print(f"Specification: {spec.num_examples} examples")
    print(f"  Sample: R0={spec.examples[0][0][0]:.0f}, "
          f"R1={spec.examples[0][0][1]:.0f} "
          f"-> R2={spec.examples[0][1][2]:.0f}")
    print()

    synthesizer = ProgramSynthesizer(
        max_program_len=8,
        num_registers=8,
        lr=0.02,
    )

    result = synthesizer.synthesize(
        spec,
        max_iters=3000,
        initial_temperature=2.0,
        final_temperature=0.1,
        tolerance=1e-3,
        length_penalty=0.001,
        verbose=True,
        print_every=500,
    )

    print()
    print("--- Discovered Program ---")
    print(result.program_text)
    print()
    print(f"Converged: {result.converged}")
    print(f"Steps: {result.steps}")
    print(f"Final loss: {result.loss_history[-1]:.6f}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print()

    # Verify on a few new examples
    print("--- Verification on new examples ---")
    engine = synthesizer.engine
    for a, b in [(7, 3), (15, 22), (0, 0), (50, 50)]:
        with torch.no_grad():
            res = engine.execute_soft(
                result.program,
                {0: float(a), 1: float(b)},
                temperature=0.1,
            )
            predicted = res.registers[2].item()
            expected = a + b
            status = "OK" if abs(predicted - expected) < 1.0 else "MISS"
            print(f"  {a} + {b} = {predicted:.1f} (expected {expected}) [{status}]")

    return result


def demo_synthesize_multiply() -> SynthesisResult:
    """Demo: gradient descent discovers MUL R2, R0, R1; HALT.

    Multiplication requires the optimizer to distinguish MUL from ADD
    across the training examples. With enough examples, the loss landscape
    strongly favors MUL.
    """
    print()
    print("=" * 64)
    print("PROGRAM SYNTHESIS: Multiplication (R0 * R1 -> R2)")
    print("=" * 64)
    print()

    random.seed(123)
    torch.manual_seed(123)

    spec = make_multiply_spec(n_examples=30)
    print(f"Specification: {spec.num_examples} examples")
    print(f"  Sample: R0={spec.examples[0][0][0]:.0f}, "
          f"R1={spec.examples[0][0][1]:.0f} "
          f"-> R2={spec.examples[0][1][2]:.0f}")
    print()

    synthesizer = ProgramSynthesizer(
        max_program_len=8,
        num_registers=8,
        lr=0.02,
    )

    result = synthesizer.synthesize(
        spec,
        max_iters=3000,
        initial_temperature=2.0,
        final_temperature=0.1,
        tolerance=1e-3,
        length_penalty=0.001,
        verbose=True,
        print_every=500,
    )

    print()
    print("--- Discovered Program ---")
    print(result.program_text)
    print()
    print(f"Converged: {result.converged}")
    print(f"Steps: {result.steps}")
    print(f"Final loss: {result.loss_history[-1]:.6f}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print()

    # Verify on new examples
    print("--- Verification on new examples ---")
    engine = synthesizer.engine
    for a, b in [(3, 7), (5, 5), (0, 10), (12, 8)]:
        with torch.no_grad():
            res = engine.execute_soft(
                result.program,
                {0: float(a), 1: float(b)},
                temperature=0.1,
            )
            predicted = res.registers[2].item()
            expected = a * b
            status = "OK" if abs(predicted - expected) < 1.0 else "MISS"
            print(f"  {a} * {b} = {predicted:.1f} (expected {expected}) [{status}]")

    return result


def demo_synthesize_polynomial() -> SynthesisResult:
    """Demo: discovers a multi-instruction program for x^2 + x.

    This is harder than single-instruction synthesis because the optimizer
    must discover that it needs TWO instructions (MUL then ADD) and must
    correctly wire intermediate registers. The program must:
      1. Compute R0 * R0 (square) into some register
      2. Add R0 to the square
      3. Store the result in R2
    """
    print()
    print("=" * 64)
    print("PROGRAM SYNTHESIS: Polynomial (R0^2 + R0 -> R2)")
    print("=" * 64)
    print()

    random.seed(456)
    torch.manual_seed(456)

    spec = make_polynomial_spec(n_examples=30)
    print(f"Specification: {spec.num_examples} examples")
    print(f"  Sample: R0={spec.examples[0][0][0]:.0f} "
          f"-> R2={spec.examples[0][1][2]:.0f}")
    print()

    # Polynomial needs more capacity and patience
    synthesizer = ProgramSynthesizer(
        max_program_len=10,
        num_registers=8,
        lr=0.01,
    )

    result = synthesizer.synthesize(
        spec,
        max_iters=5000,
        initial_temperature=2.5,
        final_temperature=0.1,
        tolerance=0.01,
        length_penalty=0.0005,
        verbose=True,
        print_every=500,
    )

    print()
    print("--- Discovered Program ---")
    print(result.program_text)
    print()
    print(f"Converged: {result.converged}")
    print(f"Steps: {result.steps}")
    print(f"Final loss: {result.loss_history[-1]:.6f}")
    print(f"Accuracy: {result.accuracy:.1%}")
    print()

    # Verify on new examples
    print("--- Verification on new examples ---")
    engine = synthesizer.engine
    for x in [0, 1, 5, 10, 20]:
        with torch.no_grad():
            res = engine.execute_soft(
                result.program,
                {0: float(x)},
                temperature=0.1,
            )
            predicted = res.registers[2].item()
            expected = x * x + x
            status = "OK" if abs(predicted - expected) < 1.0 else "MISS"
            print(
                f"  f({x}) = {predicted:.1f} "
                f"(expected {expected}) [{status}]"
            )

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("Differentiable Program Synthesis")
    print("Programs discovered from I/O examples via gradient descent")
    print("=" * 64)
    print()

    r1 = demo_synthesize_addition()
    r2 = demo_synthesize_multiply()
    r3 = demo_synthesize_polynomial()

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Addition:    {'CONVERGED' if r1.converged else 'DID NOT CONVERGE'} "
          f"in {r1.steps} steps, accuracy {r1.accuracy:.1%}")
    print(f"  Multiply:    {'CONVERGED' if r2.converged else 'DID NOT CONVERGE'} "
          f"in {r2.steps} steps, accuracy {r2.accuracy:.1%}")
    print(f"  Polynomial:  {'CONVERGED' if r3.converged else 'DID NOT CONVERGE'} "
          f"in {r3.steps} steps, accuracy {r3.accuracy:.1%}")
    print()
    print("Each program was discovered purely from input-output examples")
    print("using gradient descent through a differentiable CPU.")
    print()
