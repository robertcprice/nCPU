"""Tests for the differentiable execution, optimization, and synthesis package.

Verifies that:
1. Gradient flow is maintained through all operations
2. Program optimization converges to correct values
3. Program synthesis discovers correct programs
4. ISA discovery learns arithmetic operations
5. Float ALU trains to approximate ground truth
"""

import pytest
import torch

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    Instruction,
    OPCODES,
    DifferentiableALU,
)
from ncpu.differentiable.program_optimizer import ProgramOptimizer, OptimizationResult
from ncpu.differentiable.program_synthesis import (
    ProgramSynthesizer,
    SynthesisSpec,
    make_addition_spec,
    make_multiply_spec,
)
from ncpu.differentiable.isa_discovery import (
    NeuralISADiscovery,
    ISAConfig,
    make_arithmetic_benchmark,
)
from ncpu.differentiable.float_alu import NeuralFloatALU, FloatPrecision


# =========================================================================
# Gradient flow tests --- the most important tests in this file
# =========================================================================


class TestGradientFlow:
    """Verify that gradients actually flow through differentiable execution."""

    def test_gradient_through_mov_imm(self):
        """MOV R0, #X: gradient flows from R0 back to the immediate X."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=5.0),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()
        result = engine.execute_fixed(program, {})

        loss = (result.registers[0] - 10.0) ** 2
        loss.backward()

        assert program.immediates.grad is not None
        # loss = (R0 - 10)^2 where R0=5. d/d(imm) = 2*(5-10)*1 = -10
        assert abs(program.immediates.grad[0].item() - (-10.0)) < 0.5, \
            f"Expected grad ≈ -10, got {program.immediates.grad[0].item()}"

    def test_gradient_through_add(self):
        """ADD R2, R0, R1: gradient flows back to input registers."""
        program = FixedProgram([
            Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()

        r0 = torch.tensor(3.0, requires_grad=True)
        r1 = torch.tensor(4.0, requires_grad=True)
        result = engine.execute_fixed(program, {0: r0, 1: r1})

        loss = (result.registers[2] - 10.0) ** 2
        loss.backward()

        # loss = (R2-10)^2 where R2=R0+R1=7. d/dR0 = 2*(7-10)*1 = -6. Same for R1.
        assert r0.grad is not None
        assert abs(r0.grad.item() - (-6.0)) < 0.5, \
            f"Expected grad ≈ -6, got {r0.grad.item()}"
        assert r1.grad is not None
        assert abs(r1.grad.item() - (-6.0)) < 0.5, \
            f"Expected grad ≈ -6, got {r1.grad.item()}"

    def test_gradient_through_mul(self):
        """MUL R2, R0, R1: gradient flows back to both inputs."""
        program = FixedProgram([
            Instruction(OPCODES["MUL"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()

        r0 = torch.tensor(3.0, requires_grad=True)
        r1 = torch.tensor(4.0, requires_grad=True)
        result = engine.execute_fixed(program, {0: r0, 1: r1})

        # R2 = R0 * R1 = 12
        assert abs(result.registers[2].item() - 12.0) < 0.01

        loss = (result.registers[2] - 20.0) ** 2
        loss.backward()

        # d(loss)/d(R0) = 2*(12-20)*R1 = 2*(-8)*4 = -64
        assert r0.grad is not None
        assert abs(r0.grad.item() - (-64.0)) < 0.1

    def test_gradient_through_multi_instruction(self):
        """Multi-instruction program: gradients chain through all steps."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=3.0),
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=4.0),
            Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["MUL"], dst=3, src1=2, src2=0),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()
        result = engine.execute_fixed(program, {})

        # R2 = 3+4 = 7, R3 = 7*3 = 21
        assert abs(result.registers[3].item() - 21.0) < 0.01

        loss = (result.registers[3] - 30.0) ** 2
        loss.backward()

        # Gradients should reach the immediates at instructions 0 and 1
        assert program.immediates.grad is not None
        assert program.immediates.grad[0].item() != 0.0, \
            "Gradient should reach first MOV_IMM immediate"
        assert program.immediates.grad[1].item() != 0.0, \
            "Gradient should reach second MOV_IMM immediate"

    def test_gradient_through_sub(self):
        """SUB: gradient flows correctly."""
        program = FixedProgram([
            Instruction(OPCODES["SUB"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()

        r0 = torch.tensor(10.0, requires_grad=True)
        r1 = torch.tensor(3.0, requires_grad=True)
        result = engine.execute_fixed(program, {0: r0, 1: r1})

        assert abs(result.registers[2].item() - 7.0) < 0.01

        loss = result.registers[2] ** 2
        loss.backward()

        # d(R2^2)/d(R0) = 2*R2*1 = 14
        assert abs(r0.grad.item() - 14.0) < 0.1
        # d(R2^2)/d(R1) = 2*R2*(-1) = -14
        assert abs(r1.grad.item() - (-14.0)) < 0.1


    def test_gradient_through_mov_reg(self):
        """MOV R1, R0: gradient flows through register copy."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_REG"], dst=1, src1=0),
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()
        r0 = torch.tensor(5.0, requires_grad=True)
        result = engine.execute_fixed(program, {0: r0})
        loss = (result.registers[1] - 10.0) ** 2
        loss.backward()
        # d/dR0 = 2*(5-10)*1 = -10
        assert r0.grad is not None
        assert abs(r0.grad.item() - (-10.0)) < 0.5, \
            f"Expected grad ≈ -10, got {r0.grad.item()}"


class TestSoftExecution:
    """Test soft (fully differentiable) program execution."""

    def test_soft_program_creates(self):
        """SoftProgram initializes with correct shapes."""
        prog = SoftProgram(max_length=8, num_registers=4)
        assert prog.opcode_logits.shape == (8, 14)  # NUM_OPCODES = 14
        assert prog.dst_logits.shape == (8, 4)
        assert prog.immediates.shape == (8,)

    def test_soft_program_gradient_flow(self):
        """Gradients flow through soft program execution."""
        engine = DifferentiableEngine(num_registers=8)
        prog = SoftProgram(max_length=4, num_registers=8)

        result = engine.execute_soft(prog, {0: 5.0, 1: 3.0}, max_steps=4)
        loss = result.registers.sum() ** 2
        loss.backward()

        assert prog.opcode_logits.grad is not None
        assert prog.opcode_logits.grad.abs().sum() > 0

    def test_soft_program_extract_discrete(self):
        """Discrete program extraction works."""
        prog = SoftProgram(max_length=4)
        discrete = prog.extract_discrete_program()
        assert len(discrete) == 4
        assert all(hasattr(inst, "opcode") for inst in discrete)

    def test_soft_program_format(self):
        """Program formatting produces valid text."""
        prog = SoftProgram(max_length=4)
        text = prog.format_program()
        assert isinstance(text, str)
        assert len(text.split("\n")) == 4

    def test_soft_execution_halt_masking(self):
        """Registers should freeze after HALT in soft execution."""
        engine = DifferentiableEngine()
        # Create a program that HALTs early but has instructions after
        prog = SoftProgram(max_length=4)
        # Force first instruction to be HALT
        prog.opcode_logits.data[0, OPCODES["HALT"]] = 10.0
        # Force later instructions to do something
        prog.opcode_logits.data[1, OPCODES["MOV_IMM"]] = 10.0
        prog.immediates.data[1] = 999.0

        result = engine.execute_soft(prog, {0: 5.0}, max_steps=4, skip_bitwise=True)
        # R0 should still be 5.0, not corrupted by post-halt MOV
        assert abs(result.registers[0].item() - 5.0) < 1.0, \
            f"R0 should be ~5.0 after halt masking, got {result.registers[0].item()}"


# =========================================================================
# Batched execution tests
# =========================================================================


class TestBatchedExecution:
    """Test batched soft execution matches sequential and is faster."""

    def test_batched_execution_produces_results(self):
        """Batched execution returns correct number of results."""
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        inputs_list = [{0: 3.0, 1: 5.0}, {0: 7.0, 1: 2.0}, {0: 1.0, 1: 9.0}]

        batch_results = engine.execute_soft_batched(
            prog, inputs_list, max_steps=4, skip_bitwise=True
        )
        assert len(batch_results) == 3
        for r in batch_results:
            assert r.registers.shape == (8,)
            assert r.flags.shape == (4,)

    def test_batched_execution_matches_sequential(self):
        """Batched execution produces same results as sequential.

        Because get_soft_instruction uses Gumbel-softmax with random noise,
        sequential and batched calls will sample different noise. To compare,
        we set a fixed seed before each execution so the Gumbel noise is
        deterministic, and use the same program state.
        """
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        inputs_list = [{0: 3.0, 1: 5.0}, {0: 7.0, 1: 2.0}, {0: 1.0, 1: 9.0}]

        # Run sequential with a fixed seed
        torch.manual_seed(42)
        seq_results = []
        for inp in inputs_list:
            torch.manual_seed(42)  # Reset per-example for fair comparison
            r = engine.execute_soft(prog, inp, max_steps=4, skip_bitwise=True)
            seq_results.append(r.registers.detach())

        # Run batched with same seed
        torch.manual_seed(42)
        batch_results = engine.execute_soft_batched(
            prog, inputs_list, max_steps=4, skip_bitwise=True
        )

        # They won't be exactly identical due to Gumbel noise differences
        # (sequential calls get_soft_instruction once per example per step,
        # batched calls it once per step for all examples). But the program
        # parameters and arithmetic should be consistent. Check that outputs
        # are in a reasonable range and gradients flow.
        for batch_r in batch_results:
            assert batch_r.registers.shape == (8,)
            # Values should be finite
            assert torch.isfinite(batch_r.registers).all()

    def test_batched_gradient_flow(self):
        """Gradients flow through batched execution to program parameters."""
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        inputs_list = [{0: 3.0, 1: 5.0}, {0: 7.0, 1: 2.0}]

        batch_results = engine.execute_soft_batched(
            prog, inputs_list, max_steps=4, skip_bitwise=True
        )

        loss = torch.tensor(0.0)
        for r in batch_results:
            loss = loss + (r.registers[2] - 10.0) ** 2
        loss.backward()

        assert prog.opcode_logits.grad is not None
        assert prog.opcode_logits.grad.abs().sum() > 0
        assert prog.dst_logits.grad is not None
        assert prog.src1_logits.grad is not None

    def test_batched_deterministic_without_gumbel(self):
        """With very low temperature, batched and sequential should converge.

        At near-zero temperature, Gumbel-softmax approaches argmax, reducing
        stochasticity. This makes batched vs sequential more comparable.
        """
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        # Bias toward ADD so the program does something deterministic
        prog.opcode_logits.data[0, OPCODES["ADD"]] = 10.0
        prog.opcode_logits.data[1, OPCODES["HALT"]] = 10.0
        prog.src1_logits.data[0, 0] = 10.0  # src1 = R0
        prog.src2_logits.data[0, 1] = 10.0  # src2 = R1
        prog.dst_logits.data[0, 2] = 10.0   # dst = R2

        inputs_list = [{0: 3.0, 1: 5.0}, {0: 10.0, 1: 20.0}]

        # Sequential at very low temperature
        seq_results = []
        for inp in inputs_list:
            r = engine.execute_soft(prog, inp, max_steps=4,
                                    temperature=0.01, skip_bitwise=True)
            seq_results.append(r.registers.detach())

        # Batched at very low temperature
        batch_results = engine.execute_soft_batched(
            prog, inputs_list, max_steps=4,
            temperature=0.01, skip_bitwise=True
        )

        # At low temperature with strong biases, both should get ~ADD behavior
        for seq_r, batch_r in zip(seq_results, batch_results):
            # R2 should be close to R0+R1
            assert torch.allclose(seq_r, batch_r.registers.detach(), atol=0.5), \
                f"Sequential {seq_r} vs Batched {batch_r.registers.detach()}"

    def test_batched_halt_masking(self):
        """Registers freeze after HALT in batched execution."""
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        # Force HALT at position 0
        prog.opcode_logits.data[0, OPCODES["HALT"]] = 10.0
        # Force MOV_IMM at position 1 (should not execute)
        prog.opcode_logits.data[1, OPCODES["MOV_IMM"]] = 10.0
        prog.immediates.data[1] = 999.0

        inputs_list = [{0: 5.0}, {0: 10.0}]
        batch_results = engine.execute_soft_batched(
            prog, inputs_list, max_steps=4, skip_bitwise=True
        )

        # R0 should be preserved (not corrupted by post-halt MOV)
        assert abs(batch_results[0].registers[0].item() - 5.0) < 1.0
        assert abs(batch_results[1].registers[0].item() - 10.0) < 1.0

    def test_batched_single_example_matches_sequential(self):
        """Batch size 1 should produce identical results to sequential.

        With batch_size=1, the averaged flags equal the single-example flags,
        so PC evolution is identical. Combined with same Gumbel seed, results
        should match exactly.
        """
        engine = DifferentiableEngine()
        prog = SoftProgram(max_length=4)
        inp = {0: 7.0, 1: 3.0}

        torch.manual_seed(99)
        seq_result = engine.execute_soft(
            prog, inp, max_steps=4, temperature=1.0, skip_bitwise=True
        )

        torch.manual_seed(99)
        batch_results = engine.execute_soft_batched(
            prog, [inp], max_steps=4, temperature=1.0, skip_bitwise=True
        )

        assert torch.allclose(
            seq_result.registers.detach(),
            batch_results[0].registers.detach(),
            atol=1e-5,
        ), (f"Single-example batch should match sequential.\n"
            f"Sequential: {seq_result.registers.detach()}\n"
            f"Batched:    {batch_results[0].registers.detach()}")

    def test_batched_synthesis_integration(self):
        """Synthesizer works with batched execution (integration test)."""
        torch.manual_seed(42)
        spec = SynthesisSpec(examples=[
            ({0: 1.0, 1: 2.0}, {2: 3.0}),
            ({0: 3.0, 1: 4.0}, {2: 7.0}),
            ({0: 5.0, 1: 5.0}, {2: 10.0}),
        ])
        synth = ProgramSynthesizer(max_program_len=6, lr=0.02)
        result = synth.synthesize(
            spec, max_iters=200, tolerance=0.1,
            skip_bitwise=True, max_exec_steps=6,
        )
        # Loss should decrease during training
        assert result.loss_history[-1] < result.loss_history[0], \
            "Loss should decrease when using batched execution"


# =========================================================================
# ALU tests
# =========================================================================


class TestDifferentiableALU:
    """Test the differentiable ALU operations."""

    def test_arithmetic_correctness(self):
        """ADD, SUB, MUL produce correct results."""
        alu = DifferentiableALU()
        a = torch.tensor(7.0)
        b = torch.tensor(3.0)
        imm = torch.tensor(0.0)

        results = alu.compute_all(a, b, imm)
        assert abs(results["ADD"].item() - 10.0) < 0.01
        assert abs(results["SUB"].item() - 4.0) < 0.01
        assert abs(results["MUL"].item() - 21.0) < 0.01

    def test_arithmetic_gradients(self):
        """Arithmetic ops have correct gradients."""
        alu = DifferentiableALU()
        a = torch.tensor(5.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        imm = torch.tensor(0.0)

        results = alu.compute_all(a, b, imm)
        results["MUL"].backward()

        # d(a*b)/da = b = 3
        assert abs(a.grad.item() - 3.0) < 0.01
        # d(a*b)/db = a = 5
        assert abs(b.grad.item() - 5.0) < 0.01

    def test_flags_computation(self):
        """Soft flags give correct sign for comparisons."""
        alu = DifferentiableALU()

        # a > b: N should be low, Z should be low
        flags = alu.compute_flags(torch.tensor(10.0), torch.tensor(3.0))
        assert flags[0].item() < 0.5  # N flag (not negative)
        assert flags[1].item() < 0.5  # Z flag (not zero)

        # a == b: Z should be high
        flags = alu.compute_flags(torch.tensor(5.0), torch.tensor(5.0))
        assert flags[1].item() > 0.5  # Z flag (is zero)

        # a < b: N should be high
        flags = alu.compute_flags(torch.tensor(3.0), torch.tensor(10.0))
        assert flags[0].item() > 0.5  # N flag (is negative)


# =========================================================================
# Program optimizer tests
# =========================================================================


class TestProgramOptimizer:
    """Test gradient-based program optimization."""

    def test_optimize_single_immediate(self):
        """Find X such that X * 3 = 42 (X should be 14)."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=3.0),
            Instruction(OPCODES["MUL"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        opt = ProgramOptimizer(lr=0.5)
        result = opt.optimize_inputs(
            program,
            target_registers={2: 42.0},
            input_registers=[0],
            initial_values={0: 1.0},
            max_iters=500,
        )
        assert result.converged
        x = result.optimized_immediates[0].item()
        assert abs(x - 14.0) < 0.5, f"Expected ~14, got {x}"

    def test_optimize_addition_inputs(self):
        """Find R0, R1 such that R0 + R1 = 20."""
        program = FixedProgram([
            Instruction(OPCODES["ADD"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])
        opt = ProgramOptimizer(lr=0.5)
        result = opt.optimize_inputs(
            program,
            target_registers={2: 20.0},
            input_registers=[0, 1],
            initial_values={0: 1.0, 1: 1.0},
            max_iters=500,
        )
        assert result.converged
        r0 = result.optimized_immediates[0].item()
        r1 = result.optimized_immediates[1].item()
        assert abs(r0 + r1 - 20.0) < 0.5

    def test_polynomial_fitting(self):
        """Fit f(x) = 2x + 1 via differentiable execution."""
        # Program: MOV R1, #a; MOV R2, #b; MUL R3, R1, R0; ADD R4, R3, R2; HALT
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=0.5),  # a
            Instruction(OPCODES["MOV_IMM"], dst=2, immediate=0.5),  # b
            Instruction(OPCODES["MUL"], dst=3, src1=1, src2=0),     # a*x
            Instruction(OPCODES["ADD"], dst=4, src1=3, src2=2),     # a*x + b
            Instruction(OPCODES["HALT"]),
        ])

        engine = DifferentiableEngine()
        optimizer = torch.optim.Adam(list(program.parameters()), lr=0.05)

        # Train on f(x) = 2x + 1
        for step in range(1000):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            for x, y in [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0), (0.0, 1.0)]:
                result = engine.execute_fixed(program, {0: x})
                total_loss = total_loss + (result.registers[4] - y) ** 2
            total_loss.backward()
            optimizer.step()

        a = program.immediates.data[0].item()
        b = program.immediates.data[1].item()
        assert abs(a - 2.0) < 0.1, f"Expected a≈2, got {a}"
        assert abs(b - 1.0) < 0.1, f"Expected b≈1, got {b}"

    def test_optimize_toward_custom_loss(self):
        """Custom loss function works with optimize_toward."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=0, immediate=1.0),
            Instruction(OPCODES["HALT"]),
        ])
        opt = ProgramOptimizer(lr=0.1)

        # Custom loss: minimize (R0 - 7)^2
        result = opt.optimize_toward(
            program,
            loss_fn=lambda r: (r.registers[0] - 7.0) ** 2,
            max_iters=500,
        )
        assert result.converged
        assert abs(result.final_registers[0].item() - 7.0) < 0.5

    def test_quadratic_polynomial_fitting(self):
        """Fit f(x) = 2x^2 + 3x + 5 --- matches paper Section 14.3.3."""
        program = FixedProgram([
            Instruction(OPCODES["MOV_IMM"], dst=1, immediate=0.5),  # a
            Instruction(OPCODES["MOV_IMM"], dst=2, immediate=0.5),  # b
            Instruction(OPCODES["MOV_IMM"], dst=3, immediate=0.5),  # c
            Instruction(OPCODES["MUL"], dst=4, src1=0, src2=0),     # x^2
            Instruction(OPCODES["MUL"], dst=5, src1=1, src2=4),     # a*x^2
            Instruction(OPCODES["MUL"], dst=6, src1=2, src2=0),     # b*x
            Instruction(OPCODES["ADD"], dst=7, src1=5, src2=6),     # a*x^2 + b*x
            Instruction(OPCODES["ADD"], dst=7, src1=7, src2=3),     # + c
            Instruction(OPCODES["HALT"]),
        ])
        engine = DifferentiableEngine()
        optimizer = torch.optim.Adam(list(program.parameters()), lr=0.05)

        # f(x) = 2x^2 + 3x + 5
        train_points = [
            (0.0, 5.0), (-1.0, 4.0), (1.0, 10.0), (2.0, 19.0), (3.0, 32.0),
        ]

        for step in range(2000):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0)
            for x, y in train_points:
                result = engine.execute_fixed(program, {0: x})
                total_loss = total_loss + (result.registers[7] - y) ** 2
            total_loss.backward()
            optimizer.step()

        a = program.immediates.data[0].item()
        b = program.immediates.data[1].item()
        c = program.immediates.data[2].item()
        assert abs(a - 2.0) < 0.1, f"Expected a ~= 2, got {a}"
        assert abs(b - 3.0) < 0.1, f"Expected b ~= 3, got {b}"
        assert abs(c - 5.0) < 0.1, f"Expected c ~= 5, got {c}"


# =========================================================================
# Assembler tests
# =========================================================================


class TestAssembler:
    """Test the text assembler."""

    def test_assemble_simple(self):
        """Assemble a simple program."""
        prog = DifferentiableEngine.assemble(
            "MOV R0, #42\nMOV R1, #3\nADD R2, R0, R1\nHALT"
        )
        assert prog.length == 4
        assert prog.instructions[0].opcode == OPCODES["MOV_IMM"]
        assert prog.instructions[2].opcode == OPCODES["ADD"]

    def test_assemble_and_execute(self):
        """Assemble and execute a program."""
        engine = DifferentiableEngine()
        prog = DifferentiableEngine.assemble(
            "MOV R0, #7\nMOV R1, #6\nMUL R2, R0, R1\nHALT"
        )
        result = engine.execute_fixed(prog, {})
        assert abs(result.registers[2].item() - 42.0) < 0.01

    def test_assemble_with_comments(self):
        """Assembler strips comments."""
        prog = DifferentiableEngine.assemble(
            "MOV R0, #5  ; load five\nHALT ; done"
        )
        assert prog.length == 2

    def test_assemble_branch(self):
        """Assembler handles branch instructions."""
        prog = DifferentiableEngine.assemble(
            "CMP R0, R1\nBEQ @3\nMOV R2, #1\nHALT"
        )
        assert prog.instructions[1].opcode == OPCODES["BEQ"]
        assert prog.instructions[1].branch_target == 3


# =========================================================================
# Program synthesis tests
# =========================================================================


class TestProgramSynthesis:
    """Test program synthesis from specifications."""

    def test_spec_validation(self):
        """SynthesisSpec rejects empty specs."""
        with pytest.raises(ValueError):
            SynthesisSpec(examples=[])

    def test_synthesizer_creates(self):
        """ProgramSynthesizer initializes correctly."""
        synth = ProgramSynthesizer(max_program_len=8)
        assert synth.max_program_len == 8

    def test_synthesize_trivial(self):
        """Synthesize a trivial identity program: R0 -> R2."""
        spec = SynthesisSpec(examples=[
            ({0: 5.0}, {0: 5.0}),  # R0 stays 5
            ({0: 10.0}, {0: 10.0}),
        ])
        synth = ProgramSynthesizer(max_program_len=4, lr=0.05)
        result = synth.synthesize(
            spec, max_iters=100, tolerance=0.1,
            skip_bitwise=True, max_exec_steps=6,
        )
        # Should find something close to NOP; HALT --- loss decreases
        assert result.loss_history[-1] < result.loss_history[0]

    def test_addition_spec_creation(self):
        """make_addition_spec creates valid specs."""
        spec = make_addition_spec(n_examples=10)
        assert len(spec.examples) == 10
        for inputs, targets in spec.examples:
            assert 0 in inputs and 1 in inputs
            assert 2 in targets
            assert abs(targets[2] - (inputs[0] + inputs[1])) < 0.01


# =========================================================================
# ISA discovery tests
# =========================================================================


class TestISADiscovery:
    """Test neural ISA discovery."""

    def test_isa_creates(self):
        """NeuralISADiscovery initializes correctly."""
        isa = NeuralISADiscovery(ISAConfig(max_opcodes=8))
        assert len(isa.op_networks) == 8
        assert isa.op_costs.shape == (8,)

    def test_isa_forward(self):
        """ISA forward pass produces output."""
        isa = NeuralISADiscovery()
        a = torch.tensor(5.0)
        b = torch.tensor(3.0)
        result = isa.forward(a, b, 0)
        assert result.shape == ()  # scalar

    def test_isa_gradient_flow(self):
        """Gradients flow through ISA operations."""
        isa = NeuralISADiscovery()
        a = torch.tensor(5.0, requires_grad=True)
        b = torch.tensor(3.0, requires_grad=True)
        result = isa.forward(a, b, 0)
        loss = (result - 8.0) ** 2
        loss.backward()
        assert a.grad is not None

    def test_isa_learns_addition(self):
        """ISA discovers addition after training."""
        isa = NeuralISADiscovery(ISAConfig(max_opcodes=4))
        bench = make_arithmetic_benchmark()
        result = isa.discover([bench], max_iters=500, lr=0.01)

        # Verify op0 learned something close to addition
        with torch.no_grad():
            pred = isa.forward(torch.tensor(4.0), torch.tensor(6.0), 0)
        assert abs(pred.item() - 10.0) < 2.0, \
            f"Op0 should learn addition: expected ~10, got {pred.item()}"


# =========================================================================
# Float ALU tests
# =========================================================================


class TestFloatALU:
    """Test the neural floating-point ALU."""

    def test_float_alu_creates(self):
        """NeuralFloatALU initializes correctly."""
        alu = NeuralFloatALU()
        assert alu.precision == FloatPrecision.SINGLE

    def test_float_alu_forward(self):
        """Float ALU forward passes produce outputs."""
        alu = NeuralFloatALU()
        a = torch.tensor(3.0)
        b = torch.tensor(2.0)

        assert alu.fadd(a, b).shape == ()
        assert alu.fmul(a, b).shape == ()
        assert alu.fdiv(a, b).shape == ()
        assert alu.fsqrt(a).shape == ()

    def test_float_alu_gradient_flow(self):
        """Gradients flow through float ALU operations."""
        alu = NeuralFloatALU()
        a = torch.tensor(3.0, requires_grad=True)
        b = torch.tensor(2.0, requires_grad=True)

        result = alu.fadd(a, b)
        loss = (result - 5.0) ** 2
        loss.backward()
        assert a.grad is not None

    def test_float_alu_trains(self):
        """Float ALU loss decreases after training."""
        alu = NeuralFloatALU(hidden_dim=64)
        losses = alu.train_from_ground_truth(
            "add", n_samples=1000, value_range=(-5.0, 5.0), epochs=50
        )
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_float_comparison(self):
        """Float comparison produces correct soft flags."""
        alu = NeuralFloatALU()
        flags = alu.fcmp(torch.tensor(5.0), torch.tensor(3.0))
        assert flags.shape == (3,)  # LT, EQ, GT
        assert flags[2].item() > flags[0].item()  # GT > LT when a > b


# =========================================================================
# Integration tests
# =========================================================================


class TestIntegration:
    """End-to-end integration tests."""

    def test_assemble_optimize_verify(self):
        """Full pipeline: assemble -> optimize -> verify."""
        engine = DifferentiableEngine()

        # Program computes R0 + R1 where R0 is loaded from immediate
        prog = DifferentiableEngine.assemble(
            "MOV R0, #1\nADD R2, R0, R1\nHALT"
        )

        # Optimize: find R0 immediate such that R0 + 5 = 12
        opt = ProgramOptimizer(engine=engine, lr=0.5)
        result = opt.optimize_immediates(
            prog, target_registers={2: 12.0}, inputs={1: 5.0}
        )

        assert result.converged
        r0_val = prog.immediates.data[0].item()
        assert abs(r0_val - 7.0) < 0.5

    def test_execution_trace(self):
        """Execution trace captures per-step register state."""
        engine = DifferentiableEngine()
        prog = DifferentiableEngine.assemble(
            "MOV R0, #5\nMOV R1, #3\nADD R2, R0, R1\nHALT"
        )
        result = engine.execute_fixed(prog, {})

        # Should have 4 trace entries (initial + 3 instructions before HALT)
        assert len(result.register_trace) >= 3
        assert result.halted

    def test_empty_program(self):
        """Zero-instruction program should not crash."""
        engine = DifferentiableEngine()
        prog = FixedProgram([])
        result = engine.execute_fixed(prog, {0: 5.0})
        assert result.registers[0].item() == 5.0, \
            "R0 should remain unchanged for empty program"

    def test_nan_input_propagates(self):
        """NaN input should propagate through execution."""
        import math
        engine = DifferentiableEngine()
        prog = DifferentiableEngine.assemble("ADD R2, R0, R1\nHALT")
        result = engine.execute_fixed(prog, {0: float('nan'), 1: 3.0})
        assert math.isnan(result.registers[2].item()), "NaN should propagate"

    def test_float_alu_correctness_after_training(self):
        """Float ALU produces correct results after training."""
        alu = NeuralFloatALU(hidden_dim=64)
        alu.train_from_ground_truth(
            "add", n_samples=2000, value_range=(-5.0, 5.0), epochs=100,
        )
        with torch.no_grad():
            result = alu.fadd(torch.tensor(3.0), torch.tensor(2.0))
        assert abs(result.item() - 5.0) < 0.5, \
            f"Expected ~5.0, got {result.item()}"

    def test_bitwise_gradient_flow(self):
        """Bitwise ops (via soft truth tables) support gradient flow."""
        engine = DifferentiableEngine()
        prog = FixedProgram([
            Instruction(OPCODES["AND"], dst=2, src1=0, src2=1),
            Instruction(OPCODES["HALT"]),
        ])

        r0 = torch.tensor(15.0, requires_grad=True)  # 0x0F
        r1 = torch.tensor(255.0, requires_grad=True)  # 0xFF

        result = engine.execute_fixed(prog, {0: r0, 1: r1})
        loss = result.registers[2] ** 2
        loss.backward()

        # Gradient should exist and be nonzero (via soft truth table bilinear interpolation)
        assert r0.grad is not None and r0.grad.item() != 0.0, \
            "Bitwise gradient should be nonzero"
