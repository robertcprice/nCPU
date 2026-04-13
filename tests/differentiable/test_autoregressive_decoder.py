"""Tests for the autoregressive compiler decoder.

Verifies:
  1. AutoregressiveDecoder creates successfully with correct shapes.
  2. Compiling with the autoregressive decoder produces a valid SoftProgram.
  3. Gradients flow end-to-end through the autoregressive path.
  4. Training with the autoregressive decoder reduces loss over time.
"""

import pytest
import torch
import torch.nn as nn

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    SoftProgram,
    NUM_OPCODES,
)
from ncpu.differentiable.diff_compiler import (
    AutoregressiveDecoder,
    DifferentiableCompiler,
    DifferentiableCompilationPipeline,
    SimpleTokenizer,
)


# =========================================================================
# AutoregressiveDecoder unit tests
# =========================================================================


class TestAutoregressiveDecoder:
    """Tests for the AutoregressiveDecoder module."""

    def test_autoregressive_creates(self):
        """Decoder instantiates with correct architecture and attributes."""
        decoder = AutoregressiveDecoder(
            d_model=64,
            max_program_len=16,
            num_registers=8,
            num_opcodes=NUM_OPCODES,
            nhead=4,
        )

        assert decoder.d_model == 64
        assert decoder.max_program_len == 16
        assert decoder.num_registers == 8
        assert decoder.num_opcodes == NUM_OPCODES

        # Verify sub-modules exist
        assert decoder.decoder is not None
        assert decoder.opcode_head is not None
        assert decoder.dst_head is not None
        assert decoder.src1_head is not None
        assert decoder.src2_head is not None
        assert decoder.imm_head is not None
        assert decoder.branch_head is not None

        # Position embeddings have correct shape
        assert decoder.pos_embed.shape == (16, 64)

        # Instruction embedding maps from opcode+regs+imm to d_model
        expected_in = NUM_OPCODES + 8 * 3 + 1  # opcodes + 3 reg fields + imm
        assert decoder.inst_embed.in_features == expected_in
        assert decoder.inst_embed.out_features == 64

    def test_autoregressive_compile_returns_soft_program(self):
        """The AR decoder produces a valid SoftProgram from encoder output."""
        d_model = 32
        max_len = 8
        num_regs = 4
        decoder = AutoregressiveDecoder(
            d_model=d_model,
            max_program_len=max_len,
            num_registers=num_regs,
            num_opcodes=NUM_OPCODES,
            nhead=4,
        )

        # Simulate encoder output: [1, seq_len=5, d_model]
        encoder_output = torch.randn(1, 5, d_model)

        program = decoder(encoder_output)

        assert isinstance(program, SoftProgram)

        # Check shapes of generated logits
        assert program.opcode_logits.shape == (max_len, NUM_OPCODES)
        assert program.dst_logits.shape == (max_len, num_regs)
        assert program.src1_logits.shape == (max_len, num_regs)
        assert program.src2_logits.shape == (max_len, num_regs)
        assert program.immediates.shape == (max_len,)
        assert program.branch_logits.shape == (max_len, max_len)

    def test_autoregressive_gradient_flow(self):
        """Gradients flow from execution loss back through the AR decoder."""
        d_model = 32
        max_len = 8
        num_regs = 8

        decoder = AutoregressiveDecoder(
            d_model=d_model,
            max_program_len=max_len,
            num_registers=num_regs,
            num_opcodes=NUM_OPCODES,
            nhead=4,
        )
        engine = DifferentiableEngine(num_registers=num_regs)

        # Simulated encoder output with gradient tracking
        encoder_output = torch.randn(1, 5, d_model, requires_grad=True)

        program = decoder(encoder_output)
        result = engine.execute_soft(
            program, {}, max_steps=4, temperature=1.0, skip_bitwise=True
        )

        # Compute loss on R0
        target = 42.0
        loss = (result.registers[0] - target) ** 2
        loss.backward()

        # Encoder output should have gradients
        assert encoder_output.grad is not None
        assert encoder_output.grad.abs().sum().item() > 0

        # Decoder parameters should have gradients
        has_grad = False
        for param in decoder.parameters():
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                has_grad = True
                break
        assert has_grad, "No decoder parameter received gradients"

    def test_autoregressive_deterministic(self):
        """Same input produces same output in eval mode."""
        d_model = 32
        decoder = AutoregressiveDecoder(
            d_model=d_model, max_program_len=4, num_registers=4, nhead=4
        )
        decoder.eval()

        encoder_output = torch.randn(1, 3, d_model)

        with torch.no_grad():
            prog1 = decoder(encoder_output)
            prog2 = decoder(encoder_output)

        assert torch.allclose(prog1.opcode_logits, prog2.opcode_logits)
        assert torch.allclose(prog1.dst_logits, prog2.dst_logits)
        assert torch.allclose(prog1.immediates, prog2.immediates)


# =========================================================================
# DifferentiableCompiler with autoregressive mode
# =========================================================================


class TestCompilerAutoregressiveMode:
    """Tests for DifferentiableCompiler in autoregressive decoder mode."""

    def test_compiler_autoregressive_creates(self):
        """Compiler creates in autoregressive mode without error."""
        compiler = DifferentiableCompiler(
            vocab_size=64,
            d_model=32,
            max_program_len=8,
            num_registers=8,
            nhead=4,
            decoder_mode="autoregressive",
        )

        assert compiler.decoder_mode == "autoregressive"
        assert compiler.ar_decoder is not None
        # Single-shot heads should be None
        assert compiler.opcode_head is None
        assert compiler.dst_head is None

    def test_compiler_single_shot_backward_compat(self):
        """Default single_shot mode still works exactly as before."""
        compiler = DifferentiableCompiler(
            vocab_size=64, d_model=32, nhead=4, decoder_mode="single_shot"
        )

        assert compiler.decoder_mode == "single_shot"
        assert compiler.ar_decoder is None
        assert compiler.opcode_head is not None

    def test_compiler_invalid_mode_raises(self):
        """Invalid decoder_mode raises ValueError."""
        with pytest.raises(ValueError, match="decoder_mode must be"):
            DifferentiableCompiler(decoder_mode="invalid")

    def test_autoregressive_compile_produces_soft_program(self):
        """Compile via autoregressive path returns valid SoftProgram."""
        compiler = DifferentiableCompiler(
            vocab_size=64,
            d_model=32,
            max_source_len=16,
            max_program_len=8,
            num_registers=8,
            nhead=4,
            decoder_mode="autoregressive",
        )

        tokens = torch.randint(0, 64, (10,))
        program = compiler.compile(tokens)

        assert isinstance(program, SoftProgram)
        assert program.opcode_logits.shape == (8, NUM_OPCODES)
        assert program.dst_logits.shape == (8, 8)
        assert program.immediates.shape == (8,)

    def test_autoregressive_forward_matches_compile(self):
        """forward() and compile() produce equivalent results."""
        compiler = DifferentiableCompiler(
            vocab_size=64, d_model=32, max_program_len=4,
            nhead=4, decoder_mode="autoregressive",
        )
        compiler.eval()

        tokens = torch.randint(0, 64, (8,))

        with torch.no_grad():
            prog_compile = compiler.compile(tokens)
            prog_forward = compiler(tokens)

        assert torch.allclose(
            prog_compile.opcode_logits, prog_forward.opcode_logits
        )

    def test_autoregressive_training(self):
        """Training with the autoregressive decoder reduces loss.

        We train the compiler to emit a program that places a target
        value in R0 via MOV_IMM.  The loss should decrease over a short
        training run, demonstrating that gradients flow correctly
        through the AR decoder into meaningful weight updates.
        """
        tokenizer = SimpleTokenizer(pad_length=16)
        compiler = DifferentiableCompiler(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            max_source_len=16,
            max_program_len=8,
            num_registers=8,
            nhead=4,
            decoder_mode="autoregressive",
        )
        engine = DifferentiableEngine(num_registers=8)
        optimizer = torch.optim.Adam(compiler.parameters(), lr=0.005)

        tokens = tokenizer.tokenize("add r0 r1 r2")
        target_r2 = 15.0  # R0=10 + R1=5 = 15

        losses = []
        for _step in range(40):
            optimizer.zero_grad()

            program = compiler.compile(tokens)
            result = engine.execute_soft(
                program,
                {0: 10.0, 1: 5.0},
                max_steps=8,
                temperature=1.0,
                skip_bitwise=True,
            )

            loss = (result.registers[2] - target_r2) ** 2
            loss.backward()

            torch.nn.utils.clip_grad_norm_(compiler.parameters(), 5.0)
            optimizer.step()

            losses.append(loss.item())

        # Loss should decrease: final loss should be less than initial loss
        # We use a generous comparison since 40 steps of training on a
        # randomly initialized network may not converge fully, but the
        # trend should be downward.
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_autoregressive_pipeline_integration(self):
        """AR compiler integrates with the full compilation pipeline."""
        tokenizer = SimpleTokenizer(pad_length=16)
        compiler = DifferentiableCompiler(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            max_source_len=16,
            max_program_len=8,
            num_registers=8,
            nhead=4,
            decoder_mode="autoregressive",
        )
        pipeline = DifferentiableCompilationPipeline(
            compiler=compiler,
            engine=DifferentiableEngine(num_registers=8),
        )

        tokens = tokenizer.tokenize("add r0 r1 r2")
        result = pipeline.compile_and_execute(
            tokens,
            inputs={0: 5.0, 1: 3.0},
            max_steps=8,
            temperature=1.0,
            skip_bitwise=True,
        )

        assert result.program is not None
        assert result.execution_result is not None
        assert result.execution_result.registers.shape[0] == 8
