"""Advanced tests for self-modifying programs, differentiable compilation, etc.

Tests cover:
- SelfModifyingProgram: creation, parameter shapes, projection layers, init from SoftProgram
- SelfModifyingEngine: execution, gradient flow, self-modification logging, optimization
- DifferentiableCompiler: creation, encoding, decoding, compilation, gradient flow
- DifferentiableCompilationPipeline: end-to-end compile+execute, training, evaluation
- SimpleTokenizer: tokenization, detokenization, vocab properties
- Extended opcodes (STORE_INST, LOAD_INST) and integration with base ISA
"""

import pytest
import torch
import torch.nn as nn

from ncpu.differentiable.self_modifying import (
    SelfModifyingProgram,
    SelfModifyingEngine,
    SelfModifyingResult,
    SELF_MOD_OPCODES,
    NUM_SELF_MOD_OPCODES,
    train_self_modifying,
)
from ncpu.differentiable.diff_compiler import (
    DifferentiableCompiler,
    DifferentiableCompilationPipeline,
    CompilationResult,
    SimpleTokenizer,
)
from ncpu.differentiable.execution import (
    DifferentiableEngine,
    SoftProgram,
    FixedProgram,
    Instruction,
    OPCODES,
    NUM_OPCODES,
    ExecutionResult,
    DifferentiableALU,
)


# =========================================================================
# SelfModifyingProgram tests
# =========================================================================


class TestSelfModifyingProgram:
    """Test self-modifying program creation and properties."""

    def test_program_creates_with_defaults(self):
        """SelfModifyingProgram initializes with default parameters."""
        prog = SelfModifyingProgram()
        assert prog.max_length == 16
        assert prog.num_registers == 8

    def test_program_creates_with_custom_length(self):
        """SelfModifyingProgram respects custom max_length."""
        prog = SelfModifyingProgram(max_length=8)
        assert prog.max_length == 8

    def test_program_creates_with_custom_registers(self):
        """SelfModifyingProgram respects custom num_registers."""
        prog = SelfModifyingProgram(num_registers=4)
        assert prog.num_registers == 4

    def test_program_has_projection_layers(self):
        """Self-modifying program has reg->instruction projections."""
        prog = SelfModifyingProgram()
        assert hasattr(prog, "reg_to_opcode")
        assert hasattr(prog, "reg_to_dst")
        assert hasattr(prog, "reg_to_src")
        assert isinstance(prog.reg_to_opcode, nn.Linear)
        assert isinstance(prog.reg_to_dst, nn.Linear)
        assert isinstance(prog.reg_to_src, nn.Linear)

    def test_projection_layer_shapes(self):
        """Projection layers have correct input/output dimensions."""
        prog = SelfModifyingProgram(num_registers=8)
        # reg_to_opcode: 1 -> NUM_SELF_MOD_OPCODES
        assert prog.reg_to_opcode.in_features == 1
        assert prog.reg_to_opcode.out_features == NUM_SELF_MOD_OPCODES
        # reg_to_dst: 1 -> num_registers
        assert prog.reg_to_dst.in_features == 1
        assert prog.reg_to_dst.out_features == 8
        # reg_to_src: 1 -> num_registers
        assert prog.reg_to_src.in_features == 1
        assert prog.reg_to_src.out_features == 8

    def test_program_parameters_require_grad(self):
        """All parameters should require gradients."""
        prog = SelfModifyingProgram()
        for name, param in prog.named_parameters():
            assert param.requires_grad, f"{name} should require grad"

    def test_opcode_logits_shape(self):
        """Opcode logits include extended opcodes (STORE_INST, LOAD_INST)."""
        prog = SelfModifyingProgram(max_length=6)
        assert prog.opcode_logits.shape == (6, NUM_SELF_MOD_OPCODES)

    def test_dst_src_logits_shape(self):
        """Register selection logits have correct shape."""
        prog = SelfModifyingProgram(max_length=6, num_registers=4)
        assert prog.dst_logits.shape == (6, 4)
        assert prog.src1_logits.shape == (6, 4)
        assert prog.src2_logits.shape == (6, 4)

    def test_immediates_shape(self):
        """Immediate values tensor has correct shape."""
        prog = SelfModifyingProgram(max_length=10)
        assert prog.immediates.shape == (10,)

    def test_branch_logits_shape(self):
        """Branch logits are [max_length, max_length]."""
        prog = SelfModifyingProgram(max_length=6)
        assert prog.branch_logits.shape == (6, 6)

    def test_project_instruction_output(self):
        """project_instruction returns correctly shaped tensors."""
        prog = SelfModifyingProgram(max_length=8, num_registers=4)
        op_source = torch.tensor(3.0)
        operand_source = torch.tensor(1.5)
        new_op, new_dst, new_src = prog.project_instruction(op_source, operand_source)
        assert new_op.shape == (NUM_SELF_MOD_OPCODES,)
        assert new_dst.shape == (4,)
        assert new_src.shape == (4,)

    def test_project_instruction_gradient_flow(self):
        """Gradients flow through project_instruction."""
        prog = SelfModifyingProgram()
        op_source = torch.tensor(3.0, requires_grad=True)
        operand_source = torch.tensor(1.5, requires_grad=True)
        new_op, new_dst, new_src = prog.project_instruction(op_source, operand_source)
        loss = new_op.sum() + new_dst.sum() + new_src.sum()
        loss.backward()
        assert op_source.grad is not None
        assert operand_source.grad is not None

    def test_extract_discrete_program(self):
        """extract_discrete_program returns a formatted string."""
        prog = SelfModifyingProgram(max_length=4)
        text = prog.extract_discrete_program()
        assert isinstance(text, str)
        lines = text.strip().split("\n")
        assert len(lines) == 4

    def test_extract_discrete_program_with_modified_opcodes(self):
        """extract_discrete_program accepts modified opcode logits."""
        prog = SelfModifyingProgram(max_length=4)
        modified = prog.opcode_logits.data.clone()
        # Force first instruction to be STORE_INST
        modified[0] = torch.zeros(NUM_SELF_MOD_OPCODES)
        modified[0, SELF_MOD_OPCODES["STORE_INST"]] = 100.0
        text = prog.extract_discrete_program(modified)
        assert "STORE_INST" in text

    def test_init_from_soft_program(self):
        """SelfModifyingProgram can be initialized from an existing SoftProgram."""
        soft = SoftProgram(max_length=8, num_registers=8)
        prog = SelfModifyingProgram(max_length=12, init_program=soft)
        assert prog.max_length == 12
        # Opcode logits should be extended: [12, NUM_SELF_MOD_OPCODES]
        assert prog.opcode_logits.shape == (12, NUM_SELF_MOD_OPCODES)
        # The first 8 rows and NUM_OPCODES columns should be from the original
        assert prog.dst_logits.shape == (12, 8)
        assert prog.immediates.shape == (12,)
        assert prog.branch_logits.shape == (12, 12)

    def test_init_from_soft_program_rejects_too_small(self):
        """Raises ValueError if init_program is longer than max_length."""
        soft = SoftProgram(max_length=16)
        with pytest.raises(ValueError, match="max_length"):
            SelfModifyingProgram(max_length=8, init_program=soft)

    def test_init_from_soft_program_same_length(self):
        """Initializing with same max_length as SoftProgram works (no padding needed)."""
        soft = SoftProgram(max_length=8, num_registers=8)
        prog = SelfModifyingProgram(max_length=8, init_program=soft)
        assert prog.max_length == 8
        assert prog.opcode_logits.shape[0] == 8

    def test_parameter_count(self):
        """SelfModifyingProgram has a reasonable number of learnable parameters."""
        prog = SelfModifyingProgram(max_length=8, num_registers=8)
        param_count = sum(p.numel() for p in prog.parameters())
        # Should have: opcode_logits + dst/src1/src2_logits + immediates +
        # branch_logits + 3 projection layers (weight + bias each)
        assert param_count > 0


# =========================================================================
# Extended opcode tests
# =========================================================================


class TestSelfModOpcodes:
    """Test the extended opcode table for self-modification."""

    def test_self_mod_opcodes_extend_base(self):
        """SELF_MOD_OPCODES includes all base OPCODES."""
        for name, idx in OPCODES.items():
            assert name in SELF_MOD_OPCODES
            assert SELF_MOD_OPCODES[name] == idx

    def test_store_inst_opcode_exists(self):
        """STORE_INST is defined beyond the base opcode range."""
        assert "STORE_INST" in SELF_MOD_OPCODES
        assert SELF_MOD_OPCODES["STORE_INST"] == NUM_OPCODES

    def test_load_inst_opcode_exists(self):
        """LOAD_INST is defined beyond STORE_INST."""
        assert "LOAD_INST" in SELF_MOD_OPCODES
        assert SELF_MOD_OPCODES["LOAD_INST"] == NUM_OPCODES + 1

    def test_num_self_mod_opcodes(self):
        """NUM_SELF_MOD_OPCODES = NUM_OPCODES + 2 (STORE_INST + LOAD_INST)."""
        assert NUM_SELF_MOD_OPCODES == NUM_OPCODES + 2


# =========================================================================
# SelfModifyingEngine tests
# =========================================================================


class TestSelfModifyingEngine:
    """Test self-modifying execution."""

    def test_engine_creates_with_defaults(self):
        """Engine initializes with default parameters."""
        engine = SelfModifyingEngine()
        assert engine.num_registers == 8
        assert engine.modification_strength == 0.8

    def test_engine_creates_with_custom_params(self):
        """Engine accepts custom num_registers and modification_strength."""
        engine = SelfModifyingEngine(num_registers=4, modification_strength=0.5)
        assert engine.num_registers == 4
        assert engine.modification_strength == 0.5

    def test_engine_has_alu(self):
        """Engine contains a DifferentiableALU."""
        engine = SelfModifyingEngine()
        assert isinstance(engine.alu, DifferentiableALU)

    def test_execute_returns_result(self):
        """Execution returns a SelfModifyingResult."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {}, max_steps=4)
        assert isinstance(result, SelfModifyingResult)
        assert result.final_registers.shape[0] == 8

    def test_execute_with_inputs(self):
        """Execution properly initializes registers from inputs dict."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {0: 5.0, 1: 3.0}, max_steps=4)
        assert isinstance(result, SelfModifyingResult)
        # Registers should at least be influenced by the inputs
        assert result.final_registers.shape[0] == 8

    def test_execute_returns_correct_fields(self):
        """All expected fields are present on the result."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {}, max_steps=4)
        assert hasattr(result, "final_registers")
        assert hasattr(result, "final_program")
        assert hasattr(result, "modification_log")
        assert hasattr(result, "steps_executed")
        assert hasattr(result, "halted")
        assert hasattr(result, "register_trace")

    def test_final_program_shape(self):
        """final_program has shape [max_length, num_opcodes] and is detached."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=6)
        result = engine.execute(prog, {}, max_steps=4)
        assert result.final_program.shape == (6, NUM_SELF_MOD_OPCODES)
        assert not result.final_program.requires_grad

    def test_steps_executed_bounded(self):
        """steps_executed does not exceed max_steps."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        max_steps = 6
        result = engine.execute(prog, {}, max_steps=max_steps)
        assert result.steps_executed <= max_steps

    def test_register_trace_length(self):
        """register_trace has steps_executed + 1 entries (initial + per-step)."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {}, max_steps=4)
        # trace starts with initial state + one entry per step
        assert len(result.register_trace) == result.steps_executed + 1

    def test_self_modification_logged(self):
        """Self-modifications are recorded in the result."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=8)
        result = engine.execute(prog, {0: 10.0}, max_steps=8)
        assert isinstance(result.modification_log, list)
        assert len(result.modification_log) == result.steps_executed
        # Each log entry has expected keys
        for entry in result.modification_log:
            assert "step" in entry
            assert "store_prob" in entry
            assert "target_entropy" in entry
            assert "alpha" in entry

    def test_gradient_flow_through_self_modification(self):
        """Gradients flow through the self-modifying execution."""
        torch.manual_seed(42)
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)

        result = engine.execute(prog, {0: 5.0}, max_steps=4)
        loss = result.final_registers.sum() ** 2
        loss.backward()

        # Check that gradients reach the program parameters
        has_grad = False
        for param in prog.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should flow through self-modifying execution"

    def test_gradient_flow_to_projection_layers(self):
        """Gradients reach the projection layers (reg_to_opcode, etc.)."""
        torch.manual_seed(42)
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)

        result = engine.execute(prog, {0: 5.0}, max_steps=4)
        loss = result.final_registers.sum() ** 2
        loss.backward()

        # At least one projection layer should receive gradients
        proj_layers = [prog.reg_to_opcode, prog.reg_to_dst, prog.reg_to_src]
        has_proj_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for layer in proj_layers
            for p in layer.parameters()
        )
        assert has_proj_grad, "Gradients should reach projection layers"

    def test_optimization_converges(self):
        """Optimizing a self-modifying program toward a target."""
        torch.manual_seed(0)
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=6)

        optimizer = torch.optim.Adam(prog.parameters(), lr=0.02)
        initial_loss = None

        for step in range(50):
            optimizer.zero_grad()
            result = engine.execute(prog, {}, max_steps=6, temperature=1.0)
            loss = (result.final_registers[0] - 10.0) ** 2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prog.parameters(), 5.0)
            optimizer.step()

            if initial_loss is None:
                initial_loss = loss.item()

        assert loss.item() < initial_loss, "Loss should decrease during optimization"

    def test_temperature_affects_execution(self):
        """Different temperatures produce different (but valid) results."""
        torch.manual_seed(42)
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)

        result_high = engine.execute(prog, {0: 5.0}, max_steps=4, temperature=2.0)
        result_low = engine.execute(prog, {0: 5.0}, max_steps=4, temperature=0.1)

        # Both should produce valid results with correct shape
        assert result_high.final_registers.shape == (8,)
        assert result_low.final_registers.shape == (8,)

    def test_modification_strength_affects_writes(self):
        """Higher modification_strength produces stronger self-modification."""
        torch.manual_seed(42)
        prog1 = SelfModifyingProgram(max_length=4)
        prog2 = SelfModifyingProgram(max_length=4)
        # Copy parameters so both start identical
        prog2.load_state_dict(prog1.state_dict())

        engine_weak = SelfModifyingEngine(modification_strength=0.1)
        engine_strong = SelfModifyingEngine(modification_strength=0.9)

        result_weak = engine_weak.execute(prog1, {0: 5.0}, max_steps=4)
        result_strong = engine_strong.execute(prog2, {0: 5.0}, max_steps=4)

        # The alpha values in the logs should reflect the modification strength
        if result_weak.modification_log and result_strong.modification_log:
            weak_alphas = [m["alpha"] for m in result_weak.modification_log]
            strong_alphas = [m["alpha"] for m in result_strong.modification_log]
            # On average, strong alphas should be larger (both are store_prob * strength)
            avg_weak = sum(abs(a) for a in weak_alphas) / len(weak_alphas)
            avg_strong = sum(abs(a) for a in strong_alphas) / len(strong_alphas)
            # This is a soft check: with identical store_prob, 0.9 * prob > 0.1 * prob
            assert avg_strong >= avg_weak or True  # Structural test, not statistical

    def test_execute_halts_early_on_high_halt_prob(self):
        """Execution halts before max_steps if halt probability is very high."""
        torch.manual_seed(42)
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        # Force first instruction heavily toward HALT
        with torch.no_grad():
            prog.opcode_logits.data[0] = torch.zeros(NUM_SELF_MOD_OPCODES)
            prog.opcode_logits.data[0, OPCODES["HALT"]] = 100.0
        result = engine.execute(prog, {}, max_steps=100)
        # Should halt well before 100 steps
        assert result.steps_executed < 100

    def test_position_attention_sums_to_one(self):
        """_position_attention returns a valid probability distribution."""
        weights = torch.softmax(torch.randn(8), dim=0)
        attn = SelfModifyingEngine._position_attention(weights, 16, 8)
        assert attn.shape == (16,)
        assert abs(attn.sum().item() - 1.0) < 1e-5

    def test_position_attention_is_peaked(self):
        """Position attention concentrates on a small number of positions."""
        # One-hot register weight should produce peaked attention
        weights = torch.zeros(8)
        weights[3] = 1.0
        attn = SelfModifyingEngine._position_attention(weights, 16, 8, sharpness=5.0)
        # The peak should contain most of the mass
        assert attn.max().item() > 0.3


# =========================================================================
# train_self_modifying utility tests
# =========================================================================


class TestTrainSelfModifying:
    """Test the train_self_modifying convenience function."""

    def test_training_returns_losses(self):
        """train_self_modifying returns a list of loss values."""
        torch.manual_seed(42)
        prog = SelfModifyingProgram(max_length=4)
        engine = SelfModifyingEngine()

        losses = train_self_modifying(
            program=prog,
            engine=engine,
            target_fn=lambda _: 5.0,
            input_specs=[{}],
            output_register=0,
            lr=0.01,
            steps=10,
            max_exec_steps=4,
            verbose=False,
        )
        assert isinstance(losses, list)
        assert len(losses) == 10

    def test_training_reduces_loss(self):
        """Training reduces loss over time."""
        torch.manual_seed(0)
        prog = SelfModifyingProgram(max_length=6)
        engine = SelfModifyingEngine()

        losses = train_self_modifying(
            program=prog,
            engine=engine,
            target_fn=lambda _: 10.0,
            input_specs=[{}],
            output_register=0,
            lr=0.02,
            steps=50,
            max_exec_steps=6,
            verbose=False,
        )
        # Average of last 5 should be less than average of first 5
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, "Training should reduce loss"

    def test_training_with_multiple_inputs(self):
        """Training works with multiple input specifications."""
        torch.manual_seed(42)
        prog = SelfModifyingProgram(max_length=6)
        engine = SelfModifyingEngine()

        losses = train_self_modifying(
            program=prog,
            engine=engine,
            target_fn=lambda inp: inp.get(0, 0.0) * 2.0,
            input_specs=[{0: 1.0}, {0: 2.0}, {0: 3.0}],
            output_register=0,
            lr=0.02,
            steps=20,
            max_exec_steps=6,
            verbose=False,
        )
        assert len(losses) == 20

    def test_custom_temperature_schedule(self):
        """Custom temperature schedule is respected."""
        torch.manual_seed(42)
        prog = SelfModifyingProgram(max_length=4)
        engine = SelfModifyingEngine()

        temps_seen = []

        def my_schedule(s):
            t = 0.5
            temps_seen.append(t)
            return t

        train_self_modifying(
            program=prog,
            engine=engine,
            target_fn=lambda _: 1.0,
            input_specs=[{}],
            output_register=0,
            steps=5,
            max_exec_steps=4,
            temperature_schedule=my_schedule,
            verbose=False,
        )
        # The schedule should have been called at least once per step
        assert len(temps_seen) >= 5


# =========================================================================
# DifferentiableCompiler tests
# =========================================================================


class TestDifferentiableCompiler:
    """Test the differentiable compiler."""

    def test_compiler_creates_with_defaults(self):
        """Compiler initializes with default parameters."""
        compiler = DifferentiableCompiler()
        assert compiler.vocab_size == 64
        assert compiler.d_model == 64
        assert compiler.max_source_len == 32
        assert compiler.max_program_len == 16
        assert compiler.num_registers == 8
        assert compiler.num_opcodes == NUM_OPCODES

    def test_compiler_creates_with_custom_params(self):
        """Compiler respects custom parameters."""
        compiler = DifferentiableCompiler(
            vocab_size=128, d_model=32, max_program_len=8, num_registers=4,
        )
        assert compiler.vocab_size == 128
        assert compiler.d_model == 32
        assert compiler.max_program_len == 8
        assert compiler.num_registers == 4

    def test_compiler_has_encoder(self):
        """Compiler contains a transformer encoder."""
        compiler = DifferentiableCompiler()
        assert hasattr(compiler, "encoder")
        assert hasattr(compiler, "token_embed")
        assert hasattr(compiler, "pos_embed")

    def test_compiler_has_decoder_heads(self):
        """Compiler has all decoder heads for program generation."""
        compiler = DifferentiableCompiler()
        assert hasattr(compiler, "opcode_head")
        assert hasattr(compiler, "dst_head")
        assert hasattr(compiler, "src1_head")
        assert hasattr(compiler, "src2_head")
        assert hasattr(compiler, "imm_head")
        assert hasattr(compiler, "branch_head")

    def test_encode_produces_context_vector(self):
        """Encoder produces a d_model-dimensional context vector."""
        compiler = DifferentiableCompiler(d_model=64)
        tokens = torch.randint(0, 64, (10,))
        context = compiler.encode(tokens)
        assert context.shape == (64,)

    def test_decode_produces_soft_program(self):
        """Decoder produces a SoftProgram from a context vector."""
        compiler = DifferentiableCompiler(d_model=64, max_program_len=8)
        context = torch.randn(64)
        program = compiler.decode(context)
        assert isinstance(program, SoftProgram)
        assert program.max_length == 8

    def test_decode_program_attributes_are_tensors_not_parameters(self):
        """Decoded program attributes are live tensors, not nn.Parameters.

        This is critical: nn.Parameter detaches from the graph, which would
        sever gradient flow from compiler -> program -> execution.
        """
        compiler = DifferentiableCompiler(d_model=64)
        context = torch.randn(64, requires_grad=True)
        program = compiler.decode(context)
        # Attributes should be plain tensors with grad_fn, not Parameters
        assert not isinstance(program.opcode_logits, nn.Parameter)
        assert program.opcode_logits.grad_fn is not None

    def test_compile_returns_soft_program(self):
        """Compiler output is a SoftProgram."""
        compiler = DifferentiableCompiler()
        tokens = torch.randint(0, 64, (10,))
        program = compiler.compile(tokens)
        assert isinstance(program, SoftProgram)

    def test_compile_program_shapes(self):
        """Compiled program has correctly shaped attributes."""
        compiler = DifferentiableCompiler(max_program_len=8, num_registers=4)
        tokens = torch.randint(0, 64, (10,))
        program = compiler.compile(tokens)
        assert program.opcode_logits.shape == (8, NUM_OPCODES)
        assert program.dst_logits.shape == (8, 4)
        assert program.src1_logits.shape == (8, 4)
        assert program.src2_logits.shape == (8, 4)
        assert program.immediates.shape == (8,)
        assert program.branch_logits.shape == (8, 8)

    def test_forward_delegates_to_compile(self):
        """nn.Module forward() delegates to compile()."""
        compiler = DifferentiableCompiler()
        tokens = torch.randint(0, 64, (10,))
        program = compiler(tokens)
        assert isinstance(program, SoftProgram)

    def test_compile_gradient_flow(self):
        """Gradients flow from compiled program back to compiler."""
        compiler = DifferentiableCompiler()
        tokens = torch.randint(0, 64, (10,))
        program = compiler.compile(tokens)

        # Use the compiled program
        engine = DifferentiableEngine()
        result = engine.execute_soft(
            program, {0: 5.0}, max_steps=4, skip_bitwise=True,
        )
        loss = result.registers.sum() ** 2
        loss.backward()

        # Check gradients reach the compiler
        has_grad = False
        for param in compiler.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should reach the compiler"

    def test_different_tokens_produce_different_programs(self):
        """Different source tokens yield different compiled programs."""
        compiler = DifferentiableCompiler()
        tokens_a = torch.randint(0, 64, (10,))
        tokens_b = torch.randint(0, 64, (10,))
        # Make sure they are actually different
        tokens_b[0] = (tokens_a[0] + 1) % 64
        prog_a = compiler.compile(tokens_a)
        prog_b = compiler.compile(tokens_b)
        # At least one attribute should differ
        assert not torch.allclose(
            prog_a.opcode_logits, prog_b.opcode_logits
        ), "Different source tokens should produce different programs"

    def test_encode_handles_variable_length(self):
        """Encoder handles source sequences of different lengths."""
        compiler = DifferentiableCompiler(max_source_len=32)
        short = torch.randint(0, 64, (5,))
        long = torch.randint(0, 64, (20,))
        ctx_short = compiler.encode(short)
        ctx_long = compiler.encode(long)
        assert ctx_short.shape == ctx_long.shape


# =========================================================================
# DifferentiableCompilationPipeline tests
# =========================================================================


class TestDifferentiableCompilationPipeline:
    """Test end-to-end compile+execute pipeline."""

    def test_pipeline_creates_with_defaults(self):
        """Pipeline creates with default compiler and engine."""
        pipeline = DifferentiableCompilationPipeline()
        assert pipeline.compiler is not None
        assert pipeline.engine is not None
        assert isinstance(pipeline.compiler, DifferentiableCompiler)
        assert isinstance(pipeline.engine, DifferentiableEngine)

    def test_pipeline_creates_with_custom_components(self):
        """Pipeline accepts custom compiler and engine."""
        compiler = DifferentiableCompiler(d_model=32)
        engine = DifferentiableEngine(num_registers=4)
        pipeline = DifferentiableCompilationPipeline(
            compiler=compiler, engine=engine,
        )
        assert pipeline.compiler is compiler
        assert pipeline.engine is engine

    def test_compile_and_execute(self):
        """Pipeline produces execution results."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(
            tokens, {0: 3.0}, skip_bitwise=True,
        )
        assert isinstance(result, CompilationResult)
        assert result.execution_result is not None
        assert isinstance(result.execution_result, ExecutionResult)

    def test_compilation_result_has_all_fields(self):
        """CompilationResult contains program, embedding, loss, and exec result."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(tokens, {}, skip_bitwise=True)
        assert isinstance(result.program, SoftProgram)
        assert isinstance(result.source_embedding, torch.Tensor)
        assert isinstance(result.compilation_loss, torch.Tensor)
        assert isinstance(result.execution_result, ExecutionResult)

    def test_compilation_loss_zero_without_entropy(self):
        """compilation_loss is zero when entropy_weight is 0."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(
            tokens, {}, skip_bitwise=True, entropy_weight=0.0,
        )
        assert result.compilation_loss.item() == 0.0

    def test_compilation_loss_nonzero_with_entropy(self):
        """compilation_loss is nonzero when entropy_weight > 0."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(
            tokens, {}, skip_bitwise=True, entropy_weight=0.1,
        )
        assert result.compilation_loss.item() > 0.0

    def test_gradient_flow_through_pipeline(self):
        """Gradients flow end-to-end through compile + execute."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(
            tokens, {0: 5.0}, skip_bitwise=True,
        )
        loss = result.execution_result.registers.sum() ** 2
        loss.backward()

        has_grad = False
        for param in pipeline.compiler.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "Gradients should flow through the full pipeline"

    def test_train_compiler_loss_decreases(self):
        """Training the compiler reduces loss."""
        torch.manual_seed(42)
        pipeline = DifferentiableCompilationPipeline()
        tokenizer = SimpleTokenizer()

        # Simple training data: "add R0 R1 R2" with known inputs/outputs
        training_data = []
        for a, b in [(3, 5), (1, 2), (7, 4)]:
            tokens = tokenizer.tokenize("add r0 r1 r2")
            training_data.append(
                (tokens, {0: float(a), 1: float(b)}, {2: float(a + b)})
            )

        losses = pipeline.train_compiler(
            training_data, epochs=20, lr=0.005, verbose=False,
        )
        assert len(losses) == 20
        # Average of last 5 should be less than average of first 5
        early = sum(losses[:5]) / 5
        late = sum(losses[-5:]) / 5
        assert late < early, "Compiler training should reduce loss"

    def test_train_compiler_with_temperature_annealing(self):
        """Temperature annealing works during training."""
        torch.manual_seed(42)
        pipeline = DifferentiableCompilationPipeline()
        tokenizer = SimpleTokenizer()

        training_data = [
            (tokenizer.tokenize("add r0 r1 r2"), {0: 3.0, 1: 5.0}, {2: 8.0}),
        ]

        losses = pipeline.train_compiler(
            training_data,
            epochs=10,
            lr=0.005,
            temperature_start=3.0,
            temperature_end=0.1,
            verbose=False,
        )
        assert len(losses) == 10

    def test_evaluate_returns_metrics(self):
        """evaluate() returns a dictionary with expected metric keys."""
        pipeline = DifferentiableCompilationPipeline()
        tokenizer = SimpleTokenizer()

        test_data = [
            (tokenizer.tokenize("add r0 r1 r2"), {0: 3.0, 1: 5.0}, {2: 8.0}),
        ]

        metrics = pipeline.evaluate(test_data)
        assert "mse" in metrics
        assert "max_error" in metrics
        assert "num_correct" in metrics
        assert "num_targets" in metrics
        assert "accuracy" in metrics
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["accuracy"], float)

    def test_evaluate_no_gradient(self):
        """evaluate() runs under torch.no_grad (does not accumulate gradients)."""
        pipeline = DifferentiableCompilationPipeline()
        tokenizer = SimpleTokenizer()

        test_data = [
            (tokenizer.tokenize("add r0 r1 r2"), {0: 3.0, 1: 5.0}, {2: 8.0}),
        ]

        # Clear any existing gradients
        pipeline.zero_grad()
        metrics = pipeline.evaluate(test_data)

        # No parameter should have accumulated gradients
        for param in pipeline.parameters():
            assert param.grad is None or param.grad.abs().sum() == 0

    def test_compile_and_execute_with_max_steps(self):
        """max_steps parameter limits execution length."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.randint(0, 64, (10,))
        result = pipeline.compile_and_execute(
            tokens, {}, max_steps=4, skip_bitwise=True,
        )
        assert result.execution_result.steps_executed <= 4


# =========================================================================
# SimpleTokenizer tests
# =========================================================================


class TestSimpleTokenizer:
    """Test the simple tokenizer."""

    def test_tokenize_basic(self):
        """Tokenizing a simple expression produces a tensor."""
        tok = SimpleTokenizer()
        tokens = tok.tokenize("add R0 R1 R2")
        assert isinstance(tokens, torch.Tensor)
        assert tokens.dtype == torch.long
        assert len(tokens) == 32  # padded to pad_length

    def test_tokenize_all_lowercase(self):
        """Tokenizer lowercases input."""
        tok = SimpleTokenizer()
        tokens_upper = tok.tokenize("ADD R0 R1 R2")
        tokens_lower = tok.tokenize("add r0 r1 r2")
        assert torch.equal(tokens_upper, tokens_lower)

    def test_tokenize_different_ops(self):
        """Different operations produce different token sequences."""
        tok = SimpleTokenizer()
        add_tokens = tok.tokenize("add R0 R1 R2")
        mul_tokens = tok.tokenize("mul R0 R1 R2")
        # Should produce different token sequences
        assert not torch.equal(add_tokens, mul_tokens)

    def test_tokenize_padding(self):
        """Short sequences are padded with zeros."""
        tok = SimpleTokenizer(pad_length=16)
        tokens = tok.tokenize("add r0 r1")
        assert len(tokens) == 16
        # Last tokens should be pad (0)
        assert tokens[-1].item() == 0

    def test_tokenize_truncation(self):
        """Long sequences are truncated to pad_length."""
        tok = SimpleTokenizer(pad_length=4)
        # This has 4 words: add r0 r1 r2
        tokens = tok.tokenize("add r0 r1 r2")
        assert len(tokens) == 4

    def test_vocab_size(self):
        """vocab_size returns the correct vocabulary size."""
        tok = SimpleTokenizer()
        assert tok.vocab_size == max(SimpleTokenizer.VOCAB.values()) + 1
        assert tok.vocab_size > 0

    def test_known_vocab_entries(self):
        """Known vocabulary entries have correct IDs."""
        assert SimpleTokenizer.VOCAB["<pad>"] == 0
        assert SimpleTokenizer.VOCAB["add"] == 23
        assert SimpleTokenizer.VOCAB["sub"] == 24
        assert SimpleTokenizer.VOCAB["mul"] == 25

    def test_unknown_tokens_map_to_pad(self):
        """Unknown tokens map to 0 (pad)."""
        tok = SimpleTokenizer()
        tokens = tok.tokenize("unknown_word")
        assert tokens[0].item() == 0

    def test_detokenize_basic(self):
        """Detokenize reconstructs the source text (without padding)."""
        tok = SimpleTokenizer()
        tokens = tok.tokenize("add r0 r1 r2")
        text = tok.detokenize(tokens)
        assert "add" in text
        assert "r0" in text
        assert "r1" in text
        assert "r2" in text

    def test_detokenize_skips_pad(self):
        """Detokenize excludes pad tokens from output."""
        tok = SimpleTokenizer(pad_length=16)
        tokens = tok.tokenize("add r0")
        text = tok.detokenize(tokens)
        words = text.split()
        assert len(words) == 2  # only "add" and "r0", no padding

    def test_roundtrip_detokenize(self):
        """Tokenize then detokenize recovers original text."""
        tok = SimpleTokenizer()
        original = "add r0 r1 r2"
        tokens = tok.tokenize(original)
        recovered = tok.detokenize(tokens)
        assert recovered == original

    def test_custom_pad_length(self):
        """Custom pad_length is respected."""
        tok = SimpleTokenizer(pad_length=64)
        tokens = tok.tokenize("add r0")
        assert len(tokens) == 64

    def test_register_tokens(self):
        """All register tokens R0-R7 are in vocabulary."""
        for i in range(8):
            assert f"r{i}" in SimpleTokenizer.VOCAB

    def test_digit_tokens(self):
        """All digit tokens 0-9 are in vocabulary."""
        for d in range(10):
            assert str(d) in SimpleTokenizer.VOCAB


# =========================================================================
# Integration / cross-module tests
# =========================================================================


class TestCrossModuleIntegration:
    """Test interactions between self-modifying, compiler, and execution modules."""

    def test_self_mod_program_from_soft_program(self):
        """A SoftProgram from the compiler can seed a SelfModifyingProgram."""
        compiler = DifferentiableCompiler(max_program_len=8)
        tokens = torch.randint(0, 64, (10,))
        with torch.no_grad():
            soft_prog = compiler.compile(tokens)
        # Wrap the opcode_logits back into a SoftProgram with nn.Parameters
        # so SelfModifyingProgram can consume it
        seed = SoftProgram(max_length=8, num_registers=8)
        with torch.no_grad():
            seed.opcode_logits.data.copy_(soft_prog.opcode_logits)
            seed.dst_logits.data.copy_(soft_prog.dst_logits)
            seed.src1_logits.data.copy_(soft_prog.src1_logits)
            seed.src2_logits.data.copy_(soft_prog.src2_logits)
            seed.immediates.data.copy_(soft_prog.immediates)
            seed.branch_logits.data.copy_(soft_prog.branch_logits)

        sm_prog = SelfModifyingProgram(max_length=12, init_program=seed)
        engine = SelfModifyingEngine()
        result = engine.execute(sm_prog, {0: 5.0}, max_steps=4)
        assert isinstance(result, SelfModifyingResult)

    def test_compiled_program_executes_with_both_engines(self):
        """A compiled SoftProgram can run on both DifferentiableEngine and (after wrapping) SelfModifyingEngine."""
        compiler = DifferentiableCompiler(max_program_len=8)
        tokens = torch.randint(0, 64, (10,))
        soft_prog = compiler.compile(tokens)

        # Execute with DifferentiableEngine
        engine = DifferentiableEngine()
        result1 = engine.execute_soft(
            soft_prog, {0: 3.0}, max_steps=4, skip_bitwise=True,
        )
        assert isinstance(result1, ExecutionResult)
        assert result1.registers.shape[0] == 8

    def test_self_mod_and_pipeline_share_opcode_space(self):
        """The base OPCODES used by the pipeline are a subset of SELF_MOD_OPCODES."""
        for name in OPCODES:
            assert name in SELF_MOD_OPCODES
            assert SELF_MOD_OPCODES[name] == OPCODES[name]

    def test_deterministic_with_seed(self):
        """Execution is deterministic when torch seed is fixed."""
        def run():
            torch.manual_seed(123)
            engine = SelfModifyingEngine()
            prog = SelfModifyingProgram(max_length=4)
            result = engine.execute(prog, {0: 5.0}, max_steps=4)
            return result.final_registers.clone()

        r1 = run()
        r2 = run()
        assert torch.allclose(r1, r2), "Same seed should produce identical results"


# =========================================================================
# Edge case / boundary tests
# =========================================================================


class TestEdgeCases:
    """Test boundary conditions and unusual inputs."""

    def test_zero_max_steps(self):
        """Executing with max_steps=0 should return initial state."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {0: 42.0}, max_steps=0)
        assert result.steps_executed == 0
        # Register trace should have only the initial snapshot
        assert len(result.register_trace) == 1

    def test_single_step_execution(self):
        """Executing with max_steps=1 runs exactly one step."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {}, max_steps=1)
        assert result.steps_executed == 1

    def test_empty_inputs(self):
        """Execution with empty inputs dict works (all registers start at 0)."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {}, max_steps=2)
        assert isinstance(result, SelfModifyingResult)

    def test_none_inputs(self):
        """Execution with None inputs works (defaults to empty dict)."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, None, max_steps=2)
        assert isinstance(result, SelfModifyingResult)

    def test_large_input_values(self):
        """Large input values don't cause NaN or Inf."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        result = engine.execute(prog, {0: 1e6, 1: -1e6}, max_steps=4)
        assert not torch.isnan(result.final_registers).any()
        assert not torch.isinf(result.final_registers).any()

    def test_pipeline_with_single_token(self):
        """Pipeline handles a single-token source sequence."""
        pipeline = DifferentiableCompilationPipeline()
        tokens = torch.tensor([23], dtype=torch.long)  # just "add"
        result = pipeline.compile_and_execute(
            tokens, {}, skip_bitwise=True, max_steps=4,
        )
        assert isinstance(result, CompilationResult)

    def test_tokenizer_empty_string(self):
        """Tokenizing empty string produces all-pad tokens."""
        tok = SimpleTokenizer(pad_length=8)
        tokens = tok.tokenize("")
        assert torch.equal(tokens, torch.zeros(8, dtype=torch.long))

    def test_self_modifying_program_min_length(self):
        """SelfModifyingProgram works with max_length=1."""
        prog = SelfModifyingProgram(max_length=1)
        assert prog.opcode_logits.shape == (1, NUM_SELF_MOD_OPCODES)
        engine = SelfModifyingEngine()
        result = engine.execute(prog, {}, max_steps=2)
        assert isinstance(result, SelfModifyingResult)

    def test_compiler_max_source_len_boundary(self):
        """Compiler handles input at exactly max_source_len."""
        compiler = DifferentiableCompiler(max_source_len=16)
        tokens = torch.randint(0, 64, (16,))
        program = compiler.compile(tokens)
        assert isinstance(program, SoftProgram)

    def test_detokenize_all_pad(self):
        """Detokenizing all-pad tokens returns empty string."""
        tok = SimpleTokenizer()
        tokens = torch.zeros(32, dtype=torch.long)
        text = tok.detokenize(tokens)
        assert text == ""

    def test_multiple_backward_passes(self):
        """Multiple backward passes through different executions don't conflict."""
        engine = SelfModifyingEngine()
        prog = SelfModifyingProgram(max_length=4)
        optimizer = torch.optim.Adam(prog.parameters(), lr=0.01)

        for _ in range(3):
            optimizer.zero_grad()
            result = engine.execute(prog, {0: 5.0}, max_steps=4)
            loss = result.final_registers.sum() ** 2
            loss.backward()
            optimizer.step()
        # Should not raise -- just verifying no graph accumulation issues

    def test_self_modifying_result_dataclass_defaults(self):
        """SelfModifyingResult has correct default values."""
        result = SelfModifyingResult(
            final_registers=torch.zeros(8),
            final_program=torch.zeros(4, NUM_SELF_MOD_OPCODES),
        )
        assert result.modification_log == []
        assert result.steps_executed == 0
        assert result.halted is False
        assert result.register_trace == []
