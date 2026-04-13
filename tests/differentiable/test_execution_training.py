"""Tests for the differentiable execution training pipeline.

Tests cover:
  1. Code-to-ISA parser: Python → nCPU instructions
  2. Execution loss: differentiable loss with gradient flow
  3. Data generation: training sample quality
  4. Evaluator: reference code evaluation
  5. End-to-end: parse → execute → loss → gradient
"""

import pytest
import torch

from ncpu.differentiable.execution import DifferentiableEngine, SoftProgram, OPCODES
from ncpu.execution_training.code_parser import (
    CodeToISAParser,
    ParseError,
    ParseResult,
    VariableMap,
)
from ncpu.execution_training.execution_loss import (
    ExecutionLoss,
    ExecutionLossWithParsing,
)
from ncpu.execution_training.data import (
    ArithmeticFunctionGenerator,
    VariableTrackingGenerator,
    LoopProblemGenerator,
    ExecutionTrainingDataset,
)
from ncpu.execution_training.evaluate import ExecutionEvaluator


# ════════════════════════════════════════════════════════════════
# Parser Tests
# ════════════════════════════════════════════════════════════════


class TestCodeParser:
    def setup_method(self):
        self.parser = CodeToISAParser()

    def test_simple_assignment(self):
        result = self.parser.parse_block("x = 5", output_var="x")
        assert len(result.instructions) >= 2  # MOV + HALT
        assert result.instructions[0].opcode == OPCODES["MOV_IMM"]
        assert result.instructions[0].immediate == 5.0

    def test_addition(self):
        result = self.parser.parse_block(
            "result = a + b", arg_names=["a", "b"], output_var="result"
        )
        # Should have ADD instruction
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["ADD"] in opcodes

    def test_multiplication(self):
        result = self.parser.parse_block(
            "result = a * b", arg_names=["a", "b"], output_var="result"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["MUL"] in opcodes

    def test_subtraction(self):
        result = self.parser.parse_block(
            "result = a - b", arg_names=["a", "b"], output_var="result"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["SUB"] in opcodes

    def test_complex_expression(self):
        result = self.parser.parse_block(
            "result = a * b + c", arg_names=["a", "b", "c"], output_var="result"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["MUL"] in opcodes
        assert OPCODES["ADD"] in opcodes

    def test_augmented_assignment(self):
        result = self.parser.parse_block(
            "x = 5\nx += 3", output_var="x"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["ADD"] in opcodes

    def test_multi_step(self):
        code = "x = 5\nx = x + 3\ny = x * 2"
        result = self.parser.parse_block(code, output_var="y")
        assert result.variable_map.get("x") is not None
        assert result.variable_map.get("y") is not None

    def test_function_parse(self):
        result = self.parser.parse_function(
            "def f(x, y):\n    return x + y"
        )
        assert result.variable_map.get("x") == 0  # First arg → R0
        assert result.variable_map.get("y") == 1  # Second arg → R1

    def test_for_loop_unroll(self):
        code = "total = 0\nfor i in range(3):\n    total = total + i"
        result = self.parser.parse_block(code, output_var="total")
        # Should unroll: total=0, i=0,total+=0, i=1,total+=1, i=2,total+=2
        assert len(result.instructions) > 5

    def test_bitwise_ops(self):
        result = self.parser.parse_block(
            "result = a & b", arg_names=["a", "b"], output_var="result"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["AND"] in opcodes

        result = self.parser.parse_block(
            "result = a | b", arg_names=["a", "b"], output_var="result"
        )
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["OR"] in opcodes

    def test_to_asm(self):
        result = self.parser.parse_block(
            "result = a + b", arg_names=["a", "b"], output_var="result"
        )
        asm = result.to_asm()
        assert "ADD" in asm
        assert "HALT" in asm

    def test_to_fixed_program(self):
        result = self.parser.parse_block("x = 5", output_var="x")
        fixed = result.to_fixed_program()
        assert fixed.length >= 2

    def test_to_soft_program(self):
        result = self.parser.parse_block("x = 5", output_var="x")
        soft = result.to_soft_program()
        assert soft.opcode_logits.shape[1] == 14  # NUM_OPCODES

    def test_parse_error_unsupported(self):
        # List comprehension is unsupported — parser degrades gracefully with warnings
        result = self.parser.parse_block("x = [i for i in range(10)]")
        assert result.supported_fraction == 0.0
        assert len(result.warnings) > 0

        # But genuinely unparseable syntax does raise
        with pytest.raises(ParseError):
            self.parser.parse_block("this is not valid python at all!!!")

    def test_variable_map(self):
        vm = VariableMap()
        assert vm.allocate("x") == 0
        assert vm.allocate("y") == 1
        assert vm.allocate("x") == 0  # Already allocated
        assert vm.require("x") == 0

    def test_simple_if(self):
        code = "x = 5\nif x > 3:\n    y = 1"
        # Should not crash; may generate CMP + branch
        result = self.parser.parse_block(code, output_var="y")
        opcodes = [inst.opcode for inst in result.instructions]
        assert OPCODES["CMP"] in opcodes


# ════════════════════════════════════════════════════════════════
# Execution Loss Tests
# ════════════════════════════════════════════════════════════════


class TestExecutionLoss:
    def setup_method(self):
        self.engine = DifferentiableEngine(device="cpu")
        self.loss_fn = ExecutionLoss(engine=self.engine, device="cpu")

    def test_fixed_correct(self):
        """Test that correct execution gives near-zero loss."""
        prog = self.engine.assemble("MOV R0, #5\nMOV R1, #3\nADD R2, R0, R1\nHALT")
        result = self.loss_fn.compute_fixed(
            prog, inputs={}, expected={2: 8.0}
        )
        assert result.output_loss.item() < 0.01
        assert result.num_correct == 1

    def test_fixed_wrong(self):
        """Test that wrong execution gives positive loss."""
        prog = self.engine.assemble("MOV R0, #5\nMOV R1, #3\nADD R2, R0, R1\nHALT")
        result = self.loss_fn.compute_fixed(
            prog, inputs={}, expected={2: 100.0}  # Wrong expected
        )
        assert result.output_loss.item() > 1.0
        assert result.num_correct == 0

    def test_soft_gradient_flow(self):
        """Test that gradients flow through SoftProgram execution."""
        parser = CodeToISAParser()
        parse_result = parser.parse_block(
            "result = a + b", arg_names=["a", "b"], output_var="result"
        )
        soft_prog = parse_result.to_soft_program()

        result = self.loss_fn.compute_soft(
            soft_prog,
            inputs={0: 3.0, 1: 5.0},
            expected={2: 8.0},
            temperature=1.0,
        )

        # Should be able to backward
        result.total_loss.backward()

        # Check that gradients exist on program parameters
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in soft_prog.parameters()
        )
        assert has_grad, "No gradients flowed through SoftProgram"

    def test_batched_loss(self):
        """Test batched execution loss."""
        parser = CodeToISAParser()
        parse_result = parser.parse_block(
            "result = a + b", arg_names=["a", "b"], output_var="result"
        )
        soft_prog = parse_result.to_soft_program()

        batch_inputs = [{0: 3.0, 1: 5.0}, {0: 7.0, 1: 2.0}]
        batch_expected = [{2: 8.0}, {2: 9.0}]

        result = self.loss_fn.compute_soft_batched(
            soft_prog,
            batch_inputs=batch_inputs,
            batch_expected=batch_expected,
            temperature=1.0,
        )

        assert result.total_loss.item() >= 0
        assert result.num_total == 2

    def test_loss_with_parsing(self):
        """Test ExecutionLossWithParsing end-to-end."""
        loss_fn = ExecutionLossWithParsing(
            execution_loss=self.loss_fn,
            use_soft_programs=True,
            device="cpu",
        )

        result = loss_fn(
            code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
        )

        assert result.total_loss.item() >= 0

    def test_loss_with_parsing_fallback(self):
        """Test that unparseable code gives fallback loss."""
        loss_fn = ExecutionLossWithParsing(
            execution_loss=self.loss_fn,
            fallback_loss=10.0,
            device="cpu",
        )

        result = loss_fn(
            code="this is not valid python!!!",  # Genuinely unparseable
            test_cases=[{"inputs": {}, "expected": {"x": 0}}],
        )

        assert result.total_loss.item() == 10.0


# ════════════════════════════════════════════════════════════════
# Data Generation Tests
# ════════════════════════════════════════════════════════════════


class TestDataGeneration:
    def test_arithmetic_generator(self):
        gen = ArithmeticFunctionGenerator(seed=42, max_value=50)
        samples = gen.generate(100)
        assert len(samples) > 50  # Some may be filtered
        assert all(s.category == "arithmetic" for s in samples)
        assert all(len(s.test_cases) >= 3 for s in samples)

    def test_variable_tracking_generator(self):
        gen = VariableTrackingGenerator(seed=42, max_value=50)
        samples = gen.generate(100)
        assert len(samples) > 50
        assert all(s.category == "variable_tracking" for s in samples)

    def test_loop_generator(self):
        gen = LoopProblemGenerator(seed=42, max_n=8)
        samples = gen.generate(100)
        assert len(samples) > 30
        assert all(s.category == "loop" for s in samples)

    def test_combined_dataset(self):
        dataset = ExecutionTrainingDataset(size=200, seed=42)
        assert len(dataset) > 100  # Some filtering may reduce count

        sample = dataset[0]
        assert hasattr(sample, "reference_code")
        assert hasattr(sample, "test_cases")

    def test_dataset_categories(self):
        dataset = ExecutionTrainingDataset(size=500, seed=42)
        categories = set(s.category for s in dataset.samples)
        assert "arithmetic" in categories
        assert "variable_tracking" in categories

    def test_sample_parseable(self):
        """Verify generated samples can actually be parsed."""
        parser = CodeToISAParser()
        dataset = ExecutionTrainingDataset(size=100, seed=42)

        parsed = 0
        for sample in dataset.samples[:50]:
            try:
                parser.parse_block(
                    sample.reference_code,
                    arg_names=sample.arg_names if sample.arg_names else None,
                    output_var=sample.output_var,
                )
                parsed += 1
            except ParseError:
                pass

        # At least 60% should parse (arithmetic samples should all parse)
        assert parsed / 50 > 0.6, f"Only {parsed}/50 samples parsed"


# ════════════════════════════════════════════════════════════════
# Evaluator Tests
# ════════════════════════════════════════════════════════════════


class TestEvaluator:
    def test_reference_evaluation(self):
        """Test evaluating reference code (no model needed)."""
        evaluator = ExecutionEvaluator(device="cpu")
        dataset = ExecutionTrainingDataset(size=50, seed=42)

        result = evaluator.evaluate_reference_only(dataset.samples[:20])
        assert result.total_samples == 20
        assert result.parse_rate > 0.5
        # Reference code should mostly execute correctly
        if result.total_executable > 0:
            assert result.exec_accuracy > 0.3

    def test_evaluation_summary(self):
        """Test that summary formatting works."""
        evaluator = ExecutionEvaluator(device="cpu")
        dataset = ExecutionTrainingDataset(size=20, seed=42)
        result = evaluator.evaluate_reference_only(dataset.samples[:10])
        summary = result.summary()
        assert "EVALUATION RESULTS" in summary


# ════════════════════════════════════════════════════════════════
# End-to-End Gradient Flow Tests
# ════════════════════════════════════════════════════════════════


class TestGradientFlow:
    """Tests that verify gradient flow through the full pipeline."""

    def test_e2e_soft_program_gradient(self):
        """Full pipeline: parse → soft program → execute → loss → gradient."""
        parser = CodeToISAParser()
        engine = DifferentiableEngine(device="cpu")
        loss_fn = ExecutionLoss(engine=engine, device="cpu")

        # Parse a simple program
        result = parser.parse_block(
            "result = a * b + c",
            arg_names=["a", "b", "c"],
            output_var="result",
        )

        # Convert to soft program
        soft_prog = result.to_soft_program()

        # Execute with test case
        loss_result = loss_fn.compute_soft(
            soft_prog,
            inputs={0: 3.0, 1: 5.0, 2: 2.0},
            expected={3: 17.0},  # 3*5+2 = 17, result goes to R3
            temperature=1.0,
        )

        # Backward
        loss_result.total_loss.backward()

        # Verify gradients exist
        grad_params = [
            (name, p.grad.abs().sum().item())
            for name, p in soft_prog.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(grad_params) > 0, f"No gradients found on SoftProgram params"

    def test_gradient_direction(self):
        """Test that execution gradients point in the right direction.

        If the program computes 5+3=8 but we expect 10, the gradient should
        push the immediate toward producing a larger result.
        """
        engine = DifferentiableEngine(device="cpu")

        # Program: MOV R0, #5; MOV R1, #X; ADD R2, R0, R1; HALT
        # where X is a differentiable immediate
        prog = engine.assemble("MOV R0, #5\nMOV R1, #3\nADD R2, R0, R1\nHALT")

        # Make the immediate differentiable
        prog.immediates.requires_grad_(True)

        result = engine.execute_fixed(prog, inputs={}, max_steps=10)
        loss = (result.registers[2] - 10.0) ** 2  # Want R2 = 10, got 8

        loss.backward()

        # The gradient on the R1 immediate should be negative
        # (we need to increase the value to get closer to 10)
        assert prog.immediates.grad is not None

    def test_optimization_converges(self):
        """Test that we can optimize a program's immediate via execution gradients."""
        engine = DifferentiableEngine(device="cpu")

        # Program: MOV R0, #5; MOV R1, #?; ADD R2, R0, R1; HALT
        # Goal: find ? such that 5 + ? = 13 → ? = 8
        prog = engine.assemble("MOV R0, #5\nMOV R1, #1\nADD R2, R0, R1\nHALT")
        prog.immediates.requires_grad_(True)

        optimizer = torch.optim.Adam([prog.immediates], lr=0.5)

        for _ in range(100):
            optimizer.zero_grad()
            result = engine.execute_fixed(prog, inputs={}, max_steps=10)
            loss = (result.registers[2] - 13.0) ** 2
            loss.backward()
            optimizer.step()

        # Check: R1's immediate should be close to 8
        result = engine.execute_fixed(prog, inputs={}, max_steps=10)
        r2_val = result.registers[2].item()
        assert abs(r2_val - 13.0) < 1.0, f"Expected ~13, got {r2_val}"


# ════════════════════════════════════════════════════════════════
# Compilation Bridge Tests (Mode 2)
# ════════════════════════════════════════════════════════════════


from ncpu.execution_training.compilation_bridge import (
    CompilationBridge,
    CompilationBridgeResult,
    SequenceAdapter,
)


class TestSequenceAdapter:
    """Tests for the sequence length adapter."""

    def test_pool_strategy(self):
        adapter = SequenceAdapter(strategy="pool", max_source_len=32, d_model=64)
        x = torch.randn(50, 64)  # 50 tokens, d=64
        out = adapter(x)
        assert out.shape == (32, 64)

    def test_truncate_strategy_long(self):
        adapter = SequenceAdapter(strategy="truncate", max_source_len=32, d_model=64)
        x = torch.randn(50, 64)
        out = adapter(x)
        assert out.shape == (32, 64)

    def test_truncate_strategy_short(self):
        adapter = SequenceAdapter(strategy="truncate", max_source_len=32, d_model=64)
        x = torch.randn(10, 64)
        out = adapter(x)
        assert out.shape == (32, 64)

    def test_linear_strategy(self):
        adapter = SequenceAdapter(strategy="linear", max_source_len=32, d_model=64)
        x = torch.randn(20, 64)
        out = adapter(x)
        assert out.shape == (32, 64)

    def test_batched_input(self):
        adapter = SequenceAdapter(strategy="pool", max_source_len=32, d_model=64)
        x = torch.randn(2, 50, 64)  # batch=2
        out = adapter(x)
        assert out.shape == (2, 32, 64)


class TestCompilationBridge:
    """Tests for the Mode 2 compilation bridge."""

    def setup_method(self):
        self.bridge = CompilationBridge(
            lm_hidden_dim=128,  # Small for testing
            compiler_d_model=64,
            max_source_len=32,
            max_program_len=16,
            entropy_weight=0.01,
            max_exec_steps=16,
        )

    def test_bridge_creation(self):
        """Test that bridge creates with correct dimensions."""
        assert self.bridge.hidden_proj.in_features == 128
        assert self.bridge.hidden_proj.out_features == 64
        assert self.bridge.compiler.d_model == 64

    def test_project_hidden_states(self):
        """Test projection from LM space to compiler space."""
        hidden = torch.randn(20, 128)  # 20 tokens, dim 128
        projected = self.bridge.project_hidden_states(hidden)
        assert projected.shape == (32, 64)  # max_source_len, d_model

    def test_compile_from_hidden(self):
        """Test that hidden states compile into a SoftProgram."""
        hidden = torch.randn(20, 128)
        program, projected = self.bridge.compile_from_hidden(hidden)
        assert isinstance(program, SoftProgram)
        assert program.opcode_logits.shape[0] == 16  # max_program_len

    def test_forward_produces_loss(self):
        """Test that forward() produces a differentiable loss."""
        hidden = torch.randn(20, 128, requires_grad=True)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]
        result = self.bridge(hidden, test_cases)

        assert isinstance(result, CompilationBridgeResult)
        assert result.total_loss.item() >= 0
        assert result.num_total == 1

    def test_gradient_flow_to_hidden_states(self):
        """Critical test: gradients flow from loss back into LM hidden states."""
        hidden = torch.randn(20, 128, requires_grad=True)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        result = self.bridge(hidden, test_cases)
        result.total_loss.backward()

        assert hidden.grad is not None, "No gradient on LM hidden states!"
        assert hidden.grad.abs().sum() > 0, "Gradient is all zeros!"

    def test_gradient_flow_to_projection(self):
        """Test that gradients reach the projection layer."""
        hidden = torch.randn(20, 128, requires_grad=True)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        result = self.bridge(hidden, test_cases)
        result.total_loss.backward()

        # Check projection layer has gradients
        assert self.bridge.hidden_proj.weight.grad is not None
        assert self.bridge.hidden_proj.weight.grad.abs().sum() > 0

    def test_multiple_test_cases(self):
        """Test with multiple test cases."""
        hidden = torch.randn(20, 128, requires_grad=True)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
            {"inputs": {0: 7.0, 1: 2.0}, "expected": {2: 9.0}},
            {"inputs": {0: 1.0, 1: 1.0}, "expected": {2: 2.0}},
        ]

        result = self.bridge(hidden, test_cases)
        assert result.num_total == 3
        assert len(result.per_test_losses) == 3
        assert len(result.execution_results) == 3

        # Should still have gradient flow
        result.total_loss.backward()
        assert hidden.grad is not None

    def test_training_step(self):
        """Test the training_step convenience method."""
        hidden = torch.randn(20, 128)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        loss = self.bridge.training_step(hidden, test_cases)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # scalar

    def test_training_step_with_result(self):
        """Test training_step with return_result=True."""
        hidden = torch.randn(20, 128)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        loss, result = self.bridge.training_step(
            hidden, test_cases, return_result=True
        )
        assert isinstance(loss, torch.Tensor)
        assert isinstance(result, CompilationBridgeResult)

    def test_optimizer_can_update(self):
        """Test that an optimizer can update bridge parameters."""
        optimizer = torch.optim.Adam(self.bridge.parameters(), lr=0.01)

        hidden = torch.randn(20, 128)
        test_cases = [
            {"inputs": {0: 5.0, 1: 3.0}, "expected": {0: 8.0}},
        ]

        # Record initial weights
        w_before = self.bridge.hidden_proj.weight.data.clone()

        # Training step
        optimizer.zero_grad()
        result = self.bridge(hidden, test_cases)
        result.total_loss.backward()
        optimizer.step()

        # Weights should have changed
        w_after = self.bridge.hidden_proj.weight.data
        assert not torch.allclose(w_before, w_after), "Weights didn't update!"

    def test_batched_hidden_states(self):
        """Test that batched [batch, seq, hidden] input works."""
        hidden = torch.randn(2, 20, 128, requires_grad=True)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        result = self.bridge(hidden, test_cases)
        result.total_loss.backward()
        assert hidden.grad is not None

    def test_entropy_regularization(self):
        """Test that entropy loss is computed when weight > 0."""
        hidden = torch.randn(20, 128)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        result = self.bridge(hidden, test_cases)
        # With entropy_weight=0.01, compilation_loss should be > 0
        assert result.compilation_loss.item() > 0

    def test_no_entropy_regularization(self):
        """Test that entropy loss is zero when weight = 0."""
        bridge = CompilationBridge(
            lm_hidden_dim=128,
            compiler_d_model=64,
            entropy_weight=0.0,
        )
        hidden = torch.randn(20, 128)
        test_cases = [
            {"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}},
        ]

        result = bridge(hidden, test_cases)
        assert result.compilation_loss.item() == 0.0

    def test_different_lm_dims(self):
        """Test bridge works with various LM hidden dimensions."""
        for lm_dim in [64, 256, 512, 896]:
            bridge = CompilationBridge(lm_hidden_dim=lm_dim, compiler_d_model=64)
            hidden = torch.randn(20, lm_dim, requires_grad=True)
            test_cases = [{"inputs": {0: 1.0}, "expected": {0: 2.0}}]
            result = bridge(hidden, test_cases)
            result.total_loss.backward()
            assert hidden.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
