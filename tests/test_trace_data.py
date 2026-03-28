"""Tests for trace-level execution data generation."""

import pytest
import torch

from ncpu.differentiable.execution import DifferentiableEngine
from ncpu.execution_training.trace_data import (
    TraceGenerator,
    TraceSample,
    TraceLossDataset,
    TraceStep,
)
from ncpu.execution_training.execution_loss import ExecutionLoss


class TestTraceGenerator:
    def setup_method(self):
        self.gen = TraceGenerator()

    def test_simple_add_trace(self):
        sample = self.gen.trace_code(
            code="result = a + b",
            inputs={"a": 3, "b": 5},
            arg_names=["a", "b"],
            output_var="result",
        )
        assert isinstance(sample, TraceSample)
        assert sample.n_trace_points > 0
        # Final expected should have the result register with value 8
        assert any(abs(v - 8.0) < 0.01 for v in sample.final_expected.values())

    def test_multi_step_trace(self):
        sample = self.gen.trace_code(
            code="x = a\nx = x + 3\nx = x * 2",
            inputs={"a": 5},
            arg_names=["a"],
            output_var="x",
        )
        # Should have multiple trace points
        assert sample.n_trace_points >= 3
        # Final: x = (5 + 3) * 2 = 16
        assert any(abs(v - 16.0) < 0.01 for v in sample.final_expected.values())

    def test_trace_has_intermediate_states(self):
        sample = self.gen.trace_code(
            code="temp = a * b\nresult = temp + c",
            inputs={"a": 3, "b": 4, "c": 2},
            arg_names=["a", "b", "c"],
            output_var="result",
        )
        # Should have trace points for: MOV instructions + MUL + ADD
        assert sample.n_trace_points >= 2

        # Check that intermediate values appear
        all_expected_vals = set()
        for step in sample.trace:
            for val in step.expected_registers.values():
                all_expected_vals.add(val)

        # temp = 12 should appear in intermediate state
        assert 12.0 in all_expected_vals or any(
            abs(v - 12.0) < 0.01 for v in all_expected_vals
        )

    def test_trace_as_tuples(self):
        sample = self.gen.trace_code(
            code="result = a + b",
            inputs={"a": 3, "b": 5},
            arg_names=["a", "b"],
            output_var="result",
        )
        tuples = sample.trace_as_tuples
        assert isinstance(tuples, list)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tuples)
        assert all(isinstance(t[0], int) and isinstance(t[1], dict) for t in tuples)

    def test_trace_from_sample(self):
        from ncpu.execution_training.data import ExecutionTrainingSample

        sample = ExecutionTrainingSample(
            prompt="test",
            reference_code="result = a + b",
            test_cases=[{"inputs": {"a": 3, "b": 5}, "expected": {"result": 8}}],
            arg_names=["a", "b"],
            output_var="result",
            category="arithmetic",
            difficulty="easy",
        )
        trace = self.gen.trace_sample(sample)
        assert trace is not None
        assert trace.n_trace_points > 0

    def test_trace_from_unparseable_sample(self):
        from ncpu.execution_training.data import ExecutionTrainingSample

        sample = ExecutionTrainingSample(
            prompt="test",
            reference_code="import os",  # Not parseable
            test_cases=[],
            arg_names=[],
            output_var="",
            category="other",
            difficulty="hard",
        )
        trace = self.gen.trace_sample(sample)
        # Should return None or empty trace, not crash
        # (parser may succeed but produce empty trace)

    def test_generated_dataset(self):
        samples = self.gen.generate_traced_dataset(n_samples=50, seed=42)
        assert len(samples) > 20  # Most should succeed
        assert all(isinstance(s, TraceSample) for s in samples)
        assert all(s.n_trace_points > 0 for s in samples)


class TestTraceLossDataset:
    def test_creation(self):
        dataset = TraceLossDataset(n_samples=50, seed=42)
        assert len(dataset) > 20

    def test_indexing(self):
        dataset = TraceLossDataset(n_samples=30, seed=42)
        sample = dataset[0]
        assert isinstance(sample, TraceSample)

    def test_summary(self):
        dataset = TraceLossDataset(n_samples=30, seed=42)
        summary = dataset.summary()
        assert "TraceLossDataset" in summary
        assert "trace length" in summary

    def test_mean_trace_length(self):
        dataset = TraceLossDataset(n_samples=50, seed=42)
        assert dataset.mean_trace_length > 1.0  # Should have multiple trace points


class TestTraceWithExecutionLoss:
    """Test that trace data integrates with the execution loss module."""

    def test_trace_loss_computation(self):
        """Verify that trace-level supervision produces differentiable loss."""
        gen = TraceGenerator()
        engine = DifferentiableEngine(device="cpu")
        loss_fn = ExecutionLoss(
            engine=engine,
            trace_weight=1.0,  # Enable trace loss
            device="cpu",
        )

        sample = gen.trace_code(
            code="result = a + b",
            inputs={"a": 3, "b": 5},
            arg_names=["a", "b"],
            output_var="result",
        )

        # Build inputs for engine
        inputs = {}
        for var_name, val in sample.inputs.items():
            reg = sample.parse_result.variable_map.get(var_name)
            if reg is not None:
                inputs[reg] = float(val)

        # Use soft program for gradient flow
        soft_prog = sample.parse_result.to_soft_program()

        result = loss_fn.compute_soft(
            soft_prog,
            inputs=inputs,
            expected=sample.final_expected,
            expected_trace=sample.trace_as_tuples,
            temperature=1.0,
        )

        assert result.total_loss.item() >= 0
        assert result.trace_loss is not None

    def test_trace_gradient_flow(self):
        """Verify gradients flow through trace loss."""
        gen = TraceGenerator()
        engine = DifferentiableEngine(device="cpu")
        loss_fn = ExecutionLoss(
            engine=engine,
            trace_weight=1.0,
            device="cpu",
        )

        sample = gen.trace_code(
            code="x = a\nx = x + 3",
            inputs={"a": 5},
            arg_names=["a"],
            output_var="x",
        )

        inputs = {}
        for var_name, val in sample.inputs.items():
            reg = sample.parse_result.variable_map.get(var_name)
            if reg is not None:
                inputs[reg] = float(val)

        soft_prog = sample.parse_result.to_soft_program()

        result = loss_fn.compute_soft(
            soft_prog,
            inputs=inputs,
            expected=sample.final_expected,
            expected_trace=sample.trace_as_tuples,
            temperature=1.0,
        )

        result.total_loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in soft_prog.parameters()
        )
        assert has_grad, "No gradients from trace loss"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
