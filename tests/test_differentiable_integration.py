"""
Tests for differentiable integration with self-optimizing engine.
"""

import pytest
import numpy as np
from ncpu.self_optimizing.differentiable_integration import (
    DifferentiableExecutor,
    GradientAwareOptimizer,
    GradientOptimizationConfig,
    NeuralScheduleOptimizer,
    create_differentiable_feedback,
)


class TestDifferentiableExecutor:
    """Test the differentiable executor wrapper."""

    def test_initialization(self):
        """Test that executor initializes correctly."""
        executor = DifferentiableExecutor(
            memory_size=1024 * 1024,
            cycles_per_batch=1_000_000,
        )
        # Lazy initialization - should not fail
        assert executor.memory_size == 1024 * 1024
        assert executor.cycles_per_batch == 1_000_000

    def test_lazy_init(self):
        """Test lazy initialization."""
        executor = DifferentiableExecutor()
        # Should not fail even without GPU
        assert not executor._initialized

    def test_execute_with_gradients(self):
        """Test execute with GPU returns proper structure."""
        executor = DifferentiableExecutor()
        result = executor.execute_with_gradients(b"\x00" * 100)

        # Should have all expected fields
        assert "success" in result
        assert "gradients" in result
        assert "weights" in result
        assert "cycles" in result


class TestGradientAwareOptimizer:
    """Test gradient-aware optimization."""

    def test_config_defaults(self):
        """Test default configuration."""
        config = GradientOptimizationConfig()
        assert config.learning_rate == 0.01
        assert config.momentum == 0.9
        assert config.temperature == 1.0
        assert config.num_iterations == 10

    def test_config_custom(self):
        """Test custom configuration."""
        config = GradientOptimizationConfig(
            learning_rate=0.05,
            num_iterations=5,
        )
        assert config.learning_rate == 0.05
        assert config.num_iterations == 5

    def test_optimizer_initialization(self):
        """Test optimizer initializes correctly."""
        optimizer = GradientAwareOptimizer()
        assert optimizer.config.learning_rate == 0.01
        assert len(optimizer.history) == 0


class TestNeuralScheduleOptimizer:
    """Test neural scheduling optimizer."""

    def test_initial_weights(self):
        """Test default weights."""
        optimizer = NeuralScheduleOptimizer()
        weights = optimizer.get_weights()

        assert "priority" in weights
        assert "cpu_burst" in weights
        assert "io_wait" in weights
        assert "age" in weights

    def test_compute_score(self):
        """Test score computation."""
        optimizer = NeuralScheduleOptimizer()

        # Test with default weights
        score = optimizer.compute_score(
            priority=0.5,
            cpu_burst=0.5,
            io_wait=0.5,
            age=0.5,
        )

        # With default weights: 1.0*0.5 + 0.8*0.5 + 0.5*0.5 + 0.3*0.5 = 1.0
        expected = 0.5 * (1.0 + 0.8 + 0.5 + 0.3)
        assert abs(score - expected) < 0.001

    def test_update_weights(self):
        """Test weight update."""
        optimizer = NeuralScheduleOptimizer()

        initial = optimizer.get_weights()

        # Update with gradient
        optimizer.update_weights({
            "priority": 0.1,
            "cpu_burst": -0.1,
            "io_wait": 0.05,
            "age": 0.0,
        })

        updated = optimizer.get_weights()

        # Weights should have changed
        assert updated["priority"] != initial["priority"]
        assert updated["cpu_burst"] != initial["cpu_burst"]

        # History should have recorded
        assert len(optimizer.history) == 1


class TestDifferentiableFeedback:
    """Test feedback conversion."""

    def test_create_feedback_from_result(self):
        """Test creating feedback from execution result."""
        result = {
            "success": True,
            "cycles": 500000,
            "batch_count": 50,
            "speculative_branches": 100,
        }

        gradient = create_differentiable_feedback(result)

        assert isinstance(gradient, np.ndarray)
        assert len(gradient) > 0

    def test_create_feedback_with_gradients(self):
        """Test feedback includes actual gradients when available."""
        result = {
            "success": True,
            "gradients": np.array([0.1, 0.2, 0.3, 0.4]),
        }

        gradient = create_differentiable_feedback(result)

        assert isinstance(gradient, np.ndarray)
        np.testing.assert_array_equal(gradient, result["gradients"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
