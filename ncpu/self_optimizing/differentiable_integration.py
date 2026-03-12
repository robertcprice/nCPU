"""
Differentiable Execution Integration for Self-Optimizing Engine

This module integrates the differentiable CPU (DiffOoO) with the
self-optimizing machine engine (SOME) to enable gradient-based
code optimization.

The key innovation: instead of just capturing execution feedback,
we can now backprop through execution to optimize code generation.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Callable
import numpy as np


class DifferentiableExecutor:
    """
    Wrapper around the differentiable CPU that provides
    gradient-aware execution for the self-optimizing engine.
    """

    def __init__(
        self,
        memory_size: int = 4 * 1024 * 1024,
        cycles_per_batch: int = 10_000_000,
    ):
        """Initialize the differentiable executor."""
        self.memory_size = memory_size
        self.cycles_per_batch = cycles_per_batch
        self.cpu = None
        self._initialized = False

    def initialize(self):
        """Lazy initialization of the differentiable CPU."""
        if self._initialized:
            return

        try:
            from ncpu_metal import DiffOoOCPU
            self.cpu = DiffOoOCPU(
                memory_size=self.memory_size,
                cycles_per_batch=self.cycles_per_batch,
            )
            self._initialized = True
            print("[DiffSOME] Differentiable CPU initialized")
        except ImportError as e:
            print(f"[DiffSOME] Warning: DiffOoO not available: {e}")
            self._initialized = False

    def execute_with_gradients(
        self,
        program: bytes,
        address: int = 0x1000,
        input_data: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Execute program and return results with gradient information.

        Returns:
            Dictionary with:
            - output: execution output
            - gradients: gradient tensor for optimization
            - weights: current learnable weights
            - metrics: execution metrics
        """
        if not self._initialized:
            self.initialize()

        if self.cpu is None:
            return {
                "success": False,
                "error": "Differentiable CPU not available",
                "gradients": np.array([]),
                "weights": np.array([]),
            }

        # Load program
        self.cpu.load_program(list(program), address)

        # Set input if provided
        if input_data is not None:
            self.cpu.set_input_data(input_data.tolist())

        # Run execution
        result = self.cpu.execute(max_batches=100, timeout_seconds=10.0)

        # Get gradients for optimization
        gradients = self.cpu.get_gradients()
        weights = self.cpu.get_weights()

        return {
            "success": result.signal in (1, 2),  # 1 = success, 2 = complete
            "signal": result.signal,
            "cycles": result.total_cycles,
            "elapsed_ms": result.elapsed_seconds * 1000,
            "gradients": np.array(gradients),
            "weights": np.array(weights),
            "batch_count": result.batch_count,
            "speculative_branches": result.speculative_branches,
        }

    def optimize_weights(
        self,
        gradients: np.ndarray,
        learning_rate: float = 0.01,
    ) -> np.ndarray:
        """
        Apply gradient descent to weights.

        This is the core of differentiable optimization:
        use execution gradients to improve code generation.
        """
        if self.cpu is None:
            return np.array([])

        current_weights = self.cpu.get_weights()
        new_weights = [
            w - learning_rate * g
            for w, g in zip(current_weights, gradients)
        ]

        # Apply new weights
        self.cpu.set_weights(new_weights)

        return np.array(new_weights)

    def reset(self):
        """Reset CPU state."""
        if self.cpu is not None:
            self.cpu.reset()
            self.cpu.zero_gradients()


@dataclass
class GradientOptimizationConfig:
    """Configuration for gradient-based optimization."""

    learning_rate: float = 0.01
    momentum: float = 0.9
    temperature: float = 1.0
    num_iterations: int = 10
    gradient_clip: float = 1.0


class GradientAwareOptimizer:
    """
    Optimizer that uses differentiable execution gradients
    to improve code generation.
    """

    def __init__(self, config: GradientOptimizationConfig = None):
        self.config = config or GradientOptimizationConfig()
        self.executor = DifferentiableExecutor()
        self.history: List[Dict] = []

    def optimize(
        self,
        program_generator: Callable,
        verification_fn: Callable[[Any], bool],
        max_iterations: int = None,
    ) -> Dict[str, Any]:
        """
        Optimize code using differentiable execution.

        Args:
            program_generator: Function that generates candidate programs
            verification_fn: Function that verifies execution results
            max_iterations: Maximum optimization iterations

        Returns:
            Best program and optimization history
        """
        max_iterations = max_iterations or self.config.num_iterations
        best_program = None
        best_score = -float("inf")

        for iteration in range(max_iterations):
            # Generate candidate program
            program = program_generator(iteration)

            # Execute with gradient tracking
            result = self.executor.execute_with_gradients(program)

            if not result["success"]:
                # Failed execution - use negative gradient signal
                score = -1.0
            else:
                # Check if result is correct
                is_correct = verification_fn(result)
                score = 1.0 if is_correct else -0.5

            # Track history
            self.history.append({
                "iteration": iteration,
                "score": score,
                "gradients": result.get("gradients", np.array([])),
                "weights": result.get("weights", np.array([])),
                "cycles": result.get("cycles", 0),
            })

            # Update best
            if score > best_score:
                best_score = score
                best_program = program

            # Apply gradient optimization if we have gradients
            if "gradients" in result and len(result["gradients"]) > 0:
                # Clip gradients
                grads = result["gradients"]
                if self.config.gradient_clip > 0:
                    grads = np.clip(
                        grads,
                        -self.config.gradient_clip,
                        self.config.gradient_clip,
                    )

                # Apply gradient descent
                self.executor.optimize_weights(
                    grads,
                    learning_rate=self.config.learning_rate,
                )

            print(f"[DiffSOME] Iteration {iteration}: score={score:.3f}, "
                  f"cycles={result.get('cycles', 0)}")

        return {
            "best_program": best_program,
            "best_score": best_score,
            "history": self.history,
            "final_weights": self.executor.cpu.get_weights() if self.executor.cpu else [],
        }


class NeuralScheduleOptimizer:
    """
    Optimize the neural scheduler weights using differentiable execution.

    Uses execution feedback to learn optimal scheduling policies.
    """

    def __init__(self):
        self.weights = {
            "priority": 1.0,
            "cpu_burst": 0.8,
            "io_wait": 0.5,
            "age": 0.3,
        }
        self.history: List[Dict] = []

    def compute_score(
        self,
        priority: float,
        cpu_burst: float,
        io_wait: float,
        age: float,
    ) -> float:
        """Compute neural schedule score with current weights."""
        return (
            self.weights["priority"] * priority +
            self.weights["cpu_burst"] * cpu_burst +
            self.weights["io_wait"] * io_wait +
            self.weights["age"] * age
        )

    def update_weights(
        self,
        gradient: Dict[str, float],
        learning_rate: float = 0.1,
    ):
        """Update weights based on gradient feedback."""
        for key in self.weights:
            if key in gradient:
                self.weights[key] += learning_rate * gradient[key]

        self.history.append({
            "weights": self.weights.copy(),
            "gradient": gradient.copy(),
        })

    def get_weights(self) -> Dict[str, float]:
        """Get current scheduling weights."""
        return self.weights.copy()


def create_differentiable_feedback(
    execution_result: Dict[str, Any],
) -> np.ndarray:
    """
    Convert differentiable execution result to gradient feedback
    compatible with the self-optimizing engine.

    This bridges the gap between differentiable execution and SOME.
    """
    if "gradients" in execution_result:
        return execution_result["gradients"]

    # Construct gradient from metrics
    gradient = np.array([
        execution_result.get("cycles", 0) / 1_000_000,  # Normalize cycles
        execution_result.get("batch_count", 0) / 100,   # Normalize batches
        1.0 if execution_result.get("success", False) else -1.0,
        execution_result.get("speculative_branches", 0) / 1000,
    ])

    return gradient


# Integration with existing gradient feedback system
def integrate_with_gradient_feedback(
    execution_result: Dict[str, Any],
    feedback_system,
) -> Any:
    """
    Integrate differentiable execution with the GradientFeedbackSystem.

    This allows the self-optimizing engine to use real gradients
    from execution rather than just heuristic feedback.
    """
    from ncpu.self_optimizing.gradient_feedback import (
        ExecutionSignal,
        FeedbackType,
        ImprovementDirection,
    )

    # Convert to ExecutionSignal
    signal = ExecutionSignal(
        feedback_type=FeedbackType.SUCCESS if execution_result.get("success")
                     else FeedbackType.FAILURE,
        improvement_direction=ImprovementDirection.POSITIVE
                             if execution_result.get("success")
                             else ImprovementDirection.NEGATIVE,
        execution_time_ms=execution_result.get("elapsed_ms", 0),
        cpu_cycles=execution_result.get("cycles", 0),
        correctness_score=1.0 if execution_result.get("success") else 0.0,
    )

    return signal
