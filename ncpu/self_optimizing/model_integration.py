"""
Model Integration Layer

Connects the self-optimizing engine to neural network models.
This module provides the interface between traditional ML models
and the GPU-native protocols.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import numpy as np

from ncpu.self_optimizing.gradient_feedback import (
    GradientFeedbackSystem,
    ExecutionSignal,
    ImprovementDirection,
)


@dataclass
class TensorDescriptor:
    """
    A tensor descriptor representing executable code.

    Instead of text, the model outputs these descriptors
    which are executed directly on the GPU.
    """
    operation: str  # e.g., "matmul", "loop", "function_call"
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    shape: tuple = field(default_factory=tuple)
    dtype: str = "float32"
    metadata: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        """Serialize to bytes for GPU"""
        import json
        data = {
            "operation": self.operation,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "metadata": self.metadata,
        }
        return json.dumps(data).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> 'TensorDescriptor':
        """Deserialize from bytes"""
        import json
        parsed = json.loads(data.decode())
        return cls(**parsed)


class ModelInterface:
    """
    Interface between neural models and the self-optimizing engine.

    Provides methods for:
    - Generating tensor descriptors from model outputs
    - Converting execution feedback to model gradients
    - Managing the generation → execution → feedback loop
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        gradient_feedback: Optional[GradientFeedbackSystem] = None,
    ):
        self.model = model
        self.gfs = gradient_feedback or GradientFeedbackSystem()
        self.generation_history: list[TensorDescriptor] = []

    def generate_descriptors(
        self,
        input_text: str,
        num_candidates: int = 1,
    ) -> list[TensorDescriptor]:
        """
        Generate tensor descriptor candidates from input.

        In a real implementation, this would call the model.
        """
        # In real impl: model.generate(input_text)
        # Returns list of tensor descriptors

        candidates = []
        for i in range(num_candidates):
            desc = TensorDescriptor(
                operation="compute",
                inputs={"query": input_text},
                outputs={"result": None},
                shape=(1,),
                dtype="float32",
                metadata={"candidate_id": i}
            )
            candidates.append(desc)

        self.generation_history.extend(candidates)
        return candidates

    def apply_gradient(
        self,
        feedback_signal: ExecutionSignal,
    ) -> None:
        """
        Apply feedback signal to influence next generation.

        In a real implementation, this would update model weights
        or adjust generation parameters.
        """
        direction = feedback_signal.improvement_direction

        # In real implementation:
        # - POSITIVE: Increase probability of similar outputs
        # - NEGATIVE: Decrease probability of similar outputs
        # - REFINEMENT: Slight adjustments

        if direction == ImprovementDirection.POSITIVE:
            print(f"Applying positive gradient - reinforcing pattern")
            self.gfs.improvement_history.append({
                "direction": "positive",
                "signal": feedback_signal.feedback_type.value,
            })
        elif direction == ImprovementDirection.NEGATIVE:
            print(f"Applying negative gradient - avoiding pattern")
            self.gfs.improvement_history.append({
                "direction": "negative",
                "signal": feedback_signal.feedback_type.value,
            })
        elif direction == ImprovementDirection.REFINEMENT:
            print(f"Applying refinement gradient - tuning pattern")
            self.gfs.improvement_history.append({
                "direction": "refinement",
                "signal": feedback_signal.feedback_type.value,
            })

    def get_generation_params(self) -> dict:
        """
        Get current generation parameters that may have been
        modified by feedback.
        """
        return {
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.95,
        }


class ExecutionVerifiedModel:
    """
    A model wrapper that ensures all outputs are verified
    through execution before being returned.

    This is the key innovation: the model doesn't output
    text - it outputs tensor descriptors that are executed,
    and only successful executions become "outputs".
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        max_retries: int = 3,
    ):
        self.interface = ModelInterface(model=model)
        self.max_retries = max_retries

    def predict(
        self,
        input_data: Any,
        verify_fn: Optional[Callable] = None,
    ) -> Any:
        """
        Generate, execute, verify, return.

        Unlike traditional models that return immediately,
        this waits for execution verification.
        """
        for attempt in range(self.max_retries):
            # 1. Generate candidates
            candidates = self.interface.generate_descriptors(
                str(input_data),
                num_candidates=1,
            )

            # 2. Execute (would use GPU in real impl)
            result = self._execute_candidate(candidates[0])

            # 3. Verify
            if verify_fn:
                is_valid = verify_fn(result)
            else:
                is_valid = result is not None

            if is_valid:
                # 4. Apply positive feedback
                signal = self.interface.gfs.capture_execution(result)
                self.interface.apply_gradient(signal)
                return result

            # 5. Failed - capture negative feedback
            signal = self.interface.gfs.capture_execution(
                result,
                expected_result=None,  # Unknown
            )
            self.interface.apply_gradient(signal)

        # All retries failed
        raise RuntimeError(f"Failed after {self.max_retries} attempts")

    def _execute_candidate(self, descriptor: TensorDescriptor) -> Any:
        """
        Execute a candidate descriptor.

        In real implementation, this would use the GPU protocols.
        """
        # Simulate execution
        return {"executed": True, "operation": descriptor.operation}


class MultiCandidateExplorer:
    """
    Generates multiple candidates and explores them in parallel
    using persistent workers.
    """

    def __init__(self, model: Optional[Any] = None, num_workers: int = 8):
        self.interface = ModelInterface(model=model)
        self.num_workers = num_workers
        self.results: list[Any] = []

    def explore(
        self,
        input_data: str,
        num_candidates: int = 8,
    ) -> list[Any]:
        """
        Generate and explore multiple candidates in parallel.

        Uses persistent workers for zero-launch overhead.
        """
        # Generate candidates
        candidates = self.interface.generate_descriptors(
            input_data,
            num_candidates=num_candidates,
        )

        # Execute all candidates (in real impl: parallel on GPU)
        results = []
        for candidate in candidates:
            try:
                result = self._execute(candidate)
                results.append(result)
            except Exception as e:
                # Capture failure
                signal = self.interface.gfs.capture_from_error(
                    e,
                    {"candidate": candidate.operation}
                )
                self.interface.apply_gradient(signal)

        self.results = results
        return results

    def select_best(self, results: list[Any], metric_fn: Optional[Callable] = None) -> Any:
        """Select best result based on metric"""
        if not results:
            raise ValueError("No results to select from")

        if metric_fn:
            scored = [(metric_fn(r), r) for r in results]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1]
        else:
            # Return first successful result
            for r in results:
                if r is not None:
                    return r
            return results[0]

    def _execute(self, descriptor: TensorDescriptor) -> Any:
        """Execute descriptor"""
        return {"output": descriptor.operation, "verified": True}


def demo():
    """Demo of model integration"""
    print("=== Model Integration Demo ===\n")

    # Create interface
    interface = ModelInterface()

    # Generate candidates
    candidates = interface.generate_descriptors("Calculate fibonacci", num_candidates=3)
    print(f"Generated {len(candidates)} candidates")

    for i, c in enumerate(candidates):
        print(f"  Candidate {i}: {c.operation}")

    # Simulate feedback
    signal = ExecutionSignal(
        feedback_type=type('FeedbackType', (), {'value': 'success'})(),
        improvement_direction=ImprovementDirection.POSITIVE,
    )
    interface.apply_gradient(signal)

    print(f"\nGeneration params: {interface.get_generation_params()}")

    # Demo verified model
    print("\n--- Execution Verified Model ---")
    evm = ExecutionVerifiedModel()

    try:
        result = evm.predict("test input", verify_fn=lambda x: x is not None)
        print(f"Verified result: {result}")
    except RuntimeError as e:
        print(f"Error: {e}")

    # Demo multi-candidate explorer
    print("\n--- Multi-Candidate Explorer ---")
    explorer = MultiCandidateExplorer(num_workers=4)
    results = explorer.explore("optimize this code")
    print(f"Explored {len(results)} candidates")

    best = explorer.select_best(results)
    print(f"Best result: {best}")


if __name__ == "__main__":
    demo()
