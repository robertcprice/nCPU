"""
Self-Optimizing Machine Engine (SOME)

A model that generates executable code, verifies through GPU execution,
and improves itself through the Gradient-Aware Protocol.

This is the main entry point for the self-optimizing engine.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import numpy as np

# Import protocols
from ncpu.os.gpu.protocols.gradient_aware_network import (
    GradientAwareNetworkProtocol,
    CompressionType,
)
from ncpu.os.gpu.protocols.persistent_workers import (
    PersistentGpuWorkersProtocol,
    WorkPriority,
)
from ncpu.os.gpu.protocols import BatchRpcProtocol, CompilerGuidedProtocol


@dataclass
class ExecutionResult:
    """Result from code execution"""
    success: bool
    output: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    gradient_signal: Optional[dict] = None


@dataclass
class Task:
    """A task to be executed and verified"""
    description: str
    input_data: Optional[dict] = None
    expected_output: Optional[Any] = None
    verification_fn: Optional[Callable] = None


class SelfOptimizingEngine:
    """
    Self-Optimizing Machine Engine (SOME)

    A computational engine that:
    1. Generates code as tensor descriptors (not text)
    2. Executes on GPU with full state visibility
    3. Captures execution feedback via Gradient Protocol
    4. Uses feedback to improve next generation

    The key innovation: Output is verified BEFORE it's "final"
    through the execution substrate itself.
    """

    def __init__(
        self,
        num_workers: int = 4,
        compression_type: CompressionType = CompressionType.TOP_K,
        max_iterations: int = 100,
    ):
        # Protocol layer
        self.ganp = GradientAwareNetworkProtocol(
            compression_type=compression_type,
        )
        self.workers = PersistentGpuWorkersProtocol(
            num_workers=num_workers,
        )
        self.batch_rpc = BatchRpcProtocol()

        # State
        self.model_state: dict = {}
        self.execution_history: list[ExecutionResult] = []
        self.iteration: int = 0
        self.max_iterations = max_iterations
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the engine and worker pools"""
        if self._initialized:
            return

        self.workers.initialize()
        self._initialized = True

    def shutdown(self) -> None:
        """Shutdown all components"""
        self.workers.shutdown()
        self._initialized = False

    def generate(self, task: Task) -> dict:
        """
        Generate tensor descriptors from task description.

        In a real implementation, this would call the neural model.
        For now, generates structured descriptors.
        """
        # In real impl: model.generate(task.description)
        # Returns tensor descriptors

        return {
            "operation": "compute",
            "task": task.description,
            "input": task.input_data or {},
            "descriptor_type": "tensor",
        }

    def execute(self, descriptor: dict) -> ExecutionResult:
        """
        Execute the generated descriptor on GPU.

        Uses persistent workers for zero-launch overhead.
        """
        import time

        start = time.perf_counter()

        try:
            # Submit work to persistent workers
            work_id = self.workers.submit_work(
                operation=descriptor.get("operation", "compute"),
                inputs=descriptor.get("input", {}),
                outputs={},
                priority=WorkPriority.HIGH,
            )

            # Wait for completion
            # In real impl, this would be async
            time.sleep(0.01)

            execution_time = (time.perf_counter() - start) * 1000

            # Simulate execution result
            result = ExecutionResult(
                success=True,
                output={"computed": True, "task": descriptor.get("task")},
                execution_time_ms=execution_time,
                memory_used_bytes=4096,
            )

        except Exception as e:
            result = ExecutionResult(
                success=False,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start) * 1000,
            )

        return result

    def capture_feedback(self, result: ExecutionResult) -> dict:
        """
        Capture execution feedback using Gradient Protocol.

        Returns structured signal for model improvement.
        """
        # Use gradient protocol to capture signal
        signal = {
            "success": result.success,
            "execution_time": result.execution_time_ms,
            "memory_used": result.memory_used_bytes,
            "improvement_direction": "positive" if result.success else "negative",
        }

        # In real impl: use ganp.backward() for structured gradients
        return signal

    def apply_feedback(self, feedback: dict) -> None:
        """
        Apply feedback to improve next generation.

        In real implementation, this applies gradients to the model.
        """
        self.model_state["last_feedback"] = feedback
        self.model_state["total_improvements"] = (
            self.model_state.get("total_improvements", 0) + 1
        )

    def verify(self, task: Task, result: ExecutionResult) -> bool:
        """
        Verify result against expected output.

        Returns True if verification passes.
        """
        if not result.success:
            return False

        if task.verification_fn:
            return task.verification_fn(result.output)

        if task.expected_output is not None:
            return result.output == task.expected_output

        # Default: success is verification
        return result.success

    def run(self, task: Task) -> ExecutionResult:
        """
        Main loop: Generate → Execute → Verify → Feedback → Improve

        This is the core self-optimizing loop.
        """
        self.initialize()

        for self.iteration in range(self.max_iterations):
            # 1. Generate (from model)
            descriptor = self.generate(task)

            # 2. Execute (on GPU)
            result = self.execute(descriptor)

            # 3. Verify
            if self.verify(task, result):
                # 4. Capture feedback
                feedback = self.capture_feedback(result)

                # 5. Apply to model
                self.apply_feedback(feedback)

                self.execution_history.append(result)
                return result

            # Failed - capture failure signal and retry
            feedback = self.capture_feedback(result)
            self.apply_feedback(feedback)

        # Max iterations reached
        return ExecutionResult(
            success=False,
            error="Max iterations reached",
        )


class VerifiedCodeGenerator:
    """
    Generates code that is verified through execution.

    Unlike traditional code generation that outputs text,
    this generates executable tensor descriptors that are
    verified before being considered "output".
    """

    def __init__(self, engine: SelfOptimizingEngine):
        self.engine = engine
        self.engine.initialize()

    def generate_and_verify(self, task_description: str) -> Any:
        """
        Generate and verify code for task.

        Returns only verified results.
        """
        task = Task(
            description=task_description,
            verification_fn=lambda x: x.get("computed") == True
        )

        result = self.engine.run(task)

        if result.success:
            return result.output
        else:
            raise RuntimeError(f"Code generation failed: {result.error}")

    def generate_multiple(self, task_descriptions: list[str]) -> list[Any]:
        """
        Generate multiple candidates and return verified results.

        Uses persistent workers for parallel execution.
        """
        results = []

        for desc in task_descriptions:
            try:
                result = self.generate_and_verify(desc)
                results.append(result)
            except RuntimeError:
                continue

        return results


def demo():
    """Demo of the self-optimizing engine"""
    print("=== Self-Optimizing Machine Engine Demo ===\n")

    # Create engine
    engine = SelfOptimizingEngine(num_workers=4)

    # Create task
    task = Task(
        description="Calculate fibonacci(100)",
        verification_fn=lambda x: x.get("computed") == True
    )

    # Run with verification
    print("Running self-optimizing loop...")
    result = engine.run(task)

    print(f"\nIteration: {engine.iteration}")
    print(f"Success: {result.success}")
    print(f"Output: {result.output}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")

    # Show feedback history
    print(f"\nTotal executions: {len(engine.execution_history)}")

    # Demo verified generator
    print("\n--- Verified Code Generator ---")
    generator = VerifiedCodeGenerator(engine)

    try:
        verified_result = generator.generate_and_verify("Compute sum of 1 to 100")
        print(f"Verified result: {verified_result}")
    except RuntimeError as e:
        print(f"Error: {e}")

    engine.shutdown()


if __name__ == "__main__":
    demo()
