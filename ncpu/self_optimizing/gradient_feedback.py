"""
Gradient Feedback System

The core innovation of the self-optimizing engine: structured feedback
from execution that the model can use to improve.

This implements the "Gradient Protocol" as a feedback mechanism,
not just for ML training but for general code improvement.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
import numpy as np


class FeedbackType(Enum):
    """Types of feedback from execution"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    MEMORY_ERROR = "memory_error"
    CORRECTNESS_ERROR = "correctness_error"
    PERFORMANCE_WARNING = "performance_warning"


class ImprovementDirection(Enum):
    """Direction of improvement signal"""
    POSITIVE = "positive"  # This approach works, continue
    NEGATIVE = "negative"  # This approach failed, try different
    NEUTRAL = "neutral"    # No clear signal
    REFINEMENT = "refinement"  # Works but can be improved


@dataclass
class ExecutionSignal:
    """
    Structured signal from execution that represents
    "how to improve" - not just loss values.

    This is the key innovation: feedback is actionable
    and tied to actual execution behavior.
    """
    feedback_type: FeedbackType
    improvement_direction: ImprovementDirection

    # Detailed metrics
    execution_time_ms: float = 0.0
    memory_used_bytes: int = 0
    cpu_cycles: int = 0
    cache_hits: int = 0
    cache_misses: int = 0

    # Correctness
    expected_vs_actual: Optional[tuple] = None
    correctness_score: float = 0.0  # 0.0 to 1.0

    # Pattern signals
    similar_to_successful: bool = False
    similar_to_failed: bool = False
    pattern_id: Optional[str] = None

    # Raw data for model
    raw_metrics: dict = field(default_factory=dict)

    def to_gradient(self) -> np.ndarray:
        """
        Convert to gradient-like representation for model.

        This creates a fixed-size vector that models can
        use to understand execution behavior.
        """
        # Cache efficiency computation
        cache_total = self.cache_hits + self.cache_misses
        cache_eff = self.cache_hits / cache_total if cache_total > 0 else 0.0

        gradient = np.array([
            # Feedback type (one-hot)
            1.0 if self.feedback_type == FeedbackType.SUCCESS else 0.0,
            1.0 if self.feedback_type == FeedbackType.FAILURE else 0.0,
            1.0 if self.feedback_type == FeedbackType.TIMEOUT else 0.0,

            # Improvement direction
            1.0 if self.improvement_direction == ImprovementDirection.POSITIVE else 0.0,
            1.0 if self.improvement_direction == ImprovementDirection.NEGATIVE else 0.0,
            1.0 if self.improvement_direction == ImprovementDirection.REFINEMENT else 0.0,

            # Normalized metrics
            min(self.execution_time_ms / 1000.0, 1.0),  # Normalized time
            min(self.memory_used_bytes / (1024 * 1024), 1.0),  # Normalized memory
            self.correctness_score,

            # Cache efficiency
            cache_eff,
        ], dtype=np.float32)

        return gradient


@dataclass
class CodePattern:
    """
    Represents a pattern of code that the system has learned.

    Patterns can be:
    - Successful (use more)
    - Failed (avoid)
    - Neutral (can modify)
    """
    pattern_id: str
    pattern_type: str  # e.g., "loop", "recursion", "iteration"
    code_hash: str
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5

    def get_improvement_direction(self) -> ImprovementDirection:
        rate = self.success_rate
        if rate > 0.8:
            return ImprovementDirection.POSITIVE
        elif rate < 0.2:
            return ImprovementDirection.NEGATIVE
        else:
            return ImprovementDirection.REFINEMENT


class GradientFeedbackSystem:
    """
    System that captures execution feedback and converts it
    to structured gradients for model improvement.
    """

    def __init__(self):
        self.signals: list[ExecutionSignal] = []
        self.patterns: dict[str, CodePattern] = {}
        self.improvement_history: list[dict] = []

    def capture_execution(
        self,
        execution_result: Any,
        expected_result: Optional[Any] = None,
    ) -> ExecutionSignal:
        """
        Capture feedback from an execution result.

        Analyzes what happened and creates a structured signal.
        """
        # Analyze correctness
        expected_vs_actual = None
        if expected_result is not None:
            if execution_result == expected_result:
                feedback_type = FeedbackType.SUCCESS
                correctness_score = 1.0
            else:
                feedback_type = FeedbackType.CORRECTNESS_ERROR
                correctness_score = 0.0
                expected_vs_actual = (expected_result, execution_result)
        else:
            # No expected result - assume success if no error
            if execution_result is not None:
                feedback_type = FeedbackType.SUCCESS
                correctness_score = 1.0
            else:
                feedback_type = FeedbackType.FAILURE
                correctness_score = 0.0

        # Determine improvement direction
        if feedback_type == FeedbackType.SUCCESS:
            if correctness_score > 0.9:
                direction = ImprovementDirection.POSITIVE
            else:
                direction = ImprovementDirection.REFINEMENT
        else:
            direction = ImprovementDirection.NEGATIVE

        # Create signal
        signal = ExecutionSignal(
            feedback_type=feedback_type,
            improvement_direction=direction,
            correctness_score=correctness_score,
            expected_vs_actual=expected_vs_actual,
        )

        self.signals.append(signal)
        return signal

    def capture_from_error(
        self,
        error: Exception,
        context: dict,
    ) -> ExecutionSignal:
        """Capture feedback from an error"""
        error_type = type(error).__name__

        if "timeout" in str(error).lower():
            feedback_type = FeedbackType.TIMEOUT
        elif "memory" in str(error).lower():
            feedback_type = FeedbackType.MEMORY_ERROR
        else:
            feedback_type = FeedbackType.FAILURE

        signal = ExecutionSignal(
            feedback_type=feedback_type,
            improvement_direction=ImprovementDirection.NEGATIVE,
            raw_metrics={**context, "error": str(error)},
        )

        self.signals.append(signal)
        return signal

    def record_pattern(
        self,
        code_hash: str,
        pattern_type: str,
        success: bool,
    ) -> None:
        """Record a code pattern and its outcome"""
        if code_hash not in self.patterns:
            self.patterns[code_hash] = CodePattern(
                pattern_id=code_hash[:8],
                pattern_type=pattern_type,
                code_hash=code_hash,
            )

        if success:
            self.patterns[code_hash].success_count += 1
        else:
            self.patterns[code_hash].failure_count += 1

    def get_pattern_signal(self, code_hash: str) -> Optional[ExecutionSignal]:
        """Get feedback signal based on pattern history"""
        if code_hash not in self.patterns:
            return None

        pattern = self.patterns[code_hash]
        direction = pattern.get_improvement_direction()

        return ExecutionSignal(
            feedback_type=FeedbackType.SUCCESS if direction == ImprovementDirection.POSITIVE else FeedbackType.FAILURE,
            improvement_direction=direction,
            pattern_id=pattern.pattern_id,
            similar_to_successful=direction == ImprovementDirection.POSITIVE,
            similar_to_failed=direction == ImprovementDirection.NEGATIVE,
        )

    def compute_gradient(self) -> np.ndarray:
        """
        Compute aggregate gradient from all signals.

        Returns a vector that can be applied to model weights.
        """
        if not self.signals:
            return np.zeros(10, dtype=np.float32)

        # Average all gradients
        gradients = [s.to_gradient() for s in self.signals]
        avg_gradient = np.mean(gradients, axis=0)

        return avg_gradient

    def get_improvement_suggestion(self) -> str:
        """
        Get human-readable improvement suggestion based on feedback.
        """
        recent = self.signals[-5:]  # Last 5 signals

        if not recent:
            return "No feedback yet"

        # Analyze patterns
        failures = [s for s in recent if s.feedback_type != FeedbackType.SUCCESS]
        timeouts = [s for s in recent if s.feedback_type == FeedbackType.TIMEOUT]

        if len(failures) > len(recent) * 0.5:
            return "High failure rate - consider simplifying approach"
        elif timeouts:
            return "Timeouts detected - optimize for performance"
        else:
            return "Generally successful - minor refinements possible"

    def clear_history(self) -> None:
        """Clear feedback history (for new tasks)"""
        self.signals.clear()
        self.improvement_history.clear()


def demo():
    """Demo of gradient feedback system"""
    print("=== Gradient Feedback System Demo ===\n")

    gfs = GradientFeedbackSystem()

    # Simulate successful execution
    signal1 = gfs.capture_execution(42, expected_result=42)
    print(f"Signal 1: {signal1.feedback_type.value}, direction: {signal1.improvement_direction.value}")
    print(f"  Gradient: {signal1.to_gradient()[:4]}...")

    # Simulate failed execution
    signal2 = gfs.capture_execution(100, expected_result=42)
    print(f"\nSignal 2: {signal2.feedback_type.value}, direction: {signal2.improvement_direction.value}")

    # Record pattern
    gfs.record_pattern("abc123", "recursion", success=True)
    gfs.record_pattern("abc123", "recursion", success=True)
    gfs.record_pattern("abc123", "recursion", success=False)  # 66% success

    # Get pattern signal
    pattern_signal = gfs.get_pattern_signal("abc123")
    if pattern_signal:
        print(f"\nPattern signal: {pattern_signal.improvement_direction.value}")

    # Compute gradient
    gradient = gfs.compute_gradient()
    print(f"\nAggregate gradient: {gradient[:4]}...")

    # Get suggestion
    suggestion = gfs.get_improvement_suggestion()
    print(f"\nImprovement suggestion: {suggestion}")


if __name__ == "__main__":
    demo()
