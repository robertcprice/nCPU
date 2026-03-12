"""
Autoresearch Agent with SOME Integration

Enhanced super-autoresearch agent that uses Self-Optimizing Machine Engine
capabilities for parallel experiments, gradient feedback, and self-improvement.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
import numpy as np
import time
import json
from pathlib import Path


@dataclass
class ExperimentCandidate:
    """A candidate modification to train.py"""
    candidate_id: int
    description: str
    code_diff: str  # Diff from current train.py
    expected_change: str  # What we expect this to improve
    priority: float = 1.0


@dataclass
class ExperimentResult:
    """Result from an experiment run"""
    candidate_id: int
    val_bpb: float
    training_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_M: float
    num_steps: int
    status: str  # "keep", "discard", "crash"
    error: Optional[str] = None


@dataclass
class GradientFeedback:
    """Gradient feedback from experiments"""
    candidate_id: int
    val_bpb: float
    improvement: bool
    gradient_signal: np.ndarray
    pattern_id: str


class AutoresearchSOMEAgent:
    """
    Super-AutoResearch agent enhanced with SOME capabilities.

    Key enhancements:
    - Parallel experiment candidates via persistent workers
    - Gradient feedback for self-improvement
    - SVM for efficient weight sharing
    - Learned pattern avoidance
    """

    def __init__(
        self,
        num_workers: int = 8,
        experiment_minutes: int = 5,
        preset: str = "small",
    ):
        self.num_workers = num_workers
        self.experiment_minutes = experiment_minutes
        self.preset = preset

        # Import SOME components
        from ncpu.os.gpu.protocols import (
            PersistentGpuWorkersProtocol,
            GradientAwareNetworkProtocol,
            SharedVirtualMemoryProtocol,
        )

        # Initialize SOME components
        self.workers = PersistentGpuWorkersProtocol(num_workers=num_workers)
        self.ganp = GradientAwareNetworkProtocol()
        self.svm = SharedVirtualMemoryProtocol(devices=[0])

        # State
        self.candidates: list[ExperimentCandidate] = []
        self.results: list[ExperimentResult] = []
        self.patterns: dict[str, GradientFeedback] = {}
        self.best_val_bpb: Optional[float] = None
        self._initialized = False

    def initialize(self):
        """Initialize the agent and worker pools"""
        if self._initialized:
            return

        self.workers.initialize()
        self._initialized = True

        print(f"AutoresearchSOME Agent initialized with {self.num_workers} workers")

    def shutdown(self):
        """Shutdown all components"""
        self.workers.shutdown()
        self._initialized = False

    def generate_candidates(
        self,
        base_train_py: str,
        num_candidates: int = 8,
    ) -> list[ExperimentCandidate]:
        """
        Generate candidate modifications to train.py.

        Uses gradient feedback from previous experiments to inform generation.
        """
        candidates = []

        # Get gradient feedback from history
        feedback_signal = self._get_gradient_signal()

        for i in range(num_candidates):
            # Use feedback to inform generation
            modification = self._generate_modification(
                base_train_py,
                feedback_signal,
                candidate_num=i
            )

            candidate = ExperimentCandidate(
                candidate_id=i,
                description=modification["description"],
                code_diff=modification["diff"],
                expected_change=modification["expected"],
                priority=modification.get("priority", 1.0),
            )
            candidates.append(candidate)

        self.candidates = candidates
        return candidates

    def run_parallel_experiments(
        self,
        candidates: list[ExperimentCandidate],
    ) -> list[ExperimentResult]:
        """
        Run multiple experiment candidates in parallel.

        Uses persistent workers for zero-launch overhead.
        """
        results = []

        # Submit all candidates to workers
        for candidate in candidates:
            self.workers.submit_work(
                operation="run_experiment",
                inputs={
                    "candidate_id": candidate.candidate_id,
                    "code_diff": candidate.code_diff,
                    "experiment_minutes": self.experiment_minutes,
                },
                outputs={},
                priority=WorkPriority.HIGH if candidate.priority > 1.0 else WorkPriority.NORMAL,
            )

        # Wait for experiments to complete
        # In real impl: async wait with progress tracking
        wait_time = self.experiment_minutes * 60 + 60  # buffer
        print(f"Running {len(candidates)} experiments in parallel...")
        time.sleep(min(wait_time, 60))  # Cap for demo

        # Get completed results
        completed = self.workers.get_completed()

        # Parse results
        for work_result in completed:
            result = self._parse_experiment_result(work_result)
            results.append(result)
            self.results.append(result)

            # Update best
            if result.status == "keep":
                if self.best_val_bpb is None or result.val_bpb < self.best_val_bpb:
                    self.best_val_bpb = result.val_bpb

        return results

    def apply_gradient_learning(
        self,
        results: list[ExperimentResult],
    ) -> None:
        """
        Learn from experiment results using gradient feedback.

        Creates gradient signals for successful vs failed experiments.
        """
        for result in results:
            # Determine if this was an improvement
            improvement = (
                self.best_val_bpb is not None
                and result.val_bpb <= self.best_val_bpb
            )

            # Create gradient signal
            gradient = np.array([
                1.0 if improvement else 0.0,  # Positive gradient
                1.0 if result.status == "crash" else 0.0,  # Crash signal
                result.val_bpb / 3.0,  # Normalized val_bpb
                result.training_seconds / 300.0,  # Normalized time
            ], dtype=np.float32)

            # Store pattern
            pattern_id = f"candidate_{result.candidate_id}"
            self.patterns[pattern_id] = GradientFeedback(
                candidate_id=result.candidate_id,
                val_bpb=result.val_bpb,
                improvement=improvement,
                gradient_signal=gradient,
                pattern_id=pattern_id,
            )

            # Use gradient protocol to capture learning
            self.ganp.compress_gradients({
                pattern_id: gradient
            })

    def get_best_modification(self) -> Optional[ExperimentCandidate]:
        """Get the best modification from completed experiments"""
        if not self.results:
            return None

        valid_results = [r for r in self.results if r.status == "keep"]
        if not valid_results:
            return None

        best = min(valid_results, key=lambda r: r.val_bpb)

        for candidate in self.candidates:
            if candidate.candidate_id == best.candidate_id:
                return candidate

        return None

    def _get_gradient_signal(self) -> Optional[np.ndarray]:
        """Get gradient signal from previous experiments"""
        if not self.patterns:
            return None

        # Combine recent patterns
        recent = list(self.patterns.values())[-4:]
        signals = [p.gradient_signal for p in recent]

        if signals:
            return np.mean(signals, axis=0)

        return None

    def _generate_modification(
        self,
        base_code: str,
        feedback: Optional[np.ndarray],
        candidate_num: int,
    ) -> dict:
        """
        Generate a single modification based on feedback.

        In real implementation, this would use an LLM.
        """
        # Common modification types
        modification_types = [
            ("increase_layers", "Increase number of layers"),
            ("increase_heads", "Increase number of attention heads"),
            ("increase_embed", "Increase embedding dimension"),
            ("add_checkpointing", "Add gradient checkpointing"),
            ("adjust_lr", "Adjust learning rate"),
            ("window_pattern", "Change window pattern"),
        ]

        mod_type = modification_types[candidate_num % len(modification_types)]

        return {
            "description": mod_type[1],
            "diff": f"# Modify {mod_type[0]}",
            "expected": f"Expected to improve val_bpb",
            "priority": 1.0 + (feedback.sum() if feedback is not None else 0),
        }

    def _parse_experiment_result(self, work_result) -> ExperimentResult:
        """Parse experiment result from worker"""
        # In real impl: parse actual output
        # For demo: simulate result
        return ExperimentResult(
            candidate_id=work_result.work_id if hasattr(work_result, 'work_id') else 0,
            val_bpb=np.random.uniform(1.8, 2.2),
            training_seconds=300.0,
            peak_vram_mb=2048.0,
            mfu_percent=0.25,
            total_tokens_M=10.5,
            num_steps=160,
            status="keep",
        )

    def run_research_loop(
        self,
        base_train_py: str,
        num_iterations: int = 10,
    ) -> dict:
        """
        Run the full autoresearch loop with SOME enhancements.

        Returns final results and learned patterns.
        """
        self.initialize()

        all_results = []

        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")

            # 1. Generate candidates
            candidates = self.generate_candidates(
                base_train_py,
                num_candidates=self.num_workers,
            )
            print(f"Generated {len(candidates)} candidates")

            # 2. Run parallel experiments
            results = self.run_parallel_experiments(candidates)
            print(f"Completed {len(results)} experiments")

            # 3. Apply gradient learning
            self.apply_gradient_learning(results)
            all_results.extend(results)

            # 4. Get best so far
            best = self.get_best_modification()
            if best:
                print(f"Best val_bpb so far: {self.best_val_bpb:.4f}")

        self.shutdown()

        return {
            "total_experiments": len(all_results),
            "best_val_bpb": self.best_val_bpb,
            "patterns_learned": len(self.patterns),
            "results": all_results,
        }


def demo():
    """Demo of AutoresearchSOME agent"""
    print("=== AutoresearchSOME Agent Demo ===\n")

    # Create agent
    agent = AutoresearchSOMEAgent(num_workers=4, experiment_minutes=5)

    # Sample train.py (just for demo)
    base_code = "# Original train.py"

    # Run research loop
    results = agent.run_research_loop(
        base_code,
        num_iterations=2,
    )

    print(f"\n=== Final Results ===")
    print(f"Total experiments: {results['total_experiments']}")
    print(f"Best val_bpb: {results['best_val_bpb']}")
    print(f"Patterns learned: {results['patterns_learned']}")


if __name__ == "__main__":
    demo()
