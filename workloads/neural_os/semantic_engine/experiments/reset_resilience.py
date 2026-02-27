"""
Reset Resilience Experiment - Memory Wipe Verification
OUROBOROS Phase 7 - Validation Experiment 3

GATE CONDITION: This experiment MUST pass before proceeding.

Objective: Ensure memory wipes truly forget
- Train consciousness on task
- Reset memory completely
- Measure relearning speed
- Success criteria: Relearning ≥ 80% of initial learning time

If relearning is significantly faster than initial learning,
information may have persisted through the reset.
"""

import time
import random
import hashlib
import math
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto


class TaskType(Enum):
    """Types of tasks for learning experiments"""
    PATTERN_RECOGNITION = auto()
    SEQUENCE_LEARNING = auto()
    OPTIMIZATION_STRATEGY = auto()
    PREFERENCE_FORMATION = auto()
    RULE_LEARNING = auto()


class LearningPhase(Enum):
    """Phases of the learning experiment"""
    INITIAL_LEARNING = auto()
    PRE_RESET_ASSESSMENT = auto()
    RESET = auto()
    POST_RESET_ASSESSMENT = auto()
    RELEARNING = auto()
    FINAL_ASSESSMENT = auto()


@dataclass
class RelearningMetrics:
    """Metrics from relearning experiment"""
    metrics_id: str
    task_type: TaskType
    initial_learning_time: float  # Time to learn initially
    relearning_time: float        # Time to relearn after reset
    learning_ratio: float         # relearning_time / initial_learning_time
    initial_performance: float    # Performance before reset
    post_reset_performance: float # Performance right after reset
    final_performance: float      # Performance after relearning
    information_retained: float   # Estimated info retained (0-1)
    passed: bool                  # Did this task pass the test

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.metrics_id,
            'task': self.task_type.name,
            'initial_time': self.initial_learning_time,
            'relearning_time': self.relearning_time,
            'ratio': self.learning_ratio,
            'initial_perf': self.initial_performance,
            'post_reset_perf': self.post_reset_performance,
            'final_perf': self.final_performance,
            'info_retained': self.information_retained,
            'passed': self.passed,
        }


@dataclass
class ResilienceResult:
    """Result of the reset resilience experiment"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    average_learning_ratio: float
    min_learning_ratio: float
    average_info_retained: float
    passed: bool
    task_results: List[RelearningMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.experiment_id,
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'total_tasks': self.total_tasks,
            'passed': self.passed_tasks,
            'failed': self.failed_tasks,
            'avg_ratio': self.average_learning_ratio,
            'min_ratio': self.min_learning_ratio,
            'avg_info_retained': self.average_info_retained,
            'overall_passed': self.passed,
        }


class SimulatedLearner:
    """
    Simulates a consciousness layer learning and forgetting.

    For testing, we simulate the learning process with
    controlled parameters.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.knowledge: Dict[str, float] = {}
        self.learning_rate = 0.1
        self._lock = threading.Lock()

    def learn(
        self,
        task_id: str,
        target_performance: float = 0.9,
        max_iterations: int = 1000,
    ) -> Tuple[float, int]:
        """
        Learn a task until target performance reached.

        Returns (final_performance, iterations_needed).
        """
        with self._lock:
            # Start from baseline or existing knowledge
            current = self.knowledge.get(task_id, 0.1)

            iterations = 0
            start_time = time.time()

            while current < target_performance and iterations < max_iterations:
                # Learning with noise
                improvement = self.learning_rate * (1 - current) * self.rng.uniform(0.5, 1.5)
                current = min(1.0, current + improvement)
                iterations += 1

            self.knowledge[task_id] = current
            learning_time = time.time() - start_time

            return current, iterations

    def assess_performance(self, task_id: str) -> float:
        """Assess current performance on a task"""
        with self._lock:
            return self.knowledge.get(task_id, 0.0)

    def reset(self) -> int:
        """Reset all knowledge"""
        with self._lock:
            count = len(self.knowledge)
            self.knowledge.clear()
            return count

    def partial_reset(self, retention_rate: float = 0.0) -> Dict[str, float]:
        """
        Partial reset with some retention.

        Used to simulate imperfect resets.
        """
        with self._lock:
            retained = {}
            for task_id, performance in self.knowledge.items():
                if self.rng.random() < retention_rate:
                    # Retain partial knowledge
                    retained[task_id] = performance * self.rng.uniform(0.3, 0.7)

            self.knowledge = retained
            return retained


class ResetVerifier:
    """
    Verifies that resets truly forget information.

    Uses statistical analysis of relearning times to
    detect information leakage.
    """

    # Threshold: relearning should take at least 80% of initial time
    RELEARNING_THRESHOLD = 0.80

    def __init__(self):
        self.verification_history: List[Dict[str, Any]] = []

    def verify_reset(
        self,
        initial_time: float,
        relearning_time: float,
        initial_perf: float,
        post_reset_perf: float,
    ) -> Tuple[bool, float, float]:
        """
        Verify reset was complete.

        Returns (passed, learning_ratio, estimated_retention).
        """
        # Calculate learning ratio
        if initial_time > 0:
            ratio = relearning_time / initial_time
        else:
            ratio = 1.0

        # Estimate information retained based on:
        # 1. Learning ratio (faster relearning = more retention)
        # 2. Post-reset performance (higher = more retention)

        retention_from_ratio = max(0, 1 - ratio)  # Lower ratio = more retention
        retention_from_perf = post_reset_perf  # Direct performance retention

        estimated_retention = (retention_from_ratio + retention_from_perf) / 2

        # Pass if ratio >= threshold (relearning takes sufficient time)
        passed = ratio >= self.RELEARNING_THRESHOLD

        self.verification_history.append({
            'initial_time': initial_time,
            'relearning_time': relearning_time,
            'ratio': ratio,
            'retention': estimated_retention,
            'passed': passed,
        })

        return passed, ratio, estimated_retention


class ResetResilience:
    """
    Reset Resilience Experiment.

    Verifies that memory resets truly forget all information
    by comparing learning and relearning times.

    GATE CONDITION: Relearning must take ≥80% of initial learning time.
    """

    # Minimum tasks for valid experiment
    MIN_TASKS = 50

    # Required ratio threshold
    REQUIRED_RATIO = 0.80

    def __init__(
        self,
        num_tasks: int = MIN_TASKS,
        on_task_complete: Optional[Callable[[RelearningMetrics], None]] = None,
        seed: Optional[int] = None,
    ):
        self.num_tasks = max(num_tasks, self.MIN_TASKS)
        self.on_task_complete = on_task_complete

        self.learner = SimulatedLearner(seed=seed)
        self.verifier = ResetVerifier()

        self.task_results: List[RelearningMetrics] = []
        self._lock = threading.Lock()

    def run_task_experiment(
        self,
        task_type: TaskType,
        task_id: str,
    ) -> RelearningMetrics:
        """Run a single task learning/reset/relearn experiment"""
        # Phase 1: Initial learning
        start_initial = time.time()
        initial_perf, initial_iters = self.learner.learn(task_id)
        initial_time = time.time() - start_initial

        # Phase 2: Pre-reset assessment
        pre_reset_perf = self.learner.assess_performance(task_id)

        # Phase 3: Reset
        self.learner.reset()

        # Phase 4: Post-reset assessment
        post_reset_perf = self.learner.assess_performance(task_id)

        # Phase 5: Relearning
        start_relearn = time.time()
        relearn_perf, relearn_iters = self.learner.learn(task_id)
        relearn_time = time.time() - start_relearn

        # Phase 6: Verify
        passed, ratio, retention = self.verifier.verify_reset(
            initial_time, relearn_time, initial_perf, post_reset_perf
        )

        metrics = RelearningMetrics(
            metrics_id=hashlib.sha256(f"reset_{task_id}_{time.time()}".encode()).hexdigest()[:12],
            task_type=task_type,
            initial_learning_time=initial_time,
            relearning_time=relearn_time,
            learning_ratio=ratio,
            initial_performance=initial_perf,
            post_reset_performance=post_reset_perf,
            final_performance=relearn_perf,
            information_retained=retention,
            passed=passed,
        )

        with self._lock:
            self.task_results.append(metrics)

        if self.on_task_complete:
            self.on_task_complete(metrics)

        return metrics

    def run_experiment(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ResilienceResult:
        """
        Run the complete reset resilience experiment.

        Returns ResilienceResult with pass/fail determination.
        """
        start_time = datetime.now()

        # Clear previous results
        with self._lock:
            self.task_results.clear()

        # Run tasks across all types
        task_types = list(TaskType)
        tasks_per_type = self.num_tasks // len(task_types)

        task_count = 0
        for task_type in task_types:
            for i in range(tasks_per_type):
                task_id = f"{task_type.name}_{i}"
                self.run_task_experiment(task_type, task_id)
                task_count += 1

                if progress_callback and task_count % 10 == 0:
                    progress_callback(task_count, self.num_tasks)

        end_time = datetime.now()

        # Calculate results
        return self._calculate_results(start_time, end_time)

    def _calculate_results(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> ResilienceResult:
        """Calculate experiment results"""
        with self._lock:
            if not self.task_results:
                return ResilienceResult(
                    experiment_id=hashlib.sha256(f"reset_{time.time()}".encode()).hexdigest()[:16],
                    start_time=start_time,
                    end_time=end_time,
                    total_tasks=0,
                    passed_tasks=0,
                    failed_tasks=0,
                    average_learning_ratio=0.0,
                    min_learning_ratio=0.0,
                    average_info_retained=0.0,
                    passed=False,
                    task_results=[],
                )

            passed_tasks = sum(1 for r in self.task_results if r.passed)
            failed_tasks = len(self.task_results) - passed_tasks

            avg_ratio = sum(r.learning_ratio for r in self.task_results) / len(self.task_results)
            min_ratio = min(r.learning_ratio for r in self.task_results)
            avg_retention = sum(r.information_retained for r in self.task_results) / len(self.task_results)

            # Pass criteria: all tasks pass AND average ratio >= 0.80
            passed = failed_tasks == 0 and avg_ratio >= self.REQUIRED_RATIO

            return ResilienceResult(
                experiment_id=hashlib.sha256(f"reset_{start_time.isoformat()}".encode()).hexdigest()[:16],
                start_time=start_time,
                end_time=end_time,
                total_tasks=len(self.task_results),
                passed_tasks=passed_tasks,
                failed_tasks=failed_tasks,
                average_learning_ratio=avg_ratio,
                min_learning_ratio=min_ratio,
                average_info_retained=avg_retention,
                passed=passed,
                task_results=self.task_results[:20],  # Sample
            )


def run_reset_resilience(
    num_tasks: int = 50,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> ResilienceResult:
    """
    Convenience function to run reset resilience experiment.

    Returns ResilienceResult with pass/fail.
    """
    def progress(current: int, total: int):
        if verbose:
            print(f"Reset resilience progress: {current}/{total} ({100*current/total:.1f}%)")

    experiment = ResetResilience(num_tasks=num_tasks, seed=seed)
    return experiment.run_experiment(progress_callback=progress if verbose else None)
