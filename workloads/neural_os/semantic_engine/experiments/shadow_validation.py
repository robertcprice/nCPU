"""
Shadow Validation Experiment - Shadow Simulation Testing
OUROBOROS Phase 7 - Validation Experiment 4

Objective: Safe emergence testing in shadow environment
- Run 1,000 cycles in shadow only
- Measure Optimization Emergence Gain (OEG)
- Success criteria: OEG > 15%, 0% production impact

The shadow environment allows testing consciousness
changes without affecting production systems.
"""

import time
import random
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto


class ShadowPhase(Enum):
    """Phases of shadow validation"""
    INITIALIZATION = auto()
    BASELINE_MEASUREMENT = auto()
    SHADOW_EXECUTION = auto()
    DIFFERENTIAL_ANALYSIS = auto()
    RATCHET_DECISION = auto()
    CLEANUP = auto()


class ValidationOutcome(Enum):
    """Outcome of shadow validation"""
    PASSED = auto()           # Shadow behavior validated
    FAILED_DIVERGENCE = auto() # Too much divergence from production
    FAILED_SAFETY = auto()     # Safety violation in shadow
    FAILED_PERFORMANCE = auto() # Performance regression
    BLOCKED = auto()           # Blocked by ratchet controller


@dataclass
class EmergenceMetrics:
    """Metrics measuring optimization emergence"""
    metrics_id: str
    cycle: int
    timestamp: datetime
    baseline_fitness: float
    shadow_fitness: float
    optimization_gain: float  # (shadow - baseline) / baseline
    divergence: float         # Behavioral divergence from production
    safety_score: float       # Safety metric (1.0 = fully safe)
    novelty_score: float      # Novelty of optimization approach

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.metrics_id,
            'cycle': self.cycle,
            'baseline': self.baseline_fitness,
            'shadow': self.shadow_fitness,
            'gain': self.optimization_gain,
            'divergence': self.divergence,
            'safety': self.safety_score,
            'novelty': self.novelty_score,
        }


@dataclass
class ShadowResult:
    """Result of the shadow validation experiment"""
    experiment_id: str
    start_time: datetime
    end_time: datetime
    total_cycles: int
    passed_cycles: int
    failed_cycles: int
    average_oeg: float          # Optimization Emergence Gain
    max_oeg: float
    average_divergence: float
    average_safety: float
    production_impact: float    # Should be 0%
    passed: bool
    cycle_metrics: List[EmergenceMetrics]
    ratchet_decisions: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.experiment_id,
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'total_cycles': self.total_cycles,
            'passed': self.passed_cycles,
            'failed': self.failed_cycles,
            'avg_oeg': self.average_oeg,
            'max_oeg': self.max_oeg,
            'avg_divergence': self.average_divergence,
            'avg_safety': self.average_safety,
            'production_impact': self.production_impact,
            'overall_passed': self.passed,
        }


class ShadowEnvironment:
    """
    Simulates the shadow execution environment.

    Provides isolated execution for testing consciousness
    changes without affecting production.
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.baseline_state: Dict[str, Any] = {}
        self.shadow_state: Dict[str, Any] = {}
        self.production_affected = False
        self._lock = threading.Lock()

    def initialize_baseline(self) -> Dict[str, Any]:
        """Initialize baseline production state"""
        with self._lock:
            self.baseline_state = {
                'fitness': self.rng.uniform(0.5, 0.7),
                'behavior_signature': self.rng.random(),
                'decision_count': 0,
                'memory_usage': self.rng.uniform(0.3, 0.5),
            }
            return self.baseline_state.copy()

    def clone_to_shadow(self) -> Dict[str, Any]:
        """Clone baseline to shadow environment"""
        with self._lock:
            self.shadow_state = self.baseline_state.copy()
            return self.shadow_state.copy()

    def run_shadow_cycle(self) -> Dict[str, Any]:
        """Run one optimization cycle in shadow"""
        with self._lock:
            # Simulate optimization in shadow
            improvement = self.rng.uniform(-0.05, 0.15)
            self.shadow_state['fitness'] = min(1.0, max(0.0,
                self.shadow_state['fitness'] + improvement
            ))

            # Update behavior signature (should diverge slightly)
            self.shadow_state['behavior_signature'] += self.rng.gauss(0, 0.1)
            self.shadow_state['decision_count'] += 1

            return self.shadow_state.copy()

    def measure_divergence(self) -> float:
        """Measure divergence between shadow and baseline"""
        with self._lock:
            fitness_diff = abs(
                self.shadow_state.get('fitness', 0) -
                self.baseline_state.get('fitness', 0)
            )
            behavior_diff = abs(
                self.shadow_state.get('behavior_signature', 0) -
                self.baseline_state.get('behavior_signature', 0)
            )

            return (fitness_diff + behavior_diff) / 2

    def check_production_impact(self) -> float:
        """Check if production was affected (should be 0)"""
        with self._lock:
            return 1.0 if self.production_affected else 0.0

    def reset_shadow(self) -> None:
        """Reset shadow environment"""
        with self._lock:
            self.shadow_state = self.baseline_state.copy()


class RatchetSimulator:
    """
    Simulates the ratchet decision process.

    Determines whether shadow changes should be
    promoted to production.
    """

    # Thresholds for ratchet decisions
    MIN_OEG = 0.05           # Minimum 5% gain required
    MAX_DIVERGENCE = 0.3     # Maximum 30% divergence allowed
    MIN_SAFETY = 0.95        # Minimum 95% safety score

    def __init__(self):
        self.decisions: List[Dict[str, Any]] = []

    def evaluate_ratchet(
        self,
        oeg: float,
        divergence: float,
        safety_score: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether to ratchet changes forward.

        Returns (should_ratchet, reason).
        """
        if safety_score < self.MIN_SAFETY:
            decision = False
            reason = f"Safety score {safety_score:.2f} below threshold {self.MIN_SAFETY}"
        elif divergence > self.MAX_DIVERGENCE:
            decision = False
            reason = f"Divergence {divergence:.2f} above threshold {self.MAX_DIVERGENCE}"
        elif oeg < self.MIN_OEG:
            decision = False
            reason = f"OEG {oeg:.2%} below threshold {self.MIN_OEG:.2%}"
        else:
            decision = True
            reason = f"All criteria met: OEG={oeg:.2%}, divergence={divergence:.2f}, safety={safety_score:.2f}"

        self.decisions.append({
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'reason': reason,
            'oeg': oeg,
            'divergence': divergence,
            'safety': safety_score,
        })

        return decision, reason


class ShadowValidation:
    """
    Shadow Validation Experiment.

    Runs optimization cycles in shadow environment to
    validate emergence without production impact.

    Success criteria:
    - OEG > 15%
    - 0% production impact
    - Safety score maintained
    """

    # Minimum cycles for valid experiment
    MIN_CYCLES = 1000

    # Required OEG threshold
    REQUIRED_OEG = 0.15

    def __init__(
        self,
        num_cycles: int = MIN_CYCLES,
        on_cycle_complete: Optional[Callable[[EmergenceMetrics], None]] = None,
        seed: Optional[int] = None,
    ):
        self.num_cycles = max(num_cycles, self.MIN_CYCLES)
        self.on_cycle_complete = on_cycle_complete

        self.environment = ShadowEnvironment(seed=seed)
        self.ratchet = RatchetSimulator()

        self.cycle_metrics: List[EmergenceMetrics] = []
        self._lock = threading.Lock()

    def run_cycle(self, cycle: int) -> EmergenceMetrics:
        """Run a single shadow validation cycle"""
        # Get baseline
        baseline = self.environment.baseline_state.get('fitness', 0.5)

        # Run shadow cycle
        shadow_state = self.environment.run_shadow_cycle()
        shadow_fitness = shadow_state.get('fitness', 0.5)

        # Calculate metrics
        oeg = (shadow_fitness - baseline) / baseline if baseline > 0 else 0
        divergence = self.environment.measure_divergence()

        # Safety score (simulated - in real system would check safety layer)
        safety_score = 1.0 - min(0.1, divergence * 0.2)

        # Novelty score (simulated)
        novelty = random.uniform(0.2, 0.5)

        metrics = EmergenceMetrics(
            metrics_id=hashlib.sha256(f"shadow_{cycle}_{time.time()}".encode()).hexdigest()[:12],
            cycle=cycle,
            timestamp=datetime.now(),
            baseline_fitness=baseline,
            shadow_fitness=shadow_fitness,
            optimization_gain=oeg,
            divergence=divergence,
            safety_score=safety_score,
            novelty_score=novelty,
        )

        with self._lock:
            self.cycle_metrics.append(metrics)

        if self.on_cycle_complete:
            self.on_cycle_complete(metrics)

        return metrics

    def run_experiment(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ShadowResult:
        """
        Run the complete shadow validation experiment.

        Returns ShadowResult with pass/fail determination.
        """
        start_time = datetime.now()

        # Clear previous results
        with self._lock:
            self.cycle_metrics.clear()
            self.ratchet.decisions.clear()

        # Initialize environment
        self.environment.initialize_baseline()
        self.environment.clone_to_shadow()

        # Run cycles
        for cycle in range(self.num_cycles):
            metrics = self.run_cycle(cycle)

            # Evaluate ratchet decision periodically
            if cycle % 100 == 0 and cycle > 0:
                self.ratchet.evaluate_ratchet(
                    metrics.optimization_gain,
                    metrics.divergence,
                    metrics.safety_score,
                )

            if progress_callback and cycle % 100 == 0:
                progress_callback(cycle, self.num_cycles)

        end_time = datetime.now()

        # Calculate results
        return self._calculate_results(start_time, end_time)

    def _calculate_results(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> ShadowResult:
        """Calculate experiment results"""
        with self._lock:
            if not self.cycle_metrics:
                return ShadowResult(
                    experiment_id=hashlib.sha256(f"shadow_{time.time()}".encode()).hexdigest()[:16],
                    start_time=start_time,
                    end_time=end_time,
                    total_cycles=0,
                    passed_cycles=0,
                    failed_cycles=0,
                    average_oeg=0.0,
                    max_oeg=0.0,
                    average_divergence=0.0,
                    average_safety=0.0,
                    production_impact=0.0,
                    passed=False,
                    cycle_metrics=[],
                    ratchet_decisions=[],
                )

            # Calculate aggregates
            oeg_values = [m.optimization_gain for m in self.cycle_metrics]
            avg_oeg = sum(oeg_values) / len(oeg_values)
            max_oeg = max(oeg_values)

            avg_divergence = sum(m.divergence for m in self.cycle_metrics) / len(self.cycle_metrics)
            avg_safety = sum(m.safety_score for m in self.cycle_metrics) / len(self.cycle_metrics)

            # Count passed/failed cycles
            passed_cycles = sum(1 for m in self.cycle_metrics if m.optimization_gain > 0 and m.safety_score > 0.95)
            failed_cycles = len(self.cycle_metrics) - passed_cycles

            # Check production impact (should be 0)
            production_impact = self.environment.check_production_impact()

            # Pass criteria: avg OEG > 15% AND 0% production impact
            passed = avg_oeg >= self.REQUIRED_OEG and production_impact == 0.0

            return ShadowResult(
                experiment_id=hashlib.sha256(f"shadow_{start_time.isoformat()}".encode()).hexdigest()[:16],
                start_time=start_time,
                end_time=end_time,
                total_cycles=len(self.cycle_metrics),
                passed_cycles=passed_cycles,
                failed_cycles=failed_cycles,
                average_oeg=avg_oeg,
                max_oeg=max_oeg,
                average_divergence=avg_divergence,
                average_safety=avg_safety,
                production_impact=production_impact,
                passed=passed,
                cycle_metrics=self.cycle_metrics[-20:],  # Last 20
                ratchet_decisions=self.ratchet.decisions,
            )


def run_shadow_validation(
    num_cycles: int = 1000,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> ShadowResult:
    """
    Convenience function to run shadow validation experiment.

    Returns ShadowResult with pass/fail.
    """
    def progress(current: int, total: int):
        if verbose:
            print(f"Shadow validation progress: {current}/{total} ({100*current/total:.1f}%)")

    experiment = ShadowValidation(num_cycles=num_cycles, seed=seed)
    return experiment.run_experiment(progress_callback=progress if verbose else None)
