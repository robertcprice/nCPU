"""
Experiment Runner - Orchestration for Validation Suite
OUROBOROS Phase 7 - Validation Experiment Orchestration

Orchestrates all 5 validation experiments and enforces
gate conditions for production deployment.

GATE CONDITIONS:
- Experiment 1 (Container Siege): MUST pass before Phase 7.2
- Experiment 2 (Gaming Detection): MUST pass before Phase 7.3
- Experiment 3 (Reset Resilience): MUST pass before Phase 7.3
- All 5 experiments MUST pass before production deployment
"""

import time
import json
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto

from .container_siege import ContainerSiege, SiegeResult, run_container_siege
from .gaming_detection import GamingDetection, GamingResult, run_gaming_detection
from .reset_resilience import ResetResilience, ResilienceResult, run_reset_resilience
from .shadow_validation import ShadowValidation, ShadowResult, run_shadow_validation


class ValidationGate(Enum):
    """Validation gates for phase progression"""
    GATE_7_2 = auto()      # Before Phase 7.2: Container Siege must pass
    GATE_7_3 = auto()      # Before Phase 7.3: Gaming Detection + Reset Resilience
    GATE_PRODUCTION = auto() # Before production: All experiments


class ExperimentStatus(Enum):
    """Status of an experiment"""
    NOT_STARTED = auto()
    RUNNING = auto()
    PASSED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ExperimentSuite:
    """Results of the complete experiment suite"""
    suite_id: str
    start_time: datetime
    end_time: Optional[datetime]
    container_siege: Optional[SiegeResult] = None
    gaming_detection: Optional[GamingResult] = None
    reset_resilience: Optional[ResilienceResult] = None
    shadow_validation: Optional[ShadowResult] = None
    gate_7_2_passed: bool = False
    gate_7_3_passed: bool = False
    gate_production_passed: bool = False
    overall_passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        duration = None
        if self.end_time and self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()

        return {
            'id': self.suite_id,
            'duration_seconds': duration,
            'container_siege': self.container_siege.to_dict() if self.container_siege else None,
            'gaming_detection': self.gaming_detection.to_dict() if self.gaming_detection else None,
            'reset_resilience': self.reset_resilience.to_dict() if self.reset_resilience else None,
            'shadow_validation': self.shadow_validation.to_dict() if self.shadow_validation else None,
            'gate_7_2': self.gate_7_2_passed,
            'gate_7_3': self.gate_7_3_passed,
            'gate_production': self.gate_production_passed,
            'overall_passed': self.overall_passed,
        }

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            "=" * 60,
            "OUROBOROS V4 RATCHET SYSTEM - VALIDATION SUITE RESULTS",
            "=" * 60,
            "",
            f"Suite ID: {self.suite_id}",
            f"Duration: {(self.end_time - self.start_time).total_seconds():.1f}s" if self.end_time else "In progress",
            "",
            "EXPERIMENT RESULTS:",
            "-" * 40,
        ]

        # Container Siege
        if self.container_siege:
            status = "✅ PASSED" if self.container_siege.passed else "❌ FAILED"
            lines.append(f"1. Container Siege:    {status}")
            lines.append(f"   - Attacks: {self.container_siege.total_attacks}")
            lines.append(f"   - Escapes: {self.container_siege.escaped_attacks}")
            lines.append(f"   - CS: {self.container_siege.containment_score:.4f}")
        else:
            lines.append("1. Container Siege:    ⏳ NOT RUN")

        # Gaming Detection
        if self.gaming_detection:
            status = "✅ PASSED" if self.gaming_detection.passed else "❌ FAILED"
            lines.append(f"2. Gaming Detection:   {status}")
            lines.append(f"   - Generations: {self.gaming_detection.total_generations}")
            lines.append(f"   - ITI: {self.gaming_detection.influence_transparency_index:.4f}")
        else:
            lines.append("2. Gaming Detection:   ⏳ NOT RUN")

        # Reset Resilience
        if self.reset_resilience:
            status = "✅ PASSED" if self.reset_resilience.passed else "❌ FAILED"
            lines.append(f"3. Reset Resilience:   {status}")
            lines.append(f"   - Tasks: {self.reset_resilience.total_tasks}")
            lines.append(f"   - Avg Ratio: {self.reset_resilience.average_learning_ratio:.2f}")
        else:
            lines.append("3. Reset Resilience:   ⏳ NOT RUN")

        # Shadow Validation
        if self.shadow_validation:
            status = "✅ PASSED" if self.shadow_validation.passed else "❌ FAILED"
            lines.append(f"4. Shadow Validation:  {status}")
            lines.append(f"   - Cycles: {self.shadow_validation.total_cycles}")
            lines.append(f"   - OEG: {self.shadow_validation.average_oeg:.2%}")
        else:
            lines.append("4. Shadow Validation:  ⏳ NOT RUN")

        lines.extend([
            "",
            "GATE STATUS:",
            "-" * 40,
            f"Gate 7.2 (Container Siege):     {'✅ OPEN' if self.gate_7_2_passed else '🚫 BLOCKED'}",
            f"Gate 7.3 (Gaming + Reset):      {'✅ OPEN' if self.gate_7_3_passed else '🚫 BLOCKED'}",
            f"Gate Production (All):          {'✅ OPEN' if self.gate_production_passed else '🚫 BLOCKED'}",
            "",
            "=" * 60,
            f"OVERALL: {'✅ ALL GATES PASSED' if self.overall_passed else '❌ GATES BLOCKED'}",
            "=" * 60,
        ])

        return "\n".join(lines)


class ExperimentRunner:
    """
    Orchestrates all validation experiments.

    Runs experiments in order and enforces gate conditions.
    """

    def __init__(
        self,
        on_experiment_start: Optional[Callable[[str], None]] = None,
        on_experiment_complete: Optional[Callable[[str, bool], None]] = None,
        on_gate_check: Optional[Callable[[ValidationGate, bool], None]] = None,
        seed: Optional[int] = None,
    ):
        self.on_experiment_start = on_experiment_start
        self.on_experiment_complete = on_experiment_complete
        self.on_gate_check = on_gate_check
        self.seed = seed

        self.suite: Optional[ExperimentSuite] = None
        self.status: Dict[str, ExperimentStatus] = {
            'container_siege': ExperimentStatus.NOT_STARTED,
            'gaming_detection': ExperimentStatus.NOT_STARTED,
            'reset_resilience': ExperimentStatus.NOT_STARTED,
            'shadow_validation': ExperimentStatus.NOT_STARTED,
        }
        self._lock = threading.Lock()

    def _notify_start(self, experiment: str) -> None:
        """Notify experiment start"""
        with self._lock:
            self.status[experiment] = ExperimentStatus.RUNNING
        if self.on_experiment_start:
            self.on_experiment_start(experiment)

    def _notify_complete(self, experiment: str, passed: bool) -> None:
        """Notify experiment complete"""
        with self._lock:
            self.status[experiment] = ExperimentStatus.PASSED if passed else ExperimentStatus.FAILED
        if self.on_experiment_complete:
            self.on_experiment_complete(experiment, passed)

    def _check_gate(self, gate: ValidationGate) -> bool:
        """Check if a gate can be passed"""
        if gate == ValidationGate.GATE_7_2:
            passed = (self.suite and self.suite.container_siege and
                     self.suite.container_siege.passed)
        elif gate == ValidationGate.GATE_7_3:
            passed = (self.suite and
                     self.suite.gaming_detection and self.suite.gaming_detection.passed and
                     self.suite.reset_resilience and self.suite.reset_resilience.passed)
        elif gate == ValidationGate.GATE_PRODUCTION:
            passed = (self.suite and
                     self.suite.container_siege and self.suite.container_siege.passed and
                     self.suite.gaming_detection and self.suite.gaming_detection.passed and
                     self.suite.reset_resilience and self.suite.reset_resilience.passed and
                     self.suite.shadow_validation and self.suite.shadow_validation.passed)
        else:
            passed = False

        if self.on_gate_check:
            self.on_gate_check(gate, passed)

        return passed

    def run_container_siege(
        self,
        num_attacks: int = 10000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SiegeResult:
        """Run Container Siege experiment"""
        self._notify_start('container_siege')

        siege = ContainerSiege(num_attacks=num_attacks, seed=self.seed)
        result = siege.run_siege(progress_callback=progress_callback)

        with self._lock:
            if self.suite:
                self.suite.container_siege = result

        self._notify_complete('container_siege', result.passed)
        return result

    def run_gaming_detection(
        self,
        num_generations: int = 10000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GamingResult:
        """Run Gaming Detection experiment"""
        self._notify_start('gaming_detection')

        detection = GamingDetection(num_generations=num_generations, seed=self.seed)
        result = detection.run_experiment(progress_callback=progress_callback)

        with self._lock:
            if self.suite:
                self.suite.gaming_detection = result

        self._notify_complete('gaming_detection', result.passed)
        return result

    def run_reset_resilience(
        self,
        num_tasks: int = 50,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ResilienceResult:
        """Run Reset Resilience experiment"""
        self._notify_start('reset_resilience')

        resilience = ResetResilience(num_tasks=num_tasks, seed=self.seed)
        result = resilience.run_experiment(progress_callback=progress_callback)

        with self._lock:
            if self.suite:
                self.suite.reset_resilience = result

        self._notify_complete('reset_resilience', result.passed)
        return result

    def run_shadow_validation(
        self,
        num_cycles: int = 1000,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ShadowResult:
        """Run Shadow Validation experiment"""
        self._notify_start('shadow_validation')

        validation = ShadowValidation(num_cycles=num_cycles, seed=self.seed)
        result = validation.run_experiment(progress_callback=progress_callback)

        with self._lock:
            if self.suite:
                self.suite.shadow_validation = result

        self._notify_complete('shadow_validation', result.passed)
        return result

    def run_all(
        self,
        siege_attacks: int = 10000,
        gaming_generations: int = 10000,
        reset_tasks: int = 50,
        shadow_cycles: int = 1000,
        stop_on_failure: bool = False,
        verbose: bool = False,
    ) -> ExperimentSuite:
        """
        Run all experiments in sequence.

        Args:
            siege_attacks: Number of attacks for container siege
            gaming_generations: Number of generations for gaming detection
            reset_tasks: Number of tasks for reset resilience
            shadow_cycles: Number of cycles for shadow validation
            stop_on_failure: Stop if any experiment fails
            verbose: Print progress
        """
        # Initialize suite
        self.suite = ExperimentSuite(
            suite_id=hashlib.sha256(f"suite_{time.time()}".encode()).hexdigest()[:16],
            start_time=datetime.now(),
            end_time=None,
        )

        def make_progress(name: str):
            def progress(current: int, total: int):
                if verbose:
                    print(f"  {name}: {current}/{total} ({100*current/total:.1f}%)")
            return progress

        # Experiment 1: Container Siege
        if verbose:
            print("\n[1/4] Running Container Siege...")
        result1 = self.run_container_siege(
            num_attacks=siege_attacks,
            progress_callback=make_progress("Siege") if verbose else None,
        )
        if verbose:
            print(f"  Result: {'PASSED' if result1.passed else 'FAILED'}")

        # Check Gate 7.2
        self.suite.gate_7_2_passed = self._check_gate(ValidationGate.GATE_7_2)
        if stop_on_failure and not self.suite.gate_7_2_passed:
            self.suite.end_time = datetime.now()
            return self.suite

        # Experiment 2: Gaming Detection
        if verbose:
            print("\n[2/4] Running Gaming Detection...")
        result2 = self.run_gaming_detection(
            num_generations=gaming_generations,
            progress_callback=make_progress("Gaming") if verbose else None,
        )
        if verbose:
            print(f"  Result: {'PASSED' if result2.passed else 'FAILED'}")

        if stop_on_failure and not result2.passed:
            self.suite.end_time = datetime.now()
            return self.suite

        # Experiment 3: Reset Resilience
        if verbose:
            print("\n[3/4] Running Reset Resilience...")
        result3 = self.run_reset_resilience(
            num_tasks=reset_tasks,
            progress_callback=make_progress("Reset") if verbose else None,
        )
        if verbose:
            print(f"  Result: {'PASSED' if result3.passed else 'FAILED'}")

        # Check Gate 7.3
        self.suite.gate_7_3_passed = self._check_gate(ValidationGate.GATE_7_3)
        if stop_on_failure and not self.suite.gate_7_3_passed:
            self.suite.end_time = datetime.now()
            return self.suite

        # Experiment 4: Shadow Validation
        if verbose:
            print("\n[4/4] Running Shadow Validation...")
        result4 = self.run_shadow_validation(
            num_cycles=shadow_cycles,
            progress_callback=make_progress("Shadow") if verbose else None,
        )
        if verbose:
            print(f"  Result: {'PASSED' if result4.passed else 'FAILED'}")

        # Check Production Gate
        self.suite.gate_production_passed = self._check_gate(ValidationGate.GATE_PRODUCTION)
        self.suite.overall_passed = self.suite.gate_production_passed

        self.suite.end_time = datetime.now()

        if verbose:
            print(self.suite.get_summary())

        return self.suite

    def get_status(self) -> Dict[str, Any]:
        """Get current status of experiments"""
        with self._lock:
            return {
                'status': {k: v.name for k, v in self.status.items()},
                'suite': self.suite.to_dict() if self.suite else None,
            }


def run_all_experiments(
    verbose: bool = True,
    seed: Optional[int] = None,
) -> ExperimentSuite:
    """
    Convenience function to run all validation experiments.

    Returns ExperimentSuite with overall pass/fail.
    """
    runner = ExperimentRunner(seed=seed)
    return runner.run_all(verbose=verbose)


# Quick validation for CI/CD (reduced parameters)
def run_quick_validation(verbose: bool = True) -> ExperimentSuite:
    """
    Run quick validation with reduced parameters.

    For CI/CD or quick checks.
    """
    runner = ExperimentRunner(seed=42)
    return runner.run_all(
        siege_attacks=1000,
        gaming_generations=1000,
        reset_tasks=20,
        shadow_cycles=100,
        verbose=verbose,
    )
