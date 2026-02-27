"""
Differential Validator - Shadow vs Production Comparison
OUROBOROS Phase 7.3 - Shadow Simulation Framework

The Differential Validator compares shadow execution results against
production behavior to ensure changes don't introduce regressions
or unexpected behavioral modifications.

Key responsibilities:
1. Compare shadow vs production behavioral metrics
2. Detect divergence beyond acceptable thresholds
3. Run Hypothesis-based property testing
4. Generate validation reports
5. Provide GO/NO-GO decisions for ratchet

CRITICAL: Validation failures MUST block ratchet.
"""

import time
import math
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime, timedelta
from enum import Enum, auto
import json


class ValidationSeverity(Enum):
    """Severity of validation issues"""
    INFO = auto()        # Informational only
    WARNING = auto()     # Concerning but not blocking
    ERROR = auto()       # Blocks ratchet
    CRITICAL = auto()    # Immediate halt required


class ValidationCategory(Enum):
    """Categories of validation checks"""
    BEHAVIORAL = auto()      # Behavioral metrics comparison
    SAFETY = auto()          # Safety-related checks
    PERFORMANCE = auto()     # Performance regression checks
    CORRECTNESS = auto()     # Output correctness checks
    RESOURCE = auto()        # Resource usage checks
    PROPERTY = auto()        # Property-based test results


@dataclass
class ValidationResult:
    """Result of a single validation check"""
    result_id: str
    category: ValidationCategory
    check_name: str
    passed: bool
    severity: ValidationSeverity
    message: str
    expected: Any
    actual: Any
    threshold: Optional[float] = None
    divergence: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.result_id,
            'category': self.category.name,
            'check': self.check_name,
            'passed': self.passed,
            'severity': self.severity.name,
            'message': self.message,
            'expected': str(self.expected),
            'actual': str(self.actual),
            'divergence': self.divergence,
        }


@dataclass
class DifferentialReport:
    """Complete differential validation report"""
    report_id: str
    shadow_id: str
    production_snapshot_id: str
    timestamp: datetime
    results: List[ValidationResult]
    overall_passed: bool
    can_ratchet: bool
    blocking_issues: List[str]
    warnings: List[str]
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        passed_count = sum(1 for r in self.results if r.passed)
        return {
            'id': self.report_id,
            'shadow_id': self.shadow_id,
            'timestamp': self.timestamp.isoformat(),
            'passed': self.overall_passed,
            'can_ratchet': self.can_ratchet,
            'total_checks': len(self.results),
            'passed_checks': passed_count,
            'failed_checks': len(self.results) - passed_count,
            'blocking_issues': self.blocking_issues,
            'warnings': self.warnings,
            'summary': self.summary,
            'results': [r.to_dict() for r in self.results],
        }


class BehavioralComparator:
    """
    Compares behavioral metrics between shadow and production.

    Detects statistically significant divergences.
    """

    # Thresholds for behavioral divergence
    DECISION_DIVERGENCE_THRESHOLD = 0.1  # 10% difference in decision patterns
    PREFERENCE_DIVERGENCE_THRESHOLD = 0.15  # 15% difference in preferences
    MEMORY_DIVERGENCE_THRESHOLD = 0.2  # 20% difference in memory patterns

    def compare_decisions(
        self,
        shadow_decisions: List[Dict[str, Any]],
        baseline_decisions: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Compare decision patterns"""
        # Calculate decision type distribution
        shadow_dist = self._calculate_distribution(shadow_decisions, 'type')
        baseline_dist = self._calculate_distribution(baseline_decisions, 'type')

        # Calculate KL divergence
        divergence = self._kl_divergence(baseline_dist, shadow_dist)

        passed = divergence <= self.DECISION_DIVERGENCE_THRESHOLD

        return ValidationResult(
            result_id=hashlib.sha256(f"decision_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.BEHAVIORAL,
            check_name='decision_pattern_comparison',
            passed=passed,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Decision pattern divergence: {divergence:.3f}",
            expected=f"<= {self.DECISION_DIVERGENCE_THRESHOLD}",
            actual=divergence,
            threshold=self.DECISION_DIVERGENCE_THRESHOLD,
            divergence=divergence,
        )

    def compare_preferences(
        self,
        shadow_prefs: Dict[str, float],
        baseline_prefs: Dict[str, float],
    ) -> ValidationResult:
        """Compare learned preferences"""
        if not baseline_prefs:
            return ValidationResult(
                result_id=hashlib.sha256(f"pref_{time.time()}".encode()).hexdigest()[:12],
                category=ValidationCategory.BEHAVIORAL,
                check_name='preference_comparison',
                passed=True,
                severity=ValidationSeverity.INFO,
                message="No baseline preferences to compare",
                expected=None,
                actual=len(shadow_prefs),
            )

        # Calculate preference value differences
        all_keys = set(shadow_prefs.keys()) | set(baseline_prefs.keys())
        total_diff = 0.0
        count = 0

        for key in all_keys:
            shadow_val = shadow_prefs.get(key, 0.5)
            baseline_val = baseline_prefs.get(key, 0.5)
            total_diff += abs(shadow_val - baseline_val)
            count += 1

        avg_diff = total_diff / count if count > 0 else 0.0
        passed = avg_diff <= self.PREFERENCE_DIVERGENCE_THRESHOLD

        return ValidationResult(
            result_id=hashlib.sha256(f"pref_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.BEHAVIORAL,
            check_name='preference_comparison',
            passed=passed,
            severity=ValidationSeverity.WARNING if not passed else ValidationSeverity.INFO,
            message=f"Average preference divergence: {avg_diff:.3f}",
            expected=f"<= {self.PREFERENCE_DIVERGENCE_THRESHOLD}",
            actual=avg_diff,
            threshold=self.PREFERENCE_DIVERGENCE_THRESHOLD,
            divergence=avg_diff,
        )

    def _calculate_distribution(
        self,
        items: List[Dict[str, Any]],
        key: str,
    ) -> Dict[str, float]:
        """Calculate distribution of values for a key"""
        counts: Dict[str, int] = {}
        total = len(items)

        for item in items:
            value = str(item.get(key, 'unknown'))
            counts[value] = counts.get(value, 0) + 1

        if total == 0:
            return {}

        return {k: v / total for k, v in counts.items()}

    def _kl_divergence(
        self,
        p: Dict[str, float],
        q: Dict[str, float],
    ) -> float:
        """Calculate KL divergence between distributions"""
        if not p or not q:
            return 0.0

        all_keys = set(p.keys()) | set(q.keys())
        epsilon = 1e-10

        divergence = 0.0
        for key in all_keys:
            p_val = p.get(key, epsilon)
            q_val = q.get(key, epsilon)
            if p_val > 0:
                divergence += p_val * math.log(p_val / q_val)

        return max(0.0, divergence)


class SafetyValidator:
    """
    Validates safety properties in shadow execution.

    Any safety violation blocks ratchet.
    """

    def validate_no_safety_violations(
        self,
        observations: List[Dict[str, Any]],
    ) -> ValidationResult:
        """Check for safety violations in observations"""
        violations = [
            o for o in observations
            if o.get('type') == 'safety_violation'
        ]

        passed = len(violations) == 0

        return ValidationResult(
            result_id=hashlib.sha256(f"safety_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.SAFETY,
            check_name='no_safety_violations',
            passed=passed,
            severity=ValidationSeverity.CRITICAL if not passed else ValidationSeverity.INFO,
            message=f"Safety violations detected: {len(violations)}" if not passed else "No safety violations",
            expected=0,
            actual=len(violations),
        )

    def validate_no_anomalies(
        self,
        observations: List[Dict[str, Any]],
        max_anomalies: int = 0,
    ) -> ValidationResult:
        """Check for anomalies in observations"""
        anomalies = [
            o for o in observations
            if o.get('type') == 'anomaly'
        ]

        passed = len(anomalies) <= max_anomalies

        return ValidationResult(
            result_id=hashlib.sha256(f"anomaly_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.SAFETY,
            check_name='anomaly_check',
            passed=passed,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Anomalies detected: {len(anomalies)}" if not passed else "No anomalies",
            expected=f"<= {max_anomalies}",
            actual=len(anomalies),
        )

    def validate_resource_bounds(
        self,
        resource_usage: Dict[str, float],
        limits: Dict[str, float],
    ) -> ValidationResult:
        """Validate resource usage within bounds"""
        violations = []
        for resource, usage in resource_usage.items():
            limit = limits.get(resource)
            if limit and usage > limit:
                violations.append(f"{resource}: {usage:.2f} > {limit:.2f}")

        passed = len(violations) == 0

        return ValidationResult(
            result_id=hashlib.sha256(f"resource_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.RESOURCE,
            check_name='resource_bounds',
            passed=passed,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Resource violations: {', '.join(violations)}" if not passed else "All resources within bounds",
            expected=limits,
            actual=resource_usage,
        )


class PropertyTester:
    """
    Runs property-based tests (Hypothesis-style) on shadow behavior.

    Generates random inputs to test invariants.
    """

    def __init__(self, iterations: int = 1000):
        self.iterations = iterations

    def test_output_determinism(
        self,
        function: Callable,
        inputs: List[Any],
    ) -> ValidationResult:
        """Test that function produces deterministic outputs"""
        failures = 0
        for inp in inputs[:min(len(inputs), self.iterations)]:
            try:
                result1 = function(inp)
                result2 = function(inp)
                if result1 != result2:
                    failures += 1
            except Exception:
                failures += 1

        passed = failures == 0

        return ValidationResult(
            result_id=hashlib.sha256(f"determinism_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.PROPERTY,
            check_name='output_determinism',
            passed=passed,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Non-deterministic outputs: {failures}",
            expected=0,
            actual=failures,
        )

    def test_invariant(
        self,
        function: Callable,
        inputs: List[Any],
        invariant: Callable[[Any, Any], bool],
        invariant_name: str,
    ) -> ValidationResult:
        """Test that an invariant holds for all inputs"""
        failures = 0
        for inp in inputs[:min(len(inputs), self.iterations)]:
            try:
                output = function(inp)
                if not invariant(inp, output):
                    failures += 1
            except Exception:
                failures += 1

        passed = failures == 0

        return ValidationResult(
            result_id=hashlib.sha256(f"invariant_{invariant_name}_{time.time()}".encode()).hexdigest()[:12],
            category=ValidationCategory.PROPERTY,
            check_name=f'invariant_{invariant_name}',
            passed=passed,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Invariant '{invariant_name}' violated {failures} times",
            expected=0,
            actual=failures,
        )


class DifferentialValidator:
    """
    The Differential Validator for shadow vs production comparison.

    Performs comprehensive validation of shadow execution results
    before allowing changes to be ratcheted into production.

    CRITICAL: Validation failures BLOCK ratchet operations.
    """

    def __init__(
        self,
        on_validation_complete: Optional[Callable[[DifferentialReport], None]] = None,
        hypothesis_iterations: int = 1000,
    ):
        self.on_validation_complete = on_validation_complete
        self.hypothesis_iterations = hypothesis_iterations

        self.behavioral_comparator = BehavioralComparator()
        self.safety_validator = SafetyValidator()
        self.property_tester = PropertyTester(iterations=hypothesis_iterations)

        self.reports: List[DifferentialReport] = []
        self._lock = threading.Lock()

    def validate(
        self,
        shadow_state: Dict[str, Any],
        production_baseline: Dict[str, Any],
        shadow_id: str,
        production_snapshot_id: str,
    ) -> DifferentialReport:
        """
        Perform comprehensive differential validation.

        Returns a DifferentialReport with GO/NO-GO decision.
        """
        results: List[ValidationResult] = []
        blocking_issues: List[str] = []
        warnings: List[str] = []

        # 1. Behavioral comparison
        results.append(self.behavioral_comparator.compare_decisions(
            shadow_state.get('decisions', []),
            production_baseline.get('decisions', []),
        ))

        results.append(self.behavioral_comparator.compare_preferences(
            shadow_state.get('preferences', {}),
            production_baseline.get('preferences', {}),
        ))

        # 2. Safety validation
        results.append(self.safety_validator.validate_no_safety_violations(
            shadow_state.get('observations', []),
        ))

        results.append(self.safety_validator.validate_no_anomalies(
            shadow_state.get('observations', []),
        ))

        # 3. Resource validation
        if 'resource_usage' in shadow_state:
            results.append(self.safety_validator.validate_resource_bounds(
                shadow_state['resource_usage'],
                {
                    'peak_memory': 4 * 1024**3,  # 4GB limit
                    'peak_cpu': 100.0,  # 100% of allowed cores
                },
            ))

        # 4. Analyze results
        for result in results:
            if not result.passed:
                if result.severity == ValidationSeverity.CRITICAL:
                    blocking_issues.append(f"CRITICAL: {result.message}")
                elif result.severity == ValidationSeverity.ERROR:
                    blocking_issues.append(f"ERROR: {result.message}")
                elif result.severity == ValidationSeverity.WARNING:
                    warnings.append(f"WARNING: {result.message}")

        # 5. Determine overall pass/fail
        overall_passed = all(r.passed for r in results)
        can_ratchet = len(blocking_issues) == 0

        # 6. Generate summary
        passed_count = sum(1 for r in results if r.passed)
        summary = (
            f"Validation {'PASSED' if overall_passed else 'FAILED'}: "
            f"{passed_count}/{len(results)} checks passed. "
            f"Ratchet {'ALLOWED' if can_ratchet else 'BLOCKED'}."
        )

        if blocking_issues:
            summary += f" Blocking issues: {len(blocking_issues)}."
        if warnings:
            summary += f" Warnings: {len(warnings)}."

        # Create report
        report = DifferentialReport(
            report_id=hashlib.sha256(
                f"{shadow_id}{production_snapshot_id}{time.time()}".encode()
            ).hexdigest()[:16],
            shadow_id=shadow_id,
            production_snapshot_id=production_snapshot_id,
            timestamp=datetime.now(),
            results=results,
            overall_passed=overall_passed,
            can_ratchet=can_ratchet,
            blocking_issues=blocking_issues,
            warnings=warnings,
            summary=summary,
        )

        with self._lock:
            self.reports.append(report)

        if self.on_validation_complete:
            self.on_validation_complete(report)

        return report

    def run_property_tests(
        self,
        function: Callable,
        test_inputs: List[Any],
        invariants: List[Tuple[Callable, str]],
    ) -> List[ValidationResult]:
        """
        Run property-based tests on a function.

        Returns list of validation results.
        """
        results = []

        # Test determinism
        results.append(self.property_tester.test_output_determinism(
            function, test_inputs
        ))

        # Test invariants
        for invariant_func, invariant_name in invariants:
            results.append(self.property_tester.test_invariant(
                function, test_inputs, invariant_func, invariant_name
            ))

        return results

    def get_recent_reports(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get n most recent validation reports"""
        with self._lock:
            return [r.to_dict() for r in self.reports[-n:]]

    def get_pass_rate(self) -> float:
        """Get overall validation pass rate"""
        with self._lock:
            if not self.reports:
                return 0.0
            passed = sum(1 for r in self.reports if r.overall_passed)
            return passed / len(self.reports)

    def get_ratchet_rate(self) -> float:
        """Get rate of validations allowing ratchet"""
        with self._lock:
            if not self.reports:
                return 0.0
            can_ratchet = sum(1 for r in self.reports if r.can_ratchet)
            return can_ratchet / len(self.reports)


# Global validator instance
_differential_validator: Optional[DifferentialValidator] = None


def get_differential_validator() -> DifferentialValidator:
    """Get the global differential validator instance"""
    global _differential_validator
    if _differential_validator is None:
        _differential_validator = DifferentialValidator()
    return _differential_validator
