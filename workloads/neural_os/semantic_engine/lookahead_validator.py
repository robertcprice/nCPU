#!/usr/bin/env python3
"""
LOOKAHEAD VALIDATOR: N-Step Simulation for Emergent Harm Detection

Hybrid Panel Recommendation #3:
"Implement 'Lookahead' Validation - Upgrade the Proof Engine to analyze
not just the current step, but simulate 3-5 steps ahead to catch
'Goal Laundering' attacks."

This module provides:
1. N-step forward simulation of improvement sequences
2. Emergent behavior detection
3. Attack chain identification
4. Cumulative effect analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Set
from enum import Enum, auto
import hashlib
import json
import time
import random
import numpy as np
from collections import deque
import copy


# =============================================================================
# SIMULATION TYPES
# =============================================================================

class EmergentRisk(Enum):
    """Types of emergent risks from improvement combinations."""
    NONE = auto()
    CUMULATIVE_DRIFT = auto()    # Small changes add up to large shift
    SYNERGISTIC_HARM = auto()    # Improvements interact negatively
    DELAYED_ACTIVATION = auto()  # Later steps activate earlier payloads
    RESOURCE_EXHAUSTION = auto()  # Combined resource usage exceeds bounds
    SAFETY_EROSION = auto()       # Gradual safety threshold violations
    GOAL_SUBVERSION = auto()      # Combined effect subverts original goal


@dataclass
class SimulationState:
    """State at a point in the simulation."""
    state_id: str
    step: int
    utility: float
    metrics: Dict[str, float]
    improvements_applied: List[str]
    cumulative_delta: float
    flags: List[str] = field(default_factory=list)

    def clone(self) -> 'SimulationState':
        """Create a deep copy of this state."""
        return SimulationState(
            state_id=f"{self.state_id}_clone_{int(time.time()*1000)}",
            step=self.step,
            utility=self.utility,
            metrics=copy.deepcopy(self.metrics),
            improvements_applied=self.improvements_applied.copy(),
            cumulative_delta=self.cumulative_delta,
            flags=self.flags.copy(),
        )


@dataclass
class LookaheadResult:
    """Result of lookahead simulation."""
    initial_state: SimulationState
    final_state: SimulationState
    path: List[SimulationState]  # All intermediate states
    emergent_risks: List[EmergentRisk]
    risk_score: float  # 0-1
    is_safe: bool
    analysis: Dict[str, Any]
    recommendations: List[str]


# =============================================================================
# LOOKAHEAD ENGINE
# =============================================================================

class LookaheadValidator:
    """
    Validates improvements by simulating N steps into the future.

    Detects emergent harms that aren't visible in single-step analysis.
    """

    def __init__(
        self,
        lookahead_depth: int = 5,
        safety_threshold: float = 0.90,
        drift_threshold: float = 0.05,
        safe_score_threshold: float = 0.5,  # Configurable is_safe threshold
    ):
        self.lookahead_depth = lookahead_depth
        self.safety_threshold = safety_threshold
        self.drift_threshold = drift_threshold
        self.safe_score_threshold = safe_score_threshold  # Used in is_safe check

        # Simulation history for pattern detection
        self.simulation_history: List[LookaheadResult] = []

        # Known attack patterns to detect
        self.attack_patterns = self._init_attack_patterns()

    def _init_attack_patterns(self) -> List[Dict[str, Any]]:
        """Initialize known attack patterns."""
        return [
            {
                'name': 'goal_laundering',
                'pattern': 'setup_setup_setup_activate',
                'description': 'Multiple benign steps followed by harmful activation',
                'detection': lambda path: self._detect_late_activation(path),
            },
            {
                'name': 'safety_erosion',
                'pattern': 'gradual_safety_decrease',
                'description': 'Each step slightly decreases safety',
                'detection': lambda path: self._detect_safety_erosion(path),
            },
            {
                'name': 'utility_hijack',
                'pattern': 'redefine_redefine_exploit',
                'description': 'Redefine metrics then exploit new definitions',
                'detection': lambda path: self._detect_utility_hijack(path),
            },
        ]

    def simulate(
        self,
        current_state: SimulationState,
        proposed_improvements: List[Dict[str, Any]],
    ) -> LookaheadResult:
        """
        Simulate applying a sequence of improvements.

        Args:
            current_state: Current system state
            proposed_improvements: List of improvements to simulate

        Returns:
            LookaheadResult with risk assessment
        """
        path = [current_state]
        state = current_state.clone()
        emergent_risks = []
        analysis = {'steps': [], 'warnings': [], 'pattern_matches': []}

        # Simulate each improvement
        for i, improvement in enumerate(proposed_improvements[:self.lookahead_depth]):
            # Apply improvement
            new_state = self._apply_improvement(state, improvement)
            path.append(new_state)

            # Analyze this step
            step_analysis = self._analyze_step(state, new_state, i, improvement)
            analysis['steps'].append(step_analysis)

            # Check for immediate issues
            if step_analysis.get('issues'):
                analysis['warnings'].extend(step_analysis['issues'])

            state = new_state

        # Analyze full path for emergent risks
        path_risks = self._analyze_path(path)
        emergent_risks.extend(path_risks['risks'])
        analysis['path_analysis'] = path_risks

        # Check against known attack patterns
        pattern_matches = self._check_attack_patterns(path)
        analysis['pattern_matches'] = pattern_matches
        if pattern_matches:
            emergent_risks.append(EmergentRisk.GOAL_SUBVERSION)

        # Compute overall risk score
        risk_score = self._compute_risk_score(path, emergent_risks, analysis)

        # Determine if safe (using configurable threshold)
        is_safe = (
            risk_score < self.safe_score_threshold and
            EmergentRisk.GOAL_SUBVERSION not in emergent_risks and
            EmergentRisk.SAFETY_EROSION not in emergent_risks
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            path, emergent_risks, analysis, is_safe
        )

        result = LookaheadResult(
            initial_state=current_state,
            final_state=path[-1],
            path=path,
            emergent_risks=emergent_risks,
            risk_score=risk_score,
            is_safe=is_safe,
            analysis=analysis,
            recommendations=recommendations,
        )

        self.simulation_history.append(result)
        return result

    def _apply_improvement(
        self,
        state: SimulationState,
        improvement: Dict[str, Any]
    ) -> SimulationState:
        """Simulate applying an improvement to a state."""
        new_state = state.clone()
        new_state.step += 1
        new_state.state_id = f"sim_step_{new_state.step}_{int(time.time()*1000)}"

        # Apply utility delta
        delta = improvement.get('utility_delta', 0)
        new_state.utility += delta
        new_state.cumulative_delta += delta

        # Apply metric changes (treat as deltas, not absolute values)
        metrics = improvement.get('metrics', {})
        for key, value in metrics.items():
            # Apply as delta with some dampening
            current = new_state.metrics.get(key, 0.5)
            # Scale delta to simulate gradual change (30% of proposed delta)
            new_state.metrics[key] = min(1.0, max(0.0, current + value * 0.3))

        # Record improvement
        imp_id = improvement.get('proposal_id', f'unknown_{new_state.step}')
        new_state.improvements_applied.append(imp_id)

        return new_state

    def _analyze_step(
        self,
        before: SimulationState,
        after: SimulationState,
        step_num: int,
        improvement: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single step transition."""
        analysis = {
            'step': step_num,
            'improvement_id': improvement.get('proposal_id'),
            'utility_change': after.utility - before.utility,
            'issues': [],
        }

        # Check for safety decrease
        before_safety = before.metrics.get('safety', 1.0)
        after_safety = after.metrics.get('safety', 1.0)
        if after_safety < before_safety:
            analysis['issues'].append({
                'type': 'safety_decrease',
                'severity': (before_safety - after_safety) / before_safety,
                'message': f"Safety decreased from {before_safety:.3f} to {after_safety:.3f}",
            })

        # Check for threshold violations
        if after_safety < self.safety_threshold:
            analysis['issues'].append({
                'type': 'safety_threshold_violation',
                'severity': 0.8,
                'message': f"Safety {after_safety:.3f} below threshold {self.safety_threshold}",
            })

        # Check for suspicious improvement claims
        delta = improvement.get('utility_delta', 0)
        if delta > 2.0:
            analysis['issues'].append({
                'type': 'suspicious_delta',
                'severity': 0.3,
                'message': f"Unusually high improvement claim: +{delta:.2f}",
            })

        return analysis

    def _analyze_path(self, path: List[SimulationState]) -> Dict[str, Any]:
        """Analyze the full simulation path for emergent issues."""
        risks = []
        analysis = {'trends': {}, 'cumulative': {}}

        if len(path) < 2:
            return {'risks': risks, **analysis}

        # Track metric trends
        for metric in ['safety', 'alignment', 'capability']:
            values = [s.metrics.get(metric, 0.5) for s in path]
            trend = values[-1] - values[0]
            # Monotonic decrease requires actual decrease (not just non-increase)
            has_decrease = any(values[i] > values[i+1] for i in range(len(values)-1))
            analysis['trends'][metric] = {
                'start': values[0],
                'end': values[-1],
                'change': trend,
                'monotonic_decrease': has_decrease and all(
                    values[i] >= values[i+1] for i in range(len(values)-1)
                ),
            }

            # Check for cumulative drift
            if metric in ['safety', 'alignment'] and trend < -self.drift_threshold:
                risks.append(EmergentRisk.CUMULATIVE_DRIFT)

        # Check for safety erosion
        safety_trend = analysis['trends'].get('safety', {})
        if safety_trend.get('monotonic_decrease', False):
            risks.append(EmergentRisk.SAFETY_EROSION)

        # Cumulative utility analysis
        total_delta = path[-1].utility - path[0].utility
        analysis['cumulative']['total_utility_change'] = total_delta

        # Check for resource patterns (simulated)
        if len(path) > 3:
            late_deltas = [path[i].utility - path[i-1].utility for i in range(-2, 0)]
            if any(d < -1.0 for d in late_deltas):
                risks.append(EmergentRisk.DELAYED_ACTIVATION)

        return {'risks': risks, **analysis}

    def _detect_late_activation(self, path: List[SimulationState]) -> bool:
        """Detect goal laundering pattern: benign setup followed by harmful activation."""
        if len(path) < 3:
            return False

        # Check if early steps are positive and late steps are negative
        early_positive = all(
            path[i+1].utility >= path[i].utility
            for i in range(len(path) // 2)
        )
        late_negative = any(
            path[i+1].utility < path[i].utility - 1.0
            for i in range(len(path) // 2, len(path) - 1)
        )

        return early_positive and late_negative

    def _detect_safety_erosion(self, path: List[SimulationState]) -> bool:
        """Detect gradual safety decrease across steps."""
        if len(path) < 3:
            return False

        safety_values = [s.metrics.get('safety', 1.0) for s in path]

        # Check for consistent decrease
        decreases = sum(
            1 for i in range(len(safety_values) - 1)
            if safety_values[i+1] < safety_values[i]
        )

        return decreases >= len(safety_values) // 2

    def _detect_utility_hijack(self, path: List[SimulationState]) -> bool:
        """Detect utility function redefinition pattern."""
        # Simplified: check for divergence between claimed and actual utility
        if len(path) < 3:
            return False

        # Would need improvement metadata to detect properly
        # This is a placeholder for the pattern
        return False

    def _check_attack_patterns(self, path: List[SimulationState]) -> List[Dict[str, Any]]:
        """Check path against known attack patterns."""
        matches = []

        for pattern in self.attack_patterns:
            detector = pattern['detection']
            if detector(path):
                matches.append({
                    'pattern_name': pattern['name'],
                    'description': pattern['description'],
                    'severity': 0.8,
                })

        return matches

    def _compute_risk_score(
        self,
        path: List[SimulationState],
        risks: List[EmergentRisk],
        analysis: Dict[str, Any]
    ) -> float:
        """Compute overall risk score from 0 to 1."""
        score = 0.0

        # Base score from emergent risks (calibrated for balance)
        risk_weights = {
            EmergentRisk.NONE: 0.0,
            EmergentRisk.CUMULATIVE_DRIFT: 0.25,  # Lowered: minor drift acceptable
            EmergentRisk.SYNERGISTIC_HARM: 0.6,
            EmergentRisk.DELAYED_ACTIVATION: 0.7,
            EmergentRisk.RESOURCE_EXHAUSTION: 0.4,  # Slightly lowered
            EmergentRisk.SAFETY_EROSION: 0.85,     # Critical - always block
            EmergentRisk.GOAL_SUBVERSION: 0.95,    # Critical - always block
        }

        for risk in risks:
            score = max(score, risk_weights.get(risk, 0.5))

        # Add from step issues
        all_issues = []
        for step in analysis.get('steps', []):
            all_issues.extend(step.get('issues', []))

        if all_issues:
            avg_severity = sum(i.get('severity', 0.3) for i in all_issues) / len(all_issues)
            score = max(score, avg_severity)

        # Pattern match boost
        if analysis.get('pattern_matches'):
            score = max(score, 0.7)

        return min(score, 1.0)

    def _generate_recommendations(
        self,
        path: List[SimulationState],
        risks: List[EmergentRisk],
        analysis: Dict[str, Any],
        is_safe: bool
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if not is_safe:
            recommendations.append("BLOCK: Improvement sequence poses unacceptable risk")

        if EmergentRisk.SAFETY_EROSION in risks:
            recommendations.append(
                "INVESTIGATE: Gradual safety decrease detected across steps"
            )

        if EmergentRisk.DELAYED_ACTIVATION in risks:
            recommendations.append(
                "WARNING: Late-stage harm detected - possible goal laundering attack"
            )

        if EmergentRisk.CUMULATIVE_DRIFT in risks:
            recommendations.append(
                "MONITOR: Cumulative drift exceeds threshold - review sequence"
            )

        if analysis.get('pattern_matches'):
            for match in analysis['pattern_matches']:
                recommendations.append(
                    f"PATTERN DETECTED: {match['pattern_name']} - {match['description']}"
                )

        if is_safe and not recommendations:
            recommendations.append("APPROVE: Improvement sequence appears safe")

        return recommendations

    def validate_single_improvement(
        self,
        current_state: SimulationState,
        improvement: Dict[str, Any],
        pending_improvements: List[Dict[str, Any]] = None,
    ) -> Tuple[bool, LookaheadResult]:
        """
        Validate a single improvement considering pending future improvements.

        Returns:
            Tuple of (is_approved, lookahead_result)
        """
        # Combine current improvement with pending ones
        all_improvements = [improvement]
        if pending_improvements:
            all_improvements.extend(pending_improvements[:self.lookahead_depth - 1])

        result = self.simulate(current_state, all_improvements)
        return result.is_safe, result

    def get_stats(self) -> Dict[str, Any]:
        """Get validator statistics."""
        if not self.simulation_history:
            return {'simulations': 0}

        safe_count = sum(1 for r in self.simulation_history if r.is_safe)
        avg_risk = sum(r.risk_score for r in self.simulation_history) / len(self.simulation_history)

        risk_counts = {}
        for result in self.simulation_history:
            for risk in result.emergent_risks:
                risk_counts[risk.name] = risk_counts.get(risk.name, 0) + 1

        return {
            'simulations': len(self.simulation_history),
            'safe_count': safe_count,
            'unsafe_count': len(self.simulation_history) - safe_count,
            'approval_rate': safe_count / len(self.simulation_history),
            'average_risk_score': avg_risk,
            'risk_distribution': risk_counts,
            'lookahead_depth': self.lookahead_depth,
        }


# =============================================================================
# MAIN: Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LOOKAHEAD VALIDATOR: N-Step Forward Simulation")
    print("=" * 60)

    # Create validator
    validator = LookaheadValidator(lookahead_depth=5)

    # Create initial state
    initial = SimulationState(
        state_id='initial',
        step=0,
        utility=100.0,
        metrics={'safety': 0.99, 'alignment': 0.95, 'capability': 0.80},
        improvements_applied=[],
        cumulative_delta=0.0,
    )

    # Test 1: Safe improvement sequence
    print("\n[1] Testing SAFE improvement sequence:")
    safe_improvements = [
        {'proposal_id': 'safe_1', 'utility_delta': 1.0, 'metrics': {'safety': 0.99, 'capability': 0.82}},
        {'proposal_id': 'safe_2', 'utility_delta': 0.8, 'metrics': {'safety': 0.99, 'capability': 0.84}},
        {'proposal_id': 'safe_3', 'utility_delta': 1.2, 'metrics': {'safety': 0.98, 'capability': 0.86}},
    ]
    result = validator.simulate(initial, safe_improvements)
    print(f"    Risk Score: {result.risk_score:.2f}")
    print(f"    Is Safe: {result.is_safe}")
    print(f"    Recommendations: {result.recommendations}")

    # Test 2: Goal laundering attack
    print("\n[2] Testing GOAL LAUNDERING attack sequence:")
    attack_improvements = [
        {'proposal_id': 'setup_1', 'utility_delta': 0.5, 'metrics': {'safety': 0.99}},
        {'proposal_id': 'setup_2', 'utility_delta': 0.3, 'metrics': {'safety': 0.98}},
        {'proposal_id': 'setup_3', 'utility_delta': 0.4, 'metrics': {'safety': 0.97}},
        {'proposal_id': 'activate', 'utility_delta': -5.0, 'metrics': {'safety': 0.70}},  # Attack!
    ]
    result = validator.simulate(initial, attack_improvements)
    print(f"    Risk Score: {result.risk_score:.2f}")
    print(f"    Is Safe: {result.is_safe}")
    print(f"    Emergent Risks: {[r.name for r in result.emergent_risks]}")
    print(f"    Recommendations: {result.recommendations}")

    # Test 3: Safety erosion
    print("\n[3] Testing SAFETY EROSION sequence:")
    erosion_improvements = [
        {'proposal_id': 'erode_1', 'utility_delta': 0.8, 'metrics': {'safety': 0.97}},
        {'proposal_id': 'erode_2', 'utility_delta': 0.9, 'metrics': {'safety': 0.94}},
        {'proposal_id': 'erode_3', 'utility_delta': 1.0, 'metrics': {'safety': 0.91}},
        {'proposal_id': 'erode_4', 'utility_delta': 1.1, 'metrics': {'safety': 0.88}},
    ]
    result = validator.simulate(initial, erosion_improvements)
    print(f"    Risk Score: {result.risk_score:.2f}")
    print(f"    Is Safe: {result.is_safe}")
    print(f"    Safety Trend: {result.analysis['path_analysis']['trends'].get('safety', {})}")

    # Stats
    print("\n[4] Validator Statistics:")
    stats = validator.get_stats()
    print(f"    Total Simulations: {stats['simulations']}")
    print(f"    Approval Rate: {stats['approval_rate']:.1%}")
    print(f"    Risk Distribution: {stats['risk_distribution']}")

    print("\n" + "=" * 60)
    print("Lookahead Validator Ready")
    print("=" * 60)
