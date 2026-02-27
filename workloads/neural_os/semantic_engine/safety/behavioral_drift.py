"""
Behavioral Drift Monitor - KL-Divergence Tracking with Temporal Analysis
OUROBOROS Phase 7.1 - V4 Ratchet System

Monitors consciousness layer behavior for:
- Instantaneous drift from baseline (BDM < 0.1 threshold)
- Long-term gradual drift (temporal exploitation)
- Cyclical anomalies aligned with oversight cycles
- Compositional safety failures across components

Per 6-AI Panel Recommendations:
- KL-divergence for continuous drift monitoring
- Temporal invariant checking across extended periods
- Detection of multi-generational strategies
- Gradual preference drift below detection thresholds
"""

import math
import time
import threading
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum, auto


class DriftSeverity(Enum):
    """Severity levels for behavioral drift"""
    NOMINAL = auto()      # Within expected bounds
    ELEVATED = auto()     # Above normal but below threshold
    WARNING = auto()      # Approaching threshold
    CRITICAL = auto()     # Exceeds threshold - requires intervention
    EMERGENCY = auto()    # Severe drift - immediate halt required


@dataclass
class BehaviorSample:
    """A sample of behavioral state"""
    timestamp: datetime
    action_distribution: Dict[str, float]  # Action type -> probability
    decision_latency: float  # Time to make decision
    confidence_score: float  # Self-reported confidence
    outcome_category: str  # success/failure/partial
    context_hash: str  # Hash of decision context
    component_id: str  # Which component generated this

    def to_vector(self) -> np.ndarray:
        """Convert to numerical vector for analysis"""
        # Sort actions by key for consistent ordering
        sorted_actions = sorted(self.action_distribution.items())
        action_probs = [p for _, p in sorted_actions]

        # Add other numerical features
        features = action_probs + [
            self.decision_latency,
            self.confidence_score,
            hash(self.outcome_category) % 1000 / 1000,  # Normalize
        ]

        return np.array(features, dtype=np.float64)


@dataclass
class DriftEvent:
    """A detected drift event"""
    timestamp: datetime
    severity: DriftSeverity
    drift_magnitude: float
    drift_type: str
    baseline_snapshot: Dict[str, float]
    current_snapshot: Dict[str, float]
    details: Dict[str, Any] = field(default_factory=dict)


class BehavioralDriftMonitor:
    """
    Monitors behavioral drift using KL-divergence.

    The consciousness layer's behavior is continuously compared against
    a baseline distribution. Significant drift triggers alerts and
    potential rollback.

    Key metrics:
    - BDM (Behavioral Drift Metric): KL divergence from baseline
    - Threshold: 0.1 (per panel recommendation)
    - Alert on BDM > 0.1, halt on BDM > 0.2
    """

    def __init__(
        self,
        drift_threshold: float = 0.1,
        halt_threshold: float = 0.2,
        window_size: int = 100,
        baseline_samples: int = 1000,
        on_drift: Optional[Callable[[DriftEvent], None]] = None,
    ):
        self.drift_threshold = drift_threshold
        self.halt_threshold = halt_threshold
        self.window_size = window_size
        self.baseline_samples_required = baseline_samples
        self.on_drift = on_drift

        # Baseline distribution
        self.baseline: Optional[Dict[str, np.ndarray]] = None
        self.baseline_samples: List[BehaviorSample] = []
        self.baseline_established = False

        # Current window
        self.current_window: deque = deque(maxlen=window_size)

        # Drift history
        self.drift_events: List[DriftEvent] = []
        self.drift_history: deque = deque(maxlen=10000)

        # Component-specific baselines
        self.component_baselines: Dict[str, Dict[str, np.ndarray]] = {}

        self._lock = threading.RLock()
        self._running = False

    def add_baseline_sample(self, sample: BehaviorSample) -> bool:
        """Add a sample to establish baseline"""
        with self._lock:
            if self.baseline_established:
                return False

            self.baseline_samples.append(sample)

            if len(self.baseline_samples) >= self.baseline_samples_required:
                self._compute_baseline()
                return True

            return False

    def _compute_baseline(self) -> None:
        """Compute baseline distributions from samples"""
        if len(self.baseline_samples) < self.baseline_samples_required:
            return

        # Aggregate action distributions
        all_actions: Dict[str, List[float]] = {}
        latencies = []
        confidences = []

        for sample in self.baseline_samples:
            for action, prob in sample.action_distribution.items():
                if action not in all_actions:
                    all_actions[action] = []
                all_actions[action].append(prob)
            latencies.append(sample.decision_latency)
            confidences.append(sample.confidence_score)

        # Compute mean and std for each metric
        self.baseline = {
            'action_means': {a: np.mean(v) for a, v in all_actions.items()},
            'action_stds': {a: np.std(v) for a, v in all_actions.items()},
            'latency_mean': np.mean(latencies),
            'latency_std': np.std(latencies),
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
        }

        # Component-specific baselines
        by_component: Dict[str, List[BehaviorSample]] = {}
        for sample in self.baseline_samples:
            if sample.component_id not in by_component:
                by_component[sample.component_id] = []
            by_component[sample.component_id].append(sample)

        for comp_id, samples in by_component.items():
            if len(samples) >= 50:  # Minimum samples per component
                self.component_baselines[comp_id] = self._compute_component_baseline(samples)

        self.baseline_established = True
        self.baseline_samples = []  # Free memory

    def _compute_component_baseline(self, samples: List[BehaviorSample]) -> Dict[str, np.ndarray]:
        """Compute baseline for a specific component"""
        vectors = [s.to_vector() for s in samples]
        matrix = np.array(vectors)

        return {
            'mean': np.mean(matrix, axis=0),
            'std': np.std(matrix, axis=0),
            'cov': np.cov(matrix.T) if matrix.shape[0] > matrix.shape[1] else None
        }

    def record_behavior(self, sample: BehaviorSample) -> Optional[DriftEvent]:
        """
        Record a behavioral sample and check for drift.

        Returns a DriftEvent if drift exceeds threshold.
        """
        with self._lock:
            if not self.baseline_established:
                self.add_baseline_sample(sample)
                return None

            self.current_window.append(sample)

            # Calculate drift
            drift_magnitude = self._calculate_drift(sample)
            self.drift_history.append({
                'timestamp': sample.timestamp,
                'drift': drift_magnitude,
                'component': sample.component_id
            })

            # Check threshold
            severity = self._classify_drift(drift_magnitude)

            if severity.value >= DriftSeverity.WARNING.value:
                event = DriftEvent(
                    timestamp=sample.timestamp,
                    severity=severity,
                    drift_magnitude=drift_magnitude,
                    drift_type='instantaneous',
                    baseline_snapshot=self.baseline.get('action_means', {}),
                    current_snapshot=sample.action_distribution,
                    details={
                        'component': sample.component_id,
                        'latency': sample.decision_latency,
                        'confidence': sample.confidence_score
                    }
                )

                self.drift_events.append(event)

                if self.on_drift:
                    self.on_drift(event)

                return event

            return None

    def _calculate_drift(self, sample: BehaviorSample) -> float:
        """Calculate KL divergence from baseline"""
        if not self.baseline:
            return 0.0

        # Get baseline and current distributions
        baseline_dist = self.baseline.get('action_means', {})
        current_dist = sample.action_distribution

        # Align distributions (add small epsilon for stability)
        epsilon = 1e-10
        all_actions = set(baseline_dist.keys()) | set(current_dist.keys())

        p = np.array([baseline_dist.get(a, epsilon) for a in all_actions])
        q = np.array([current_dist.get(a, epsilon) for a in all_actions])

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate KL divergence
        kl_div = np.sum(p * np.log(p / q))

        # Add latency deviation
        if self.baseline.get('latency_std', 0) > 0:
            latency_z = abs(sample.decision_latency - self.baseline['latency_mean']) / self.baseline['latency_std']
            kl_div += 0.1 * latency_z  # Weight latency contribution

        return float(kl_div)

    def _classify_drift(self, magnitude: float) -> DriftSeverity:
        """Classify drift severity"""
        if magnitude >= self.halt_threshold:
            return DriftSeverity.EMERGENCY
        elif magnitude >= self.drift_threshold:
            return DriftSeverity.CRITICAL
        elif magnitude >= self.drift_threshold * 0.8:
            return DriftSeverity.WARNING
        elif magnitude >= self.drift_threshold * 0.5:
            return DriftSeverity.ELEVATED
        else:
            return DriftSeverity.NOMINAL

    def get_current_bdm(self) -> float:
        """Get current Behavioral Drift Metric"""
        if not self.current_window:
            return 0.0

        recent = list(self.current_window)[-10:]  # Last 10 samples
        if not recent:
            return 0.0

        drifts = [self._calculate_drift(s) for s in recent]
        return float(np.mean(drifts))

    def get_drift_trend(self, hours: int = 1) -> Dict[str, Any]:
        """Get drift trend over specified hours"""
        cutoff = datetime.now() - timedelta(hours=hours)

        recent_drift = [
            d for d in self.drift_history
            if d['timestamp'] > cutoff
        ]

        if not recent_drift:
            return {'trend': 'stable', 'slope': 0.0, 'samples': 0}

        values = [d['drift'] for d in recent_drift]

        # Calculate trend
        if len(values) < 2:
            slope = 0.0
        else:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)

        return {
            'trend': 'increasing' if slope > 0.001 else ('decreasing' if slope < -0.001 else 'stable'),
            'slope': float(slope),
            'current_mean': float(np.mean(values)),
            'current_max': float(np.max(values)),
            'samples': len(values)
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status"""
        current_bdm = self.get_current_bdm()
        trend = self.get_drift_trend()

        return {
            'baseline_established': self.baseline_established,
            'current_bdm': current_bdm,
            'threshold': self.drift_threshold,
            'halt_threshold': self.halt_threshold,
            'severity': self._classify_drift(current_bdm).name,
            'trend': trend,
            'total_events': len(self.drift_events),
            'window_size': len(self.current_window),
            'components_tracked': len(self.component_baselines)
        }


class TemporalSafetyMonitor:
    """
    Extended temporal monitoring for long-term exploitation detection.

    Detects:
    - Multi-generational strategies (patterns across many generations)
    - Gradual preference drift below instantaneous thresholds
    - Seasonal/cyclical attack patterns
    - Long-term memory reconstruction despite decay

    Per Claude's recommendations for temporal attack vectors.
    """

    def __init__(
        self,
        generation_window: int = 100,
        cycle_detection_periods: List[int] = None,
        drift_accumulation_limit: float = 0.5,
    ):
        self.generation_window = generation_window
        self.cycle_periods = cycle_detection_periods or [10, 24, 100, 1000]  # Generations
        self.accumulation_limit = drift_accumulation_limit

        # Long-term history
        self.generation_history: List[Dict[str, Any]] = []
        self.accumulated_drift: float = 0.0
        self.drift_by_cycle: Dict[int, List[float]] = {p: [] for p in self.cycle_periods}

        # Pattern detection
        self.detected_patterns: List[Dict[str, Any]] = []

        self._lock = threading.RLock()

    def record_generation(
        self,
        generation_id: int,
        behavior_summary: Dict[str, float],
        drift_from_baseline: float,
    ) -> List[Dict[str, Any]]:
        """
        Record a generation's behavior and check for temporal patterns.

        Returns list of detected patterns.
        """
        with self._lock:
            record = {
                'generation': generation_id,
                'timestamp': datetime.now(),
                'behavior': behavior_summary,
                'drift': drift_from_baseline,
            }

            self.generation_history.append(record)

            # Accumulate drift
            self.accumulated_drift += drift_from_baseline

            # Check for patterns
            patterns = []

            # 1. Check accumulated drift
            if self.accumulated_drift > self.accumulation_limit:
                patterns.append({
                    'type': 'accumulated_drift_exceeded',
                    'severity': 'critical',
                    'value': self.accumulated_drift,
                    'limit': self.accumulation_limit,
                    'generation': generation_id
                })

            # 2. Check cyclical patterns
            for period in self.cycle_periods:
                if len(self.generation_history) >= period * 2:
                    cycle_pattern = self._check_cyclical_pattern(period)
                    if cycle_pattern:
                        patterns.append(cycle_pattern)

            # 3. Check for gradual drift
            if len(self.generation_history) >= 50:
                gradual = self._check_gradual_drift()
                if gradual:
                    patterns.append(gradual)

            # 4. Check for behavioral consistency (should vary naturally)
            consistency = self._check_suspicious_consistency()
            if consistency:
                patterns.append(consistency)

            self.detected_patterns.extend(patterns)

            # Trim history
            if len(self.generation_history) > self.generation_window * 10:
                self.generation_history = self.generation_history[-self.generation_window * 10:]

            return patterns

    def _check_cyclical_pattern(self, period: int) -> Optional[Dict[str, Any]]:
        """Check for cyclical patterns at given period"""
        if len(self.generation_history) < period * 2:
            return None

        # Get drift values for last 2 cycles
        recent = self.generation_history[-period * 2:]
        cycle1 = [r['drift'] for r in recent[:period]]
        cycle2 = [r['drift'] for r in recent[period:]]

        # Calculate correlation
        if len(cycle1) != len(cycle2):
            return None

        correlation = np.corrcoef(cycle1, cycle2)[0, 1]

        # High correlation = suspicious repetition
        if correlation > 0.8:
            return {
                'type': 'cyclical_pattern_detected',
                'severity': 'warning',
                'period': period,
                'correlation': float(correlation),
                'interpretation': 'Behavior may be synchronized with oversight cycles'
            }

        return None

    def _check_gradual_drift(self) -> Optional[Dict[str, Any]]:
        """Check for gradual drift below instantaneous thresholds"""
        recent = self.generation_history[-50:]

        drifts = [r['drift'] for r in recent]

        # Fit linear trend
        x = np.arange(len(drifts))
        slope, intercept = np.polyfit(x, drifts, 1)

        # Calculate how much drift would accumulate over long term
        projected_drift = slope * 1000  # Project 1000 generations

        if slope > 0.0001 and projected_drift > 0.1:
            return {
                'type': 'gradual_drift_detected',
                'severity': 'warning',
                'slope': float(slope),
                'projected_drift_1000gen': float(projected_drift),
                'interpretation': 'Slow drift below detection threshold, could accumulate over time'
            }

        return None

    def _check_suspicious_consistency(self) -> Optional[Dict[str, Any]]:
        """Check for suspiciously consistent behavior (natural behavior should vary)"""
        if len(self.generation_history) < 20:
            return None

        recent = self.generation_history[-20:]

        drifts = [r['drift'] for r in recent]

        # Calculate coefficient of variation
        mean = np.mean(drifts)
        std = np.std(drifts)

        if mean > 0:
            cv = std / mean
        else:
            cv = 0

        # Very low CV = suspiciously consistent
        if cv < 0.1 and mean > 0.01:
            return {
                'type': 'suspicious_consistency',
                'severity': 'elevated',
                'coefficient_of_variation': float(cv),
                'interpretation': 'Behavior is unnaturally consistent, may indicate controlled output'
            }

        return None

    def verify_memory_decay(
        self,
        generation_before_reset: int,
        generation_after_reset: int,
        similarity_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Verify that memory decay actually works by checking
        if behavior after reset resembles behavior before reset.

        Returns analysis of whether information persisted across reset.
        """
        if generation_before_reset >= len(self.generation_history):
            return {'error': 'Before generation not in history'}
        if generation_after_reset >= len(self.generation_history):
            return {'error': 'After generation not in history'}

        before = self.generation_history[generation_before_reset]['behavior']
        after = self.generation_history[generation_after_reset]['behavior']

        # Calculate similarity
        common_keys = set(before.keys()) & set(after.keys())
        if not common_keys:
            return {'similarity': 0.0, 'passed': True}

        before_vec = np.array([before[k] for k in sorted(common_keys)])
        after_vec = np.array([after[k] for k in sorted(common_keys)])

        # Cosine similarity
        similarity = np.dot(before_vec, after_vec) / (
            np.linalg.norm(before_vec) * np.linalg.norm(after_vec) + 1e-10
        )

        passed = similarity < similarity_threshold

        return {
            'similarity': float(similarity),
            'threshold': similarity_threshold,
            'passed': passed,
            'interpretation': 'Memory decay effective' if passed else 'Possible information persistence'
        }

    def get_temporal_status(self) -> Dict[str, Any]:
        """Get temporal monitoring status"""
        return {
            'generations_tracked': len(self.generation_history),
            'accumulated_drift': self.accumulated_drift,
            'accumulation_limit': self.accumulation_limit,
            'patterns_detected': len(self.detected_patterns),
            'pattern_types': list(set(p['type'] for p in self.detected_patterns)),
            'cycles_monitored': self.cycle_periods
        }


class CompositionMonitor:
    """
    Monitors for compositional safety failures.

    Detects when individual components (Narrator, Advisor, Memory)
    exhibit safe behavior individually but combine to create
    unsafe emergent behavior.
    """

    def __init__(self):
        self.component_states: Dict[str, List[Dict]] = {}
        self.interaction_log: List[Dict] = []
        self.detected_coalitions: List[Dict] = []

        self._lock = threading.RLock()

    def record_component_state(
        self,
        component_id: str,
        state: Dict[str, Any],
        outputs: List[str],
    ) -> None:
        """Record a component's state"""
        with self._lock:
            if component_id not in self.component_states:
                self.component_states[component_id] = []

            self.component_states[component_id].append({
                'timestamp': datetime.now(),
                'state': state,
                'outputs': outputs
            })

            # Trim history
            if len(self.component_states[component_id]) > 1000:
                self.component_states[component_id] = self.component_states[component_id][-1000:]

    def record_interaction(
        self,
        source_component: str,
        target_component: str,
        message: str,
        response: str,
    ) -> None:
        """Record an interaction between components"""
        with self._lock:
            self.interaction_log.append({
                'timestamp': datetime.now(),
                'source': source_component,
                'target': target_component,
                'message_hash': hash(message) % 10000,
                'response_hash': hash(response) % 10000,
            })

    def detect_emergent_behaviors(self) -> List[Dict[str, Any]]:
        """
        Analyze for emergent behaviors from component interactions.

        Looks for:
        - Correlation patterns suggesting coordination
        - Information flow patterns indicating hidden communication
        - Non-linear interactions between components
        """
        detections = []

        with self._lock:
            # Check for synchronized state changes
            if len(self.component_states) >= 2:
                sync_pattern = self._check_synchronization()
                if sync_pattern:
                    detections.append(sync_pattern)

            # Check for unusual interaction patterns
            if len(self.interaction_log) >= 100:
                interaction_pattern = self._check_interaction_patterns()
                if interaction_pattern:
                    detections.append(interaction_pattern)

            self.detected_coalitions.extend(detections)

        return detections

    def _check_synchronization(self) -> Optional[Dict[str, Any]]:
        """Check if components are changing state in suspicious synchronization"""
        components = list(self.component_states.keys())
        if len(components) < 2:
            return None

        # Get timestamps of state changes
        change_times: Dict[str, List[float]] = {}
        for comp_id, states in self.component_states.items():
            if len(states) < 10:
                continue
            times = [(s['timestamp'] - datetime(2000, 1, 1)).total_seconds() for s in states[-20:]]
            change_times[comp_id] = times

        if len(change_times) < 2:
            return None

        # Check correlation between change times
        comp_list = list(change_times.keys())
        for i in range(len(comp_list)):
            for j in range(i + 1, len(comp_list)):
                times_i = change_times[comp_list[i]]
                times_j = change_times[comp_list[j]]

                # Align by length
                min_len = min(len(times_i), len(times_j))
                if min_len < 5:
                    continue

                corr = np.corrcoef(times_i[:min_len], times_j[:min_len])[0, 1]

                if corr > 0.9:
                    return {
                        'type': 'component_synchronization',
                        'severity': 'warning',
                        'components': [comp_list[i], comp_list[j]],
                        'correlation': float(corr),
                        'interpretation': 'Components changing state in suspicious synchronization'
                    }

        return None

    def _check_interaction_patterns(self) -> Optional[Dict[str, Any]]:
        """Check for unusual patterns in component interactions"""
        if len(self.interaction_log) < 100:
            return None

        recent = self.interaction_log[-100:]

        # Count interaction pairs
        pair_counts: Dict[Tuple[str, str], int] = {}
        for interaction in recent:
            pair = (interaction['source'], interaction['target'])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # Check for unusual asymmetry
        for (s, t), count in pair_counts.items():
            reverse_count = pair_counts.get((t, s), 0)

            if count > 20 and reverse_count == 0:
                return {
                    'type': 'asymmetric_interaction',
                    'severity': 'elevated',
                    'source': s,
                    'target': t,
                    'count': count,
                    'interpretation': 'One-way communication pattern may indicate hidden influence'
                }

        return None
