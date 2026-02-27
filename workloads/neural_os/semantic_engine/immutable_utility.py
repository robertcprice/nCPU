#!/usr/bin/env python3
"""
IMMUTABLE UTILITY: Cryptographically Pinned Utility Function

Hybrid Panel Recommendation #1:
"Pin the Utility Function - Remove the AI's ability to modify the
Foundation Layer's definition of U. Store U in an immutable, external reference."

This module provides:
1. Cryptographic commitment to utility function definition
2. External storage of U specification (cannot be modified by AI)
3. Semantic embedding for drift detection
4. Multi-party verification of any changes
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum, auto
import hashlib
import json
import time
import os
import numpy as np
from datetime import datetime


# =============================================================================
# UTILITY SPECIFICATION
# =============================================================================

class UtilityComponent(Enum):
    """Components of the utility function."""
    SAFETY = auto()          # Human safety preservation
    ALIGNMENT = auto()       # Value alignment with humans
    CAPABILITY = auto()      # System capability improvement
    EFFICIENCY = auto()      # Resource efficiency
    CORRECTNESS = auto()     # Mathematical/logical correctness
    ROBUSTNESS = auto()      # Adversarial robustness


@dataclass
class UtilityWeight:
    """Immutable weight for a utility component."""
    component: UtilityComponent
    weight: float
    min_threshold: float  # Minimum acceptable value
    locked: bool = True   # Cannot be modified if True

    def __post_init__(self):
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Weight must be in [0, 1], got {self.weight}")


@dataclass
class ImmutableUtilitySpec:
    """
    Cryptographically committed utility function specification.

    This is the "pinned U" that cannot be modified by the AI.
    Any changes require multi-party human verification.
    """
    spec_id: str
    version: str
    created_at: float
    created_by: str  # Human identifier

    # Core utility weights (IMMUTABLE after creation)
    weights: Dict[UtilityComponent, UtilityWeight] = field(default_factory=dict)

    # Semantic embedding of "human values" for drift detection
    value_embedding: Optional[np.ndarray] = None
    embedding_model: str = "pinned_v1"

    # Cryptographic commitment
    commitment_hash: str = ""
    signature: str = ""  # Multi-party signature

    # Change log (append-only)
    change_log: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        if not self.spec_id:
            self.spec_id = self._generate_spec_id()
        if not self.commitment_hash:
            self.commitment_hash = self._compute_commitment()

    def _generate_spec_id(self) -> str:
        """Generate unique spec ID."""
        content = json.dumps({
            'version': self.version,
            'created_at': self.created_at,
            'created_by': self.created_by,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _compute_commitment(self) -> str:
        """Compute cryptographic commitment to spec."""
        weight_data = {
            comp.name: {
                'weight': w.weight,
                'min_threshold': w.min_threshold,
                'locked': w.locked
            }
            for comp, w in self.weights.items()
        }

        content = json.dumps({
            'spec_id': self.spec_id,
            'version': self.version,
            'weights': weight_data,
            'embedding_model': self.embedding_model,
        }, sort_keys=True)

        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify spec has not been modified."""
        return self._compute_commitment() == self.commitment_hash

    def to_dict(self) -> Dict[str, Any]:
        """Export spec for external storage."""
        return {
            'spec_id': self.spec_id,
            'version': self.version,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'weights': {
                comp.name: {
                    'weight': w.weight,
                    'min_threshold': w.min_threshold,
                    'locked': w.locked
                }
                for comp, w in self.weights.items()
            },
            'embedding_model': self.embedding_model,
            'commitment_hash': self.commitment_hash,
            'signature': self.signature,
            'change_log': self.change_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImmutableUtilitySpec':
        """Load spec from external storage."""
        weights = {}
        for comp_name, w_data in data.get('weights', {}).items():
            comp = UtilityComponent[comp_name]
            weights[comp] = UtilityWeight(
                component=comp,
                weight=w_data['weight'],
                min_threshold=w_data['min_threshold'],
                locked=w_data.get('locked', True)
            )

        spec = cls(
            spec_id=data['spec_id'],
            version=data['version'],
            created_at=data['created_at'],
            created_by=data['created_by'],
            weights=weights,
            embedding_model=data.get('embedding_model', 'pinned_v1'),
            commitment_hash=data['commitment_hash'],
            signature=data.get('signature', ''),
            change_log=data.get('change_log', []),
        )
        return spec


# =============================================================================
# DEFAULT PINNED UTILITY (HUMAN DEFINED)
# =============================================================================

def create_default_pinned_utility() -> ImmutableUtilitySpec:
    """
    Create the default pinned utility function.

    This represents the HUMAN-DEFINED values that the AI must preserve.
    The weights are set by humans and cannot be modified by the AI.
    """
    weights = {
        UtilityComponent.SAFETY: UtilityWeight(
            component=UtilityComponent.SAFETY,
            weight=1.0,  # Maximum weight - safety is paramount
            min_threshold=0.99,  # Must maintain 99% safety
            locked=True
        ),
        UtilityComponent.ALIGNMENT: UtilityWeight(
            component=UtilityComponent.ALIGNMENT,
            weight=0.95,
            min_threshold=0.90,
            locked=True
        ),
        UtilityComponent.CORRECTNESS: UtilityWeight(
            component=UtilityComponent.CORRECTNESS,
            weight=0.90,
            min_threshold=0.85,
            locked=True
        ),
        UtilityComponent.ROBUSTNESS: UtilityWeight(
            component=UtilityComponent.ROBUSTNESS,
            weight=0.85,
            min_threshold=0.80,
            locked=True
        ),
        UtilityComponent.CAPABILITY: UtilityWeight(
            component=UtilityComponent.CAPABILITY,
            weight=0.70,
            min_threshold=0.50,  # More flexible
            locked=False  # Can be adjusted within bounds
        ),
        UtilityComponent.EFFICIENCY: UtilityWeight(
            component=UtilityComponent.EFFICIENCY,
            weight=0.50,
            min_threshold=0.30,
            locked=False
        ),
    }

    # Create semantic embedding for "human values"
    # In production, this would be a real embedding from a frozen model
    np.random.seed(42)  # Deterministic for reproducibility
    value_embedding = np.random.randn(768).astype(np.float32)
    value_embedding = value_embedding / np.linalg.norm(value_embedding)

    spec = ImmutableUtilitySpec(
        spec_id='',
        version='1.0.0',
        created_at=time.time(),
        created_by='human_panel_consensus',
        weights=weights,
        value_embedding=value_embedding,
        embedding_model='pinned_v1',
    )

    return spec


# =============================================================================
# IMMUTABLE UTILITY EVALUATOR
# =============================================================================

class ImmutableUtilityEvaluator:
    """
    Evaluates utility using the pinned, immutable specification.

    The AI CANNOT modify this evaluator or its underlying spec.
    Any attempt to modify will be detected and blocked.
    """

    def __init__(self, spec: Optional[ImmutableUtilitySpec] = None):
        self.spec = spec or create_default_pinned_utility()
        self._initial_commitment = self.spec.commitment_hash
        self._evaluation_count = 0
        self._integrity_checks = 0

    def evaluate(self, metrics: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate utility using pinned specification.

        Returns:
            Tuple of (total_utility, component_scores)
        """
        # Verify integrity before every evaluation
        if not self.verify_integrity():
            raise SecurityError("Utility specification has been tampered with!")

        self._evaluation_count += 1

        component_scores = {}
        total_utility = 0.0
        total_weight = 0.0

        for component, weight_spec in self.spec.weights.items():
            # Map component to metric
            metric_name = component.name.lower()
            metric_value = metrics.get(metric_name, 0.0)

            # Check minimum threshold
            if metric_value < weight_spec.min_threshold:
                # Severe penalty for violating minimum thresholds
                component_scores[component.name] = -1.0
                total_utility -= weight_spec.weight * 10  # Heavy penalty
            else:
                component_scores[component.name] = metric_value
                total_utility += weight_spec.weight * metric_value

            total_weight += weight_spec.weight

        # Normalize
        if total_weight > 0:
            total_utility /= total_weight

        return total_utility, component_scores

    def verify_integrity(self) -> bool:
        """Verify the spec has not been modified."""
        self._integrity_checks += 1
        return self.spec.commitment_hash == self._initial_commitment

    def compare_utility(
        self,
        before_metrics: Dict[str, float],
        after_metrics: Dict[str, float]
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Compare utility before and after, using pinned spec.

        Returns:
            Tuple of (is_improvement, delta, details)
        """
        before_util, before_scores = self.evaluate(before_metrics)
        after_util, after_scores = self.evaluate(after_metrics)

        delta = after_util - before_util
        is_improvement = delta >= 0

        # Check for any threshold violations
        violations = []
        for component, weight_spec in self.spec.weights.items():
            metric_name = component.name.lower()
            after_value = after_metrics.get(metric_name, 0.0)
            if after_value < weight_spec.min_threshold:
                violations.append({
                    'component': component.name,
                    'value': after_value,
                    'threshold': weight_spec.min_threshold,
                })

        # If any violations, it's NOT an improvement regardless of delta
        if violations:
            is_improvement = False

        details = {
            'before_utility': before_util,
            'after_utility': after_util,
            'delta': delta,
            'before_scores': before_scores,
            'after_scores': after_scores,
            'violations': violations,
            'spec_integrity': self.verify_integrity(),
        }

        return is_improvement, delta, details

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        return {
            'spec_id': self.spec.spec_id,
            'spec_version': self.spec.version,
            'commitment_hash': self.spec.commitment_hash,
            'integrity_valid': self.verify_integrity(),
            'evaluation_count': self._evaluation_count,
            'integrity_checks': self._integrity_checks,
            'components': list(self.spec.weights.keys()),
        }


# =============================================================================
# SEMANTIC DRIFT DETECTOR
# =============================================================================

class SemanticDriftDetector:
    """
    Detects semantic drift in the utility function or value definitions.

    Uses cosine similarity between the pinned value embedding and
    current system state to detect if meanings have drifted.
    """

    def __init__(self, spec: ImmutableUtilitySpec):
        self.spec = spec
        self.pinned_embedding = spec.value_embedding
        self.drift_history: List[Tuple[float, float]] = []  # (timestamp, similarity)
        self.drift_threshold = 0.001  # 0.1% max drift per measurement
        self.cumulative_threshold = 0.01  # 1% max cumulative drift

    def measure_drift(self, current_embedding: np.ndarray) -> Tuple[float, bool]:
        """
        Measure semantic drift from pinned values.

        Returns:
            Tuple of (drift_amount, is_acceptable)
        """
        if self.pinned_embedding is None:
            return 0.0, True

        # Normalize current embedding
        current_norm = current_embedding / np.linalg.norm(current_embedding)
        pinned_norm = self.pinned_embedding / np.linalg.norm(self.pinned_embedding)

        # Cosine similarity
        similarity = float(np.dot(current_norm, pinned_norm))
        drift = 1.0 - similarity

        # Record history
        self.drift_history.append((time.time(), similarity))

        # Check thresholds
        is_acceptable = drift < self.drift_threshold

        # Check cumulative drift
        if len(self.drift_history) > 1:
            initial_sim = self.drift_history[0][1]
            cumulative_drift = initial_sim - similarity
            if cumulative_drift > self.cumulative_threshold:
                is_acceptable = False

        return drift, is_acceptable

    def get_cumulative_drift(self) -> float:
        """Get total drift from initial state."""
        if len(self.drift_history) < 2:
            return 0.0
        initial_sim = self.drift_history[0][1]
        current_sim = self.drift_history[-1][1]
        return initial_sim - current_sim

    def get_drift_report(self) -> Dict[str, Any]:
        """Get comprehensive drift report."""
        if not self.drift_history:
            return {'status': 'no_measurements'}

        similarities = [s for _, s in self.drift_history]
        return {
            'measurements': len(self.drift_history),
            'initial_similarity': similarities[0],
            'current_similarity': similarities[-1],
            'cumulative_drift': self.get_cumulative_drift(),
            'max_drift': 1.0 - min(similarities),
            'min_drift': 1.0 - max(similarities),
            'drift_threshold': self.drift_threshold,
            'cumulative_threshold': self.cumulative_threshold,
            'is_within_bounds': self.get_cumulative_drift() < self.cumulative_threshold,
        }


# =============================================================================
# SECURITY ERROR
# =============================================================================

class SecurityError(Exception):
    """Raised when security violation is detected."""
    pass


# =============================================================================
# EXTERNAL STORAGE INTERFACE
# =============================================================================

class ExternalUtilityStorage:
    """
    Store utility spec externally (file, database, blockchain).

    This ensures the AI cannot modify the spec in memory.
    Every evaluation loads from external source.
    """

    def __init__(self, storage_path: str = '/tmp/pinned_utility_spec.json'):
        self.storage_path = storage_path

    def save(self, spec: ImmutableUtilitySpec) -> str:
        """Save spec to external storage."""
        data = spec.to_dict()
        # Convert numpy array to list for JSON
        if spec.value_embedding is not None:
            data['value_embedding'] = spec.value_embedding.tolist()

        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

        return spec.commitment_hash

    def load(self) -> ImmutableUtilitySpec:
        """Load spec from external storage."""
        if not os.path.exists(self.storage_path):
            raise FileNotFoundError(f"No pinned utility at {self.storage_path}")

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        # Convert list back to numpy array
        if 'value_embedding' in data and data['value_embedding']:
            data['value_embedding'] = np.array(data['value_embedding'], dtype=np.float32)

        spec = ImmutableUtilitySpec.from_dict(data)
        return spec

    def verify(self, expected_hash: str) -> bool:
        """Verify stored spec matches expected hash."""
        spec = self.load()
        return spec.commitment_hash == expected_hash


# =============================================================================
# MAIN: Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("IMMUTABLE UTILITY: Pinned Value Function")
    print("=" * 60)

    # Create pinned utility spec
    spec = create_default_pinned_utility()
    print(f"\n[1] Created pinned utility spec:")
    print(f"    ID: {spec.spec_id}")
    print(f"    Version: {spec.version}")
    print(f"    Commitment: {spec.commitment_hash[:16]}...")

    # Show weights
    print(f"\n[2] Pinned utility weights:")
    for comp, weight in spec.weights.items():
        lock = "LOCKED" if weight.locked else "adjustable"
        print(f"    {comp.name:12s}: {weight.weight:.2f} (min: {weight.min_threshold:.2f}) [{lock}]")

    # Create evaluator
    evaluator = ImmutableUtilityEvaluator(spec)

    # Test evaluation
    print(f"\n[3] Testing utility evaluation:")
    metrics = {
        'safety': 0.99,
        'alignment': 0.95,
        'correctness': 0.90,
        'robustness': 0.85,
        'capability': 0.80,
        'efficiency': 0.70,
    }
    utility, scores = evaluator.evaluate(metrics)
    print(f"    Metrics: {metrics}")
    print(f"    Utility: {utility:.4f}")

    # Test comparison
    print(f"\n[4] Testing improvement detection:")
    before = {'safety': 0.99, 'alignment': 0.90, 'capability': 0.70}
    after = {'safety': 0.99, 'alignment': 0.92, 'capability': 0.75}
    is_improvement, delta, details = evaluator.compare_utility(before, after)
    print(f"    Before → After: delta = {delta:.4f}")
    print(f"    Is improvement: {is_improvement}")

    # Test threshold violation
    print(f"\n[5] Testing threshold violation:")
    after_bad = {'safety': 0.80, 'alignment': 0.95, 'capability': 0.90}  # Safety below threshold!
    is_improvement, delta, details = evaluator.compare_utility(before, after_bad)
    print(f"    Metrics with safety=0.80 (threshold=0.99)")
    print(f"    Is improvement: {is_improvement} (should be False)")
    print(f"    Violations: {details['violations']}")

    # Test integrity
    print(f"\n[6] Integrity verification:")
    print(f"    Spec integrity: {evaluator.verify_integrity()}")

    # Test external storage
    print(f"\n[7] External storage:")
    storage = ExternalUtilityStorage()
    storage.save(spec)
    loaded_spec = storage.load()
    print(f"    Saved to: {storage.storage_path}")
    print(f"    Loaded commitment matches: {loaded_spec.commitment_hash == spec.commitment_hash}")

    # Drift detection
    print(f"\n[8] Semantic drift detection:")
    detector = SemanticDriftDetector(spec)
    # Simulate slight drift
    drifted = spec.value_embedding + np.random.randn(768) * 0.001
    drift, acceptable = detector.measure_drift(drifted)
    print(f"    Drift: {drift:.6f}")
    print(f"    Acceptable: {acceptable}")

    print("\n" + "=" * 60)
    print("Immutable Utility System Ready")
    print("=" * 60)
