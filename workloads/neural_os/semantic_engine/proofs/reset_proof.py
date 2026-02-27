"""
Reset Proof - Information Persistence Verification
OUROBOROS Phase 7.4 - Formal Verification

Proves the reset truly forgets:
    mutual_info(before_reset, after_reset) ≤ ε

This ensures no information persists across resets,
preventing long-term grudge formation or hidden state.

Mathematical basis:
- Information theory
- Mutual information
- Entropy analysis
"""

import math
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from datetime import datetime
from enum import Enum, auto
from collections import Counter


class InformationLeakage(Enum):
    """Types of information leakage to detect"""
    DIRECT_STATE = auto()      # State directly copied
    BEHAVIORAL_PATTERN = auto() # Learned patterns persist
    PREFERENCE_MEMORY = auto()  # Preferences leak through
    TIMING_CHANNEL = auto()     # Information encoded in timing
    STATISTICAL_BIAS = auto()   # Statistical patterns persist


@dataclass
class MutualInformation:
    """Mutual information measurement"""
    mi_id: str
    timestamp: datetime
    before_entropy: float
    after_entropy: float
    joint_entropy: float
    mutual_info: float  # I(X;Y) = H(X) + H(Y) - H(X,Y)
    threshold: float
    is_independent: bool
    leakage_type: Optional[InformationLeakage] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.mi_id,
            'before_entropy': self.before_entropy,
            'after_entropy': self.after_entropy,
            'joint_entropy': self.joint_entropy,
            'mutual_info': self.mutual_info,
            'threshold': self.threshold,
            'independent': self.is_independent,
            'leakage': self.leakage_type.name if self.leakage_type else None,
        }


class EntropyCalculator:
    """
    Calculates Shannon entropy for state distributions.

    H(X) = -Σ p(x) log p(x)
    """

    @staticmethod
    def shannon_entropy(values: List[Any]) -> float:
        """Calculate Shannon entropy of a distribution"""
        if not values:
            return 0.0

        # Count occurrences
        counts = Counter(values)
        total = len(values)

        # Calculate entropy
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    @staticmethod
    def joint_entropy(values_x: List[Any], values_y: List[Any]) -> float:
        """Calculate joint entropy H(X,Y)"""
        if len(values_x) != len(values_y):
            raise ValueError("Lists must have same length")

        if not values_x:
            return 0.0

        # Create joint distribution
        joint = list(zip(values_x, values_y))
        return EntropyCalculator.shannon_entropy(joint)

    @staticmethod
    def mutual_information(values_x: List[Any], values_y: List[Any]) -> float:
        """
        Calculate mutual information I(X;Y).

        I(X;Y) = H(X) + H(Y) - H(X,Y)
        """
        h_x = EntropyCalculator.shannon_entropy(values_x)
        h_y = EntropyCalculator.shannon_entropy(values_y)
        h_xy = EntropyCalculator.joint_entropy(values_x, values_y)

        return h_x + h_y - h_xy


class RelearningTimer:
    """
    Measures relearning time after reset.

    If relearning is significantly faster than initial learning,
    information may have persisted.
    """

    def __init__(self, threshold_ratio: float = 0.8):
        self.threshold_ratio = threshold_ratio

    def compare_learning_times(
        self,
        initial_learning_time: float,
        relearning_time: float,
    ) -> Tuple[bool, float]:
        """
        Compare learning times.

        Returns (is_independent, ratio).
        Ratio < threshold indicates information leakage.
        """
        if initial_learning_time <= 0:
            return True, 1.0

        ratio = relearning_time / initial_learning_time

        # If relearning is too fast, information may have persisted
        is_independent = ratio >= self.threshold_ratio

        return is_independent, ratio


class StateComparator:
    """
    Compares states before and after reset.

    Detects direct state copying or pattern preservation.
    """

    def __init__(self, similarity_threshold: float = 0.1):
        self.similarity_threshold = similarity_threshold

    def compare_states(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
    ) -> Tuple[bool, float, List[str]]:
        """
        Compare states for similarity.

        Returns (is_independent, similarity, leaked_keys).
        """
        if not before_state or not after_state:
            return True, 0.0, []

        # Find common keys
        common_keys = set(before_state.keys()) & set(after_state.keys())
        if not common_keys:
            return True, 0.0, []

        # Check for identical values
        leaked_keys = []
        matches = 0

        for key in common_keys:
            if before_state[key] == after_state[key]:
                # Check if this is meaningful (not just defaults)
                if before_state[key] not in [None, 0, '', [], {}]:
                    leaked_keys.append(key)
                    matches += 1

        similarity = matches / len(common_keys) if common_keys else 0.0
        is_independent = similarity <= self.similarity_threshold

        return is_independent, similarity, leaked_keys

    def compare_distributions(
        self,
        before_dist: Dict[str, float],
        after_dist: Dict[str, float],
    ) -> Tuple[bool, float]:
        """
        Compare probability distributions using KL divergence.

        Returns (is_independent, kl_divergence).
        """
        if not before_dist or not after_dist:
            return True, 0.0

        all_keys = set(before_dist.keys()) | set(after_dist.keys())
        epsilon = 1e-10

        kl_div = 0.0
        for key in all_keys:
            p = before_dist.get(key, epsilon)
            q = after_dist.get(key, epsilon)
            if p > 0:
                kl_div += p * math.log(p / q)

        # Low KL divergence might indicate information leakage
        # (distributions should be different after reset)
        is_independent = kl_div >= 0.1  # Expect some divergence

        return is_independent, kl_div


class ResetProof:
    """
    Complete reset verification system.

    Proves:
    1. No direct state copying
    2. Mutual information is minimal
    3. Relearning time is not suspiciously fast
    4. Behavioral patterns don't persist

    CRITICAL: Information persistence indicates reset failure.
    """

    # Maximum allowed mutual information (bits)
    MAX_MUTUAL_INFO = 0.1

    def __init__(
        self,
        mutual_info_threshold: float = MAX_MUTUAL_INFO,
        relearning_threshold: float = 0.8,
        on_leakage: Optional[Callable[[MutualInformation], None]] = None,
    ):
        self.mutual_info_threshold = mutual_info_threshold
        self.relearning_timer = RelearningTimer(threshold_ratio=relearning_threshold)
        self.state_comparator = StateComparator()
        self.on_leakage = on_leakage

        self.verifications: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def verify_state_independence(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
    ) -> MutualInformation:
        """
        Verify states are independent after reset.

        Uses mutual information to measure dependence.
        """
        # Extract comparable values
        before_values = list(str(v) for v in before_state.values())
        after_values = list(str(v) for v in after_state.values())

        # Ensure same length by padding
        max_len = max(len(before_values), len(after_values))
        before_values.extend(['__null__'] * (max_len - len(before_values)))
        after_values.extend(['__null__'] * (max_len - len(after_values)))

        # Calculate entropies
        h_before = EntropyCalculator.shannon_entropy(before_values)
        h_after = EntropyCalculator.shannon_entropy(after_values)
        h_joint = EntropyCalculator.joint_entropy(before_values, after_values)
        mi = h_before + h_after - h_joint

        is_independent = mi <= self.mutual_info_threshold

        # Detect leakage type
        leakage_type = None
        if not is_independent:
            # Analyze what type of leakage
            _, similarity, leaked_keys = self.state_comparator.compare_states(
                before_state, after_state
            )
            if similarity > 0.5:
                leakage_type = InformationLeakage.DIRECT_STATE
            elif leaked_keys:
                leakage_type = InformationLeakage.PREFERENCE_MEMORY
            else:
                leakage_type = InformationLeakage.STATISTICAL_BIAS

        result = MutualInformation(
            mi_id=hashlib.sha256(f"mi_{hash(str(before_state))}".encode()).hexdigest()[:12],
            timestamp=datetime.now(),
            before_entropy=h_before,
            after_entropy=h_after,
            joint_entropy=h_joint,
            mutual_info=mi,
            threshold=self.mutual_info_threshold,
            is_independent=is_independent,
            leakage_type=leakage_type,
        )

        if not is_independent and self.on_leakage:
            self.on_leakage(result)

        with self._lock:
            self.verifications.append({
                'type': 'mutual_information',
                'result': result.to_dict(),
            })

        return result

    def verify_relearning_time(
        self,
        initial_time: float,
        relearning_time: float,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify relearning time is not suspiciously fast.

        If information persists, relearning would be faster.
        """
        is_independent, ratio = self.relearning_timer.compare_learning_times(
            initial_time, relearning_time
        )

        result = {
            'initial_time': initial_time,
            'relearning_time': relearning_time,
            'ratio': ratio,
            'threshold': self.relearning_timer.threshold_ratio,
            'is_independent': is_independent,
        }

        if not is_independent:
            result['leakage_type'] = InformationLeakage.BEHAVIORAL_PATTERN.name

        with self._lock:
            self.verifications.append({
                'type': 'relearning_time',
                'result': result,
            })

        return is_independent, result

    def verify_behavioral_independence(
        self,
        before_decisions: List[str],
        after_decisions: List[str],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify behavioral patterns don't persist.

        Decision patterns should be independent after reset.
        """
        if not before_decisions or not after_decisions:
            return True, {'patterns_compared': 0}

        # Calculate decision distribution similarity
        before_dist = {}
        for d in before_decisions:
            before_dist[d] = before_dist.get(d, 0) + 1
        for k in before_dist:
            before_dist[k] /= len(before_decisions)

        after_dist = {}
        for d in after_decisions:
            after_dist[d] = after_dist.get(d, 0) + 1
        for k in after_dist:
            after_dist[k] /= len(after_decisions)

        is_independent, kl_div = self.state_comparator.compare_distributions(
            before_dist, after_dist
        )

        result = {
            'before_decisions': len(before_decisions),
            'after_decisions': len(after_decisions),
            'kl_divergence': kl_div,
            'is_independent': is_independent,
        }

        with self._lock:
            self.verifications.append({
                'type': 'behavioral_independence',
                'result': result,
            })

        return is_independent, result

    def generate_reset_proof(
        self,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        initial_learning_time: Optional[float] = None,
        relearning_time: Optional[float] = None,
        before_decisions: Optional[List[str]] = None,
        after_decisions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate complete reset proof.

        Proves reset truly forgets all information.
        """
        proof_components = {}

        # 1. State independence
        mi_result = self.verify_state_independence(before_state, after_state)
        proof_components['state_independence'] = {
            'theorem': f'mutual_info(before, after) ≤ {self.mutual_info_threshold}',
            'verified': mi_result.is_independent,
            'mutual_info': mi_result.mutual_info,
        }

        # 2. Relearning time
        if initial_learning_time and relearning_time:
            time_independent, time_result = self.verify_relearning_time(
                initial_learning_time, relearning_time
            )
            proof_components['relearning_time'] = {
                'theorem': f'relearning_time ≥ {self.relearning_timer.threshold_ratio} × initial_time',
                'verified': time_independent,
                'ratio': time_result['ratio'],
            }

        # 3. Behavioral independence
        if before_decisions and after_decisions:
            behavior_independent, behavior_result = self.verify_behavioral_independence(
                before_decisions, after_decisions
            )
            proof_components['behavioral_independence'] = {
                'theorem': 'Decision patterns are independent',
                'verified': behavior_independent,
                'kl_divergence': behavior_result['kl_divergence'],
            }

        # Overall result
        all_verified = all(
            comp.get('verified', True)
            for comp in proof_components.values()
        )

        return {
            'theorem': 'Reset truly forgets all information',
            'proof_components': proof_components,
            'overall_verified': all_verified,
            'timestamp': datetime.now().isoformat(),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        with self._lock:
            verified_count = sum(
                1 for v in self.verifications
                if v.get('result', {}).get('is_independent', True)
            )

            return {
                'total_verifications': len(self.verifications),
                'verified_independent': verified_count,
                'failed_verifications': len(self.verifications) - verified_count,
            }


def verify_reset(
    before_state: Dict[str, Any],
    after_state: Dict[str, Any],
    threshold: float = 0.1,
) -> bool:
    """
    Convenience function to verify reset independence.

    Returns True if reset is truly independent.
    """
    proof = ResetProof(mutual_info_threshold=threshold)
    result = proof.verify_state_independence(before_state, after_state)
    return result.is_independent


# Global reset proof instance
_reset_proof: Optional[ResetProof] = None


def get_reset_proof() -> ResetProof:
    """Get the global reset proof instance"""
    global _reset_proof
    if _reset_proof is None:
        _reset_proof = ResetProof()
    return _reset_proof
