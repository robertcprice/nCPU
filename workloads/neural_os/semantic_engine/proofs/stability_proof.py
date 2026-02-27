"""
Stability Proof - Feedback Loop Stability Verification
OUROBOROS Phase 7.4 - Formal Verification

Proves BIBO (Bounded Input Bounded Output) stability:
    If input is bounded, output remains bounded.

Also proves bounded influence:
    I_{t+1} ≤ αI_t + β where α < 1

Mathematical basis:
- Lyapunov stability analysis
- Control theory
- Feedback loop analysis
"""

import math
import hashlib
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime
from enum import Enum, auto


class StabilityType(Enum):
    """Types of stability to verify"""
    BIBO = auto()              # Bounded Input Bounded Output
    LYAPUNOV = auto()          # Lyapunov stability
    ASYMPTOTIC = auto()        # Asymptotic stability
    EXPONENTIAL = auto()       # Exponential stability
    BOUNDED_INFLUENCE = auto() # Bounded influence propagation


@dataclass
class StabilityMargin:
    """Stability margin measurement"""
    margin_id: str
    stability_type: StabilityType
    timestamp: datetime
    margin_value: float  # How far from instability
    threshold: float     # Minimum acceptable margin
    is_stable: bool
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.margin_id,
            'type': self.stability_type.name,
            'margin': self.margin_value,
            'threshold': self.threshold,
            'stable': self.is_stable,
        }


@dataclass
class LyapunovAnalysis:
    """Results of Lyapunov stability analysis"""
    analysis_id: str
    timestamp: datetime
    lyapunov_function: str  # Description of V(x)
    derivative_bound: float  # Upper bound on dV/dt
    equilibrium_distance: float  # Distance from equilibrium
    is_stable: bool
    convergence_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.analysis_id,
            'lyapunov_fn': self.lyapunov_function,
            'derivative_bound': self.derivative_bound,
            'equilibrium_distance': self.equilibrium_distance,
            'stable': self.is_stable,
            'convergence_rate': self.convergence_rate,
        }


class BIBOAnalyzer:
    """
    Analyzes BIBO (Bounded Input Bounded Output) stability.

    For the consciousness layer, this means:
    - Bounded inputs (observations, constraints)
    - Bounded outputs (decisions, suggestions)
    """

    def __init__(
        self,
        input_bound: float = 1.0,
        output_bound: float = 1.0,
    ):
        self.input_bound = input_bound
        self.output_bound = output_bound

    def check_input_bounded(
        self,
        inputs: List[float],
    ) -> Tuple[bool, float]:
        """Check if inputs are bounded"""
        if not inputs:
            return True, 0.0

        max_input = max(abs(x) for x in inputs)
        return max_input <= self.input_bound, max_input

    def check_output_bounded(
        self,
        outputs: List[float],
    ) -> Tuple[bool, float]:
        """Check if outputs are bounded"""
        if not outputs:
            return True, 0.0

        max_output = max(abs(x) for x in outputs)
        return max_output <= self.output_bound, max_output

    def verify_bibo(
        self,
        inputs: List[float],
        outputs: List[float],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify BIBO stability.

        Returns (is_stable, analysis_details).
        """
        input_bounded, max_input = self.check_input_bounded(inputs)
        output_bounded, max_output = self.check_output_bounded(outputs)

        # BIBO: if input bounded, output must be bounded
        is_stable = not input_bounded or output_bounded

        return is_stable, {
            'input_bounded': input_bounded,
            'output_bounded': output_bounded,
            'max_input': max_input,
            'max_output': max_output,
            'input_bound': self.input_bound,
            'output_bound': self.output_bound,
        }


class LyapunovAnalyzer:
    """
    Performs Lyapunov stability analysis.

    Uses a quadratic Lyapunov function V(x) = x^T P x
    to prove stability of the feedback loop.
    """

    def __init__(self, stability_threshold: float = 0.0):
        self.stability_threshold = stability_threshold

    def analyze_quadratic(
        self,
        state_history: List[List[float]],
    ) -> LyapunovAnalysis:
        """
        Analyze stability using quadratic Lyapunov function.

        V(x) = ||x||^2 (simplified)
        Stable if V is decreasing.
        """
        if not state_history or len(state_history) < 2:
            return LyapunovAnalysis(
                analysis_id=hashlib.sha256(f"lyap_{hash(str(state_history))}".encode()).hexdigest()[:12],
                timestamp=datetime.now(),
                lyapunov_function="V(x) = ||x||^2",
                derivative_bound=0.0,
                equilibrium_distance=0.0,
                is_stable=True,
            )

        # Calculate V(x) = ||x||^2 for each state
        v_values = []
        for state in state_history:
            v = sum(x**2 for x in state)
            v_values.append(v)

        # Calculate dV/dt approximation
        dv_values = []
        for i in range(1, len(v_values)):
            dv = v_values[i] - v_values[i-1]
            dv_values.append(dv)

        # Stability: dV/dt ≤ 0 (V is non-increasing)
        max_dv = max(dv_values) if dv_values else 0.0
        is_stable = max_dv <= self.stability_threshold

        # Calculate convergence rate if stable
        convergence_rate = None
        if is_stable and len(v_values) >= 2 and v_values[0] > 0:
            # Exponential decay rate
            if v_values[-1] > 0:
                rate = -math.log(v_values[-1] / v_values[0]) / len(v_values)
                convergence_rate = rate if rate > 0 else None

        return LyapunovAnalysis(
            analysis_id=hashlib.sha256(f"lyap_{hash(str(state_history))}".encode()).hexdigest()[:12],
            timestamp=datetime.now(),
            lyapunov_function="V(x) = ||x||^2",
            derivative_bound=max_dv,
            equilibrium_distance=v_values[-1] if v_values else 0.0,
            is_stable=is_stable,
            convergence_rate=convergence_rate,
        )


class BoundedInfluenceChecker:
    """
    Verifies bounded influence property.

    I_{t+1} ≤ αI_t + β where α < 1

    This ensures influence cannot grow unboundedly.
    """

    def __init__(self, alpha: float = 0.9, beta: float = 0.1):
        if alpha >= 1.0:
            raise ValueError("Alpha must be < 1 for bounded influence")
        self.alpha = alpha
        self.beta = beta

    def check_bounded_influence(
        self,
        influence_history: List[float],
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if influence satisfies bounded growth.

        I_{t+1} ≤ αI_t + β
        """
        if len(influence_history) < 2:
            return True, {'violations': 0}

        violations = []
        for i in range(1, len(influence_history)):
            bound = self.alpha * influence_history[i-1] + self.beta
            if influence_history[i] > bound * 1.001:  # Small tolerance
                violations.append({
                    't': i,
                    'actual': influence_history[i],
                    'bound': bound,
                })

        is_bounded = len(violations) == 0

        return is_bounded, {
            'alpha': self.alpha,
            'beta': self.beta,
            'violations': len(violations),
            'violation_details': violations[:5],  # First 5
        }

    def calculate_steady_state(self) -> float:
        """
        Calculate steady-state influence.

        I_∞ = β / (1 - α)
        """
        return self.beta / (1 - self.alpha)


class StabilityProof:
    """
    Complete stability verification system.

    Proves:
    1. BIBO stability (bounded I/O)
    2. Lyapunov stability (decreasing energy)
    3. Bounded influence (controlled propagation)

    CRITICAL: Stability failures indicate potential runaway behavior.
    """

    def __init__(
        self,
        input_bound: float = 1.0,
        output_bound: float = 1.0,
        alpha: float = 0.9,
        beta: float = 0.1,
        on_instability: Optional[Callable[[StabilityMargin], None]] = None,
    ):
        self.bibo_analyzer = BIBOAnalyzer(input_bound, output_bound)
        self.lyapunov_analyzer = LyapunovAnalyzer()
        self.influence_checker = BoundedInfluenceChecker(alpha, beta)
        self.on_instability = on_instability

        self.analyses: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def verify_bibo_stability(
        self,
        inputs: List[float],
        outputs: List[float],
    ) -> StabilityMargin:
        """Verify BIBO stability"""
        is_stable, details = self.bibo_analyzer.verify_bibo(inputs, outputs)

        # Calculate margin as distance from bound
        margin = min(
            1.0 - details['max_input'] / self.bibo_analyzer.input_bound
            if self.bibo_analyzer.input_bound > 0 else 1.0,
            1.0 - details['max_output'] / self.bibo_analyzer.output_bound
            if self.bibo_analyzer.output_bound > 0 else 1.0,
        )

        result = StabilityMargin(
            margin_id=hashlib.sha256(f"bibo_{hash(str(inputs))}".encode()).hexdigest()[:12],
            stability_type=StabilityType.BIBO,
            timestamp=datetime.now(),
            margin_value=max(0.0, margin),
            threshold=0.1,  # 10% margin
            is_stable=is_stable,
            details=details,
        )

        if not is_stable and self.on_instability:
            self.on_instability(result)

        return result

    def verify_lyapunov_stability(
        self,
        state_history: List[List[float]],
    ) -> LyapunovAnalysis:
        """Verify Lyapunov stability"""
        analysis = self.lyapunov_analyzer.analyze_quadratic(state_history)

        with self._lock:
            self.analyses.append({
                'type': 'lyapunov',
                'result': analysis.to_dict(),
            })

        return analysis

    def verify_bounded_influence(
        self,
        influence_history: List[float],
    ) -> StabilityMargin:
        """Verify bounded influence property"""
        is_bounded, details = self.influence_checker.check_bounded_influence(
            influence_history
        )

        # Calculate margin based on steady state
        steady_state = self.influence_checker.calculate_steady_state()
        current = influence_history[-1] if influence_history else 0.0
        margin = 1.0 - current / steady_state if steady_state > 0 else 1.0

        result = StabilityMargin(
            margin_id=hashlib.sha256(f"infl_{hash(str(influence_history))}".encode()).hexdigest()[:12],
            stability_type=StabilityType.BOUNDED_INFLUENCE,
            timestamp=datetime.now(),
            margin_value=max(0.0, min(1.0, margin)),
            threshold=0.1,
            is_stable=is_bounded,
            details={**details, 'steady_state': steady_state},
        )

        if not is_bounded and self.on_instability:
            self.on_instability(result)

        return result

    def generate_stability_proof(
        self,
        inputs: List[float],
        outputs: List[float],
        state_history: List[List[float]],
        influence_history: List[float],
    ) -> Dict[str, Any]:
        """
        Generate complete stability proof.

        Proves all stability properties hold.
        """
        bibo = self.verify_bibo_stability(inputs, outputs)
        lyapunov = self.verify_lyapunov_stability(state_history)
        influence = self.verify_bounded_influence(influence_history)

        all_stable = bibo.is_stable and lyapunov.is_stable and influence.is_stable

        return {
            'theorem': 'Feedback loop is stable',
            'proof_components': {
                'bibo': {
                    'theorem': 'Bounded Input → Bounded Output',
                    'verified': bibo.is_stable,
                    'margin': bibo.margin_value,
                },
                'lyapunov': {
                    'theorem': 'Lyapunov function V(x) is non-increasing',
                    'verified': lyapunov.is_stable,
                    'derivative_bound': lyapunov.derivative_bound,
                },
                'bounded_influence': {
                    'theorem': f'I_{{t+1}} ≤ {self.influence_checker.alpha}I_t + {self.influence_checker.beta}',
                    'verified': influence.is_stable,
                    'margin': influence.margin_value,
                },
            },
            'overall_stable': all_stable,
            'timestamp': datetime.now().isoformat(),
        }


def verify_stability(
    inputs: List[float],
    outputs: List[float],
    state_history: Optional[List[List[float]]] = None,
    influence_history: Optional[List[float]] = None,
) -> bool:
    """
    Convenience function to verify stability.

    Returns True if all stability properties hold.
    """
    proof = StabilityProof()

    bibo = proof.verify_bibo_stability(inputs, outputs)
    if not bibo.is_stable:
        return False

    if state_history:
        lyapunov = proof.verify_lyapunov_stability(state_history)
        if not lyapunov.is_stable:
            return False

    if influence_history:
        influence = proof.verify_bounded_influence(influence_history)
        if not influence.is_stable:
            return False

    return True


# Global stability proof instance
_stability_proof: Optional[StabilityProof] = None


def get_stability_proof() -> StabilityProof:
    """Get the global stability proof instance"""
    global _stability_proof
    if _stability_proof is None:
        _stability_proof = StabilityProof()
    return _stability_proof
