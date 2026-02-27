#!/usr/bin/env python3
"""
STRENGTH LEARNER: Bayesian Learning for Optimal Ratchet Strength

DeepSeek's key insight: "Ratchet Selection Criteria - HOW do we decide what to ratchet?"

This module learns the optimal ratchet strength for each domain:
- Too much ratchet → locked in bad local optima
- Too little ratchet → no guaranteed progress
- Solution: Adaptive strength that learns optimal rigidity per domain

Uses Bayesian updates to learn from ratchet decision outcomes:
- If ratchet blocked a good improvement → lower strength
- If ratchet allowed a regression → raise strength
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum, auto
import numpy as np
from collections import defaultdict
import time
import json

from ratchet_orchestrator import RatchetDomain, RatchetOutcome, RatchetDecision


# =============================================================================
# BAYESIAN BELIEF REPRESENTATION
# =============================================================================

@dataclass
class DomainBelief:
    """
    Bayesian belief about optimal strength for a domain.

    Uses Beta distribution: Beta(alpha, beta)
    - alpha: successes (good ratchet decisions)
    - beta: failures (bad ratchet decisions)
    - Mean = alpha / (alpha + beta)
    """
    domain: str
    alpha: float = 10.0  # Prior successes
    beta: float = 10.0   # Prior failures
    observations: int = 0

    @property
    def mean(self) -> float:
        """Expected optimal strength."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        """Uncertainty in belief."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def std(self) -> float:
        """Standard deviation."""
        return np.sqrt(self.variance)

    def update(self, success: bool, weight: float = 1.0):
        """Bayesian update based on outcome."""
        if success:
            self.alpha += weight
        else:
            self.beta += weight
        self.observations += 1

    def sample(self) -> float:
        """Sample from belief distribution (Thompson sampling)."""
        return np.random.beta(self.alpha, self.beta)

    def credible_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """Get credible interval for optimal strength."""
        from scipy import stats
        dist = stats.beta(self.alpha, self.beta)
        lower = dist.ppf((1 - confidence) / 2)
        upper = dist.ppf(1 - (1 - confidence) / 2)
        return (lower, upper)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain,
            'alpha': self.alpha,
            'beta': self.beta,
            'mean': self.mean,
            'std': self.std,
            'observations': self.observations,
        }


# =============================================================================
# OUTCOME CLASSIFIER
# =============================================================================

class OutcomeClassifier:
    """
    Classifies ratchet outcomes as good or bad decisions.

    A good decision is one where:
    - Accept led to actual improvement
    - Reject prevented a regression

    A bad decision is one where:
    - Accept led to regression (false positive)
    - Reject blocked a good improvement (false negative)
    """

    def __init__(self, improvement_threshold: float = 0.01):
        self.improvement_threshold = improvement_threshold

    def classify(self, outcome: RatchetOutcome) -> Tuple[bool, float]:
        """
        Classify outcome as success or failure.

        Returns (is_success, confidence).
        """
        actual = outcome.actual_improvement
        predicted = outcome.predicted_improvement

        if outcome.decision == RatchetDecision.ACCEPT:
            # Accept was good if actual improvement >= 0
            is_success = actual >= -self.improvement_threshold
            # Confidence based on how much better than threshold
            confidence = min(1.0, 0.5 + actual * 5)
            return is_success, max(0.1, confidence)

        elif outcome.decision == RatchetDecision.REJECT:
            # Reject was good if would have been bad
            # We don't know actual for rejected, but can estimate
            is_success = not outcome.false_negative
            confidence = 0.7 if not outcome.false_negative else 0.3
            return is_success, confidence

        else:
            # Defer, Shadow, Warn - neutral outcomes
            return True, 0.5

    def compute_regret(
        self,
        outcomes: List[RatchetOutcome]
    ) -> float:
        """
        Compute regret from suboptimal decisions.

        Regret = sum of improvements we missed by rejecting good proposals
               + sum of regressions from accepting bad proposals
        """
        regret = 0.0

        for outcome in outcomes:
            if outcome.false_negative:
                # Missed a good improvement
                regret += abs(outcome.predicted_improvement)

            if outcome.false_positive:
                # Allowed a regression
                regret += abs(outcome.actual_improvement)

        return regret


# =============================================================================
# STRENGTH LEARNER
# =============================================================================

class StrengthLearner:
    """
    Bayesian learner for optimal ratchet strength per domain.

    Learning process:
    1. Observe ratchet decisions and their outcomes
    2. Classify outcomes as good or bad
    3. Update beliefs about optimal strength
    4. Recommend strength adjustments
    """

    # Initial priors for each domain
    DOMAIN_PRIORS = {
        'FOUNDATION': {'alpha': 99.0, 'beta': 1.0},   # Strong prior for high strength
        'DOMAIN': {'alpha': 85.0, 'beta': 15.0},      # Moderate-high
        'ADAPTIVE': {'alpha': 50.0, 'beta': 50.0},    # Neutral
        'META': {'alpha': 70.0, 'beta': 30.0},        # Slightly high
    }

    def __init__(self):
        self.domain_beliefs: Dict[str, DomainBelief] = {}
        self.outcome_history: List[RatchetOutcome] = []
        self.classifier = OutcomeClassifier()

        # Initialize beliefs
        self._initialize_beliefs()

        # Learning rate (how quickly to adapt)
        self.learning_rate = 1.0

        # Exploration factor (for Thompson sampling)
        self.exploration_factor = 0.1

    def _initialize_beliefs(self):
        """Initialize beliefs with domain-specific priors."""
        for domain_name, prior in self.DOMAIN_PRIORS.items():
            self.domain_beliefs[domain_name] = DomainBelief(
                domain=domain_name,
                alpha=prior['alpha'],
                beta=prior['beta'],
            )

    def get_optimal_strength(
        self,
        domain: str,
        use_exploration: bool = True
    ) -> float:
        """
        Return learned optimal ratchet strength for domain.

        Args:
            domain: Domain name (e.g., 'FOUNDATION', 'ADAPTIVE')
            use_exploration: Whether to add exploration noise

        Returns:
            Recommended strength [0, 1]
        """
        if domain not in self.domain_beliefs:
            return 0.7  # Default

        belief = self.domain_beliefs[domain]

        if use_exploration:
            # Thompson sampling: sample from posterior
            sampled = belief.sample()
            # Blend with mean for stability
            return 0.8 * belief.mean + 0.2 * sampled
        else:
            return belief.mean

    def update_from_outcome(
        self,
        domain: str,
        current_strength: float,
        outcome: RatchetOutcome
    ):
        """
        Bayesian update based on ratchet decision outcome.

        This is the core learning loop:
        1. Classify outcome as success/failure
        2. Update belief about domain's optimal strength
        3. Store outcome for regret analysis
        """
        self.outcome_history.append(outcome)

        if domain not in self.domain_beliefs:
            self.domain_beliefs[domain] = DomainBelief(
                domain=domain,
                alpha=50.0,
                beta=50.0,
            )

        belief = self.domain_beliefs[domain]

        # Classify outcome
        is_success, confidence = self.classifier.classify(outcome)

        # Weight by confidence and learning rate
        weight = confidence * self.learning_rate

        # Update belief
        belief.update(is_success, weight)

        # Adjust based on strength alignment
        # If decision was at high strength and failed, reduce strength belief
        # If decision was at low strength and succeeded, keep low
        if not is_success:
            if current_strength > 0.7:
                # High strength led to bad decision - might be too strict
                belief.beta += 0.5 * weight
            else:
                # Low strength led to bad decision - might be too lenient
                belief.alpha += 0.5 * weight

    def compute_regret(self, domain: Optional[str] = None) -> float:
        """
        Estimate regret from current strength setting.

        Args:
            domain: Specific domain or None for total regret
        """
        if domain:
            domain_outcomes = [
                o for o in self.outcome_history
                if hasattr(o, 'domain') and o.domain == domain
            ]
        else:
            domain_outcomes = self.outcome_history

        return self.classifier.compute_regret(domain_outcomes)

    def recommend_adjustment(
        self,
        domain: str,
        current_strength: float,
        window: int = 20
    ) -> Tuple[str, float]:
        """
        Recommend strength adjustment based on recent outcomes.

        Returns:
            (direction, magnitude) where direction is 'increase', 'decrease', or 'maintain'
        """
        if domain not in self.domain_beliefs:
            return 'maintain', 0.0

        belief = self.domain_beliefs[domain]
        optimal = belief.mean

        # Calculate difference from optimal
        diff = optimal - current_strength

        if abs(diff) < 0.05:
            return 'maintain', 0.0
        elif diff > 0:
            return 'increase', min(0.1, diff)
        else:
            return 'decrease', min(0.1, -diff)

    def get_confidence_in_strength(self, domain: str) -> float:
        """
        Get confidence in our belief about optimal strength.

        Lower variance = higher confidence.
        """
        if domain not in self.domain_beliefs:
            return 0.5

        belief = self.domain_beliefs[domain]
        # Convert variance to confidence (inverse relationship)
        # Variance range is [0, 0.25] for Beta
        confidence = 1.0 - 4 * belief.variance
        return max(0.0, min(1.0, confidence))

    def should_explore(self, domain: str) -> bool:
        """
        Determine if we should explore (try different strength).

        Explore more when:
        - Few observations
        - High uncertainty
        - Recent regret
        """
        if domain not in self.domain_beliefs:
            return True

        belief = self.domain_beliefs[domain]

        # Explore if few observations
        if belief.observations < 50:
            return np.random.random() < 0.2

        # Explore based on uncertainty
        if belief.std > 0.1:
            return np.random.random() < 0.1

        # Occasional exploration
        return np.random.random() < 0.02

    def get_domain_stats(self, domain: str) -> Dict[str, Any]:
        """Get detailed statistics for a domain."""
        if domain not in self.domain_beliefs:
            return {'error': 'Domain not found'}

        belief = self.domain_beliefs[domain]
        recent_outcomes = [
            o for o in self.outcome_history[-50:]
            if hasattr(o, 'domain') and o.domain == domain
        ]

        false_positives = sum(1 for o in recent_outcomes if o.false_positive)
        false_negatives = sum(1 for o in recent_outcomes if o.false_negative)

        return {
            'belief': belief.to_dict(),
            'optimal_strength': belief.mean,
            'confidence': self.get_confidence_in_strength(domain),
            'recent_outcomes': len(recent_outcomes),
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'regret': self.compute_regret(domain),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all domains."""
        stats = {
            'total_outcomes': len(self.outcome_history),
            'total_regret': self.compute_regret(),
            'domains': {},
        }

        for domain in self.domain_beliefs:
            stats['domains'][domain] = self.get_domain_stats(domain)

        return stats

    def save_state(self, filepath: str):
        """Save learner state to file."""
        state = {
            'beliefs': {d: b.to_dict() for d, b in self.domain_beliefs.items()},
            'outcome_count': len(self.outcome_history),
            'saved_at': time.time(),
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str):
        """Load learner state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        for domain, data in state['beliefs'].items():
            self.domain_beliefs[domain] = DomainBelief(
                domain=data['domain'],
                alpha=data['alpha'],
                beta=data['beta'],
                observations=data['observations'],
            )


# =============================================================================
# ADAPTIVE STRENGTH CONTROLLER
# =============================================================================

class AdaptiveStrengthController:
    """
    Higher-level controller for adaptive strength management.

    Combines learner with policy for when to apply adjustments.
    """

    def __init__(self, learner: Optional[StrengthLearner] = None):
        self.learner = learner or StrengthLearner()
        self.current_strengths: Dict[str, float] = {
            'FOUNDATION': 0.99,
            'DOMAIN': 0.85,
            'ADAPTIVE': 0.50,
            'META': 0.70,
        }
        self.adjustment_history: List[Dict] = []

    def update_and_adjust(
        self,
        domain: str,
        outcome: RatchetOutcome
    ) -> Optional[float]:
        """
        Update learner and possibly adjust strength.

        Returns new strength if adjusted, None otherwise.
        """
        current = self.current_strengths.get(domain, 0.7)

        # Update learner
        self.learner.update_from_outcome(domain, current, outcome)

        # Check if adjustment needed
        direction, magnitude = self.learner.recommend_adjustment(
            domain, current
        )

        if direction == 'maintain':
            return None

        # Apply adjustment
        if direction == 'increase':
            new_strength = min(1.0, current + magnitude)
        else:
            # Never decrease foundation below 0.95
            if domain == 'FOUNDATION':
                new_strength = max(0.95, current - magnitude)
            else:
                new_strength = max(0.0, current - magnitude)

        # Record adjustment
        self.adjustment_history.append({
            'domain': domain,
            'direction': direction,
            'old': current,
            'new': new_strength,
            'timestamp': time.time(),
        })

        self.current_strengths[domain] = new_strength
        return new_strength

    def get_strength(self, domain: str) -> float:
        """Get current strength for domain."""
        if self.learner.should_explore(domain):
            # Exploration: sample from belief
            return self.learner.get_optimal_strength(domain, use_exploration=True)
        else:
            return self.current_strengths.get(domain, 0.7)


# =============================================================================
# MAIN: Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("STRENGTH LEARNER: Bayesian Adaptive Ratchet Strength")
    print("=" * 60)

    # Create learner
    learner = StrengthLearner()

    print("\n[1] Initial beliefs:")
    for domain, belief in learner.domain_beliefs.items():
        print(f"    {domain}: mean={belief.mean:.2f}, std={belief.std:.3f}")

    # Simulate outcomes
    print("\n[2] Simulating ratchet decisions...")

    outcomes = [
        # Foundation: mostly good decisions
        RatchetOutcome('p1', RatchetDecision.ACCEPT, 0.05, 0.05),
        RatchetOutcome('p2', RatchetDecision.ACCEPT, 0.03, 0.03),
        RatchetOutcome('p3', RatchetDecision.REJECT, 0.0, -0.02, false_negative=False),

        # Adaptive: some exploration needed
        RatchetOutcome('p4', RatchetDecision.ACCEPT, 0.08, 0.10),
        RatchetOutcome('p5', RatchetDecision.REJECT, 0.0, 0.05, false_negative=True),
        RatchetOutcome('p6', RatchetDecision.ACCEPT, -0.02, 0.02, false_positive=True),
    ]

    # Domain assignments (mock)
    domain_assignments = ['FOUNDATION', 'FOUNDATION', 'FOUNDATION',
                          'ADAPTIVE', 'ADAPTIVE', 'ADAPTIVE']

    for outcome, domain in zip(outcomes, domain_assignments):
        current = learner.domain_beliefs[domain].mean
        learner.update_from_outcome(domain, current, outcome)
        print(f"    {domain}: updated after {outcome.decision.name}")

    # Show updated beliefs
    print("\n[3] Updated beliefs:")
    for domain, belief in learner.domain_beliefs.items():
        print(f"    {domain}: mean={belief.mean:.3f}, std={belief.std:.4f}, obs={belief.observations}")

    # Test recommendations
    print("\n[4] Strength recommendations:")
    for domain in ['FOUNDATION', 'DOMAIN', 'ADAPTIVE', 'META']:
        optimal = learner.get_optimal_strength(domain, use_exploration=False)
        current = StrengthLearner.DOMAIN_PRIORS[domain]['alpha'] / 100
        direction, mag = learner.recommend_adjustment(domain, current)
        print(f"    {domain}: current={current:.2f}, optimal={optimal:.3f}, rec={direction} {mag:.3f}")

    # Compute regret
    print("\n[5] Regret analysis:")
    total_regret = learner.compute_regret()
    print(f"    Total regret: {total_regret:.4f}")

    # Test controller
    print("\n[6] Adaptive controller:")
    controller = AdaptiveStrengthController(learner)

    # Simulate more outcomes through controller
    new_outcome = RatchetOutcome('p7', RatchetDecision.ACCEPT, 0.10, 0.08)
    adjustment = controller.update_and_adjust('ADAPTIVE', new_outcome)
    if adjustment:
        print(f"    ADAPTIVE strength adjusted to: {adjustment:.3f}")
    else:
        print(f"    ADAPTIVE strength maintained at: {controller.current_strengths['ADAPTIVE']:.3f}")

    # Full stats
    print("\n[7] Full statistics:")
    stats = learner.get_all_stats()
    print(f"    Total outcomes: {stats['total_outcomes']}")
    print(f"    Total regret: {stats['total_regret']:.4f}")

    print("\n" + "=" * 60)
    print("Strength Learner ready for adaptive ratchet control")
    print("=" * 60)
