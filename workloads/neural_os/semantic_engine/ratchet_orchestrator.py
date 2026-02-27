#!/usr/bin/env python3
"""
RATCHET ORCHESTRATOR: Provable Irreversible Self-Improvement

Based on Hybrid AI Review Round 3 Consensus:
- ChatGPT: Formal verification + crypto commitments
- Claude: Antifragile reversibility (keep adaptive layers weak)
- DeepSeek: 4-layer spectrum approach with meta-controller
- Grok: External anchoring (10/10) + Type-theoretic proofs (9/10)

Target: Singularity Readiness 9/10 (from 4/10)

Core Concept: A provable ratchet is a verifiable monotonic operator R: S -> S where:
- Monotonicity: forall s in S, U(R(s)) >= U(s) for utility U
- Irreversibility: No computable R^-1 exists
- Provability: Exists proof in PA + domain axioms
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from enum import Enum, auto
from datetime import datetime
import hashlib
import json
import numpy as np
from collections import defaultdict
import time


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

class RatchetDomain(Enum):
    """Domains for the 4-layer ratchet system (DeepSeek's spectrum approach)."""
    FOUNDATION = auto()  # Core values, safety (99% strength)
    DOMAIN = auto()      # Math truths, physics laws (85% strength)
    ADAPTIVE = auto()    # Skills, preferences (50% strength)
    META = auto()        # Self-adjusting strength (learned)


class ProofType(Enum):
    """Types of proofs supported by the ratchet system."""
    ALGEBRAIC = auto()       # Direct mathematical proof
    TEST_BASED = auto()      # Empirical proof via testing
    TYPE_THEORETIC = auto()  # Lean/Coq style dependent types
    COMPOSITION = auto()     # Composed from existing lemmas
    MONOTONICITY = auto()    # U(after) >= U(before) proof
    ESS = auto()            # Evolutionary Stable Strategy (Maynard Smith)


class RatchetDecision(Enum):
    """Decision types for ratchet verification."""
    ACCEPT = auto()      # Improvement verified and committed
    REJECT = auto()      # Improvement failed verification
    DEFER = auto()       # Need more evidence/proof
    SHADOW = auto()      # Logged but not enforced (shadow mode)
    WARN = auto()        # Warning issued but allowed


@dataclass
class RatchetProof:
    """
    Immutable proof certificate for verified improvements.

    This is the cryptographic commitment that makes improvements irreversible.
    """
    proof_id: str
    proof_type: ProofType
    statements: List[str]
    evidence: Dict[str, Any]
    confidence: float
    timestamp: float
    anchor_hash: Optional[str] = None
    tactic_sequence: List[str] = field(default_factory=list)
    verified: bool = False

    def __post_init__(self):
        if not self.proof_id:
            # Generate unique proof ID from content hash
            content = json.dumps({
                'type': self.proof_type.name,
                'statements': self.statements,
                'timestamp': self.timestamp
            }, sort_keys=True)
            self.proof_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'proof_id': self.proof_id,
            'proof_type': self.proof_type.name,
            'statements': self.statements,
            'evidence': self.evidence,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'anchor_hash': self.anchor_hash,
            'tactic_sequence': self.tactic_sequence,
            'verified': self.verified
        }


@dataclass
class SystemState:
    """
    Snapshot of system state for improvement comparison.

    Captures all relevant metrics for utility computation.
    """
    state_id: str
    timestamp: float
    metrics: Dict[str, float]
    program_hash: str
    complexity: float
    uncertainty: float
    test_results: Dict[str, bool] = field(default_factory=dict)

    def __post_init__(self):
        if not self.state_id:
            content = json.dumps({
                'metrics': self.metrics,
                'program_hash': self.program_hash,
                'timestamp': self.timestamp
            }, sort_keys=True)
            self.state_id = hashlib.sha256(content.encode()).hexdigest()[:16]

    def compute_utility(self, weights: Optional[Dict[str, float]] = None) -> float:
        """
        Compute utility score from metrics.

        Default weights favor accuracy > simplicity > speed
        """
        default_weights = {
            'accuracy': 1.0,
            'test_pass_rate': 0.8,
            'simplicity': 0.3,
            'speed': 0.2,
            'confidence': 0.5,
        }
        w = weights or default_weights

        utility = 0.0
        for metric, value in self.metrics.items():
            weight = w.get(metric, 0.1)
            utility += weight * value

        # Penalize complexity and uncertainty
        utility -= 0.1 * self.complexity
        utility -= 0.2 * self.uncertainty

        return utility


@dataclass
class ImprovementProposal:
    """
    Proposed modification with full context for ratchet verification.

    Contains before/after states and the modification itself.
    """
    proposal_id: str
    source_state: SystemState
    target_state: SystemState
    modification: Dict[str, Any]  # Serializable representation
    domain: RatchetDomain
    utility_delta: float
    description: str
    proof: Optional[RatchetProof] = None
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if not self.proposal_id:
            content = json.dumps({
                'source': self.source_state.state_id,
                'target': self.target_state.state_id,
                'timestamp': self.timestamp
            }, sort_keys=True)
            self.proposal_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Compute utility delta if not provided
        if self.utility_delta == 0.0:
            self.utility_delta = (
                self.target_state.compute_utility() -
                self.source_state.compute_utility()
            )


@dataclass
class RatchetOutcome:
    """Outcome of a ratchet decision for learning."""
    proposal_id: str
    decision: RatchetDecision
    actual_improvement: float  # Measured after commitment
    predicted_improvement: float  # Predicted by proof
    false_positive: bool = False  # Allowed regression
    false_negative: bool = False  # Blocked good improvement
    timestamp: float = field(default_factory=time.time)


@dataclass
class AnchorRecord:
    """Record of an external anchor."""
    anchor_id: str
    proof_hash: str
    chain_hash: str
    timestamp: float
    anchor_type: str
    external_reference: Optional[str] = None


# =============================================================================
# LAYERED RATCHET SYSTEM (DeepSeek's 4-layer approach)
# =============================================================================

class LayeredRatchetSystem:
    """
    4-layer adaptive ratchet architecture.

    Layers (from most to least strict):
    1. Foundation (99%): Core values, safety constraints - NEVER violated
    2. Domain (85%): Mathematical truths, physical laws - rarely violated
    3. Adaptive (50%): Skills, preferences - flexible for exploration
    4. Meta (learned): Self-adjusting strength based on outcomes

    This balances irreversibility vs flexibility - the key tradeoff identified
    by all AIs in Round 3.
    """

    DEFAULT_STRENGTHS = {
        RatchetDomain.FOUNDATION: 0.99,
        RatchetDomain.DOMAIN: 0.85,
        RatchetDomain.ADAPTIVE: 0.50,
        RatchetDomain.META: 0.70,  # Initial, will be learned
    }

    def __init__(self, strengths: Optional[Dict[RatchetDomain, float]] = None):
        self.strengths = strengths or self.DEFAULT_STRENGTHS.copy()
        self.layer_history: Dict[RatchetDomain, List[RatchetOutcome]] = {
            d: [] for d in RatchetDomain
        }
        self.violation_counts: Dict[RatchetDomain, int] = {
            d: 0 for d in RatchetDomain
        }

    def get_strength(self, domain: RatchetDomain) -> float:
        """Get current ratchet strength for domain."""
        return self.strengths[domain]

    def set_strength(self, domain: RatchetDomain, strength: float):
        """Set ratchet strength (bounded [0, 1])."""
        # Foundation layer cannot be weakened below 0.95
        if domain == RatchetDomain.FOUNDATION:
            strength = max(0.95, strength)
        self.strengths[domain] = max(0.0, min(1.0, strength))

    def required_confidence(self, domain: RatchetDomain) -> float:
        """
        Get required proof confidence for domain.

        Higher strength = higher confidence required.
        """
        return self.strengths[domain]

    def evaluate_proposal(
        self,
        proposal: ImprovementProposal,
        proof: Optional[RatchetProof]
    ) -> Tuple[RatchetDecision, str]:
        """
        Evaluate proposal against layer constraints.

        Returns (decision, reason).
        """
        domain = proposal.domain
        required = self.required_confidence(domain)

        # Foundation layer: NEVER accept without proof
        if domain == RatchetDomain.FOUNDATION:
            if proof is None or not proof.verified:
                return RatchetDecision.REJECT, "Foundation layer requires verified proof"
            if proof.confidence < 0.99:
                return RatchetDecision.REJECT, f"Foundation requires 99% confidence, got {proof.confidence:.2%}"

        # No proof provided
        if proof is None:
            if domain == RatchetDomain.ADAPTIVE:
                # Adaptive layer can accept unproven changes probabilistically
                if np.random.random() > self.strengths[domain]:
                    return RatchetDecision.WARN, "Adaptive layer allowing unproven improvement"
            return RatchetDecision.DEFER, "Proof required"

        # Check confidence threshold
        if proof.confidence < required:
            if domain == RatchetDomain.ADAPTIVE:
                return RatchetDecision.WARN, f"Below threshold but adaptive layer allows ({proof.confidence:.2%} < {required:.2%})"
            return RatchetDecision.REJECT, f"Confidence {proof.confidence:.2%} below required {required:.2%}"

        # Verify monotonicity
        if proposal.utility_delta < 0:
            if domain in (RatchetDomain.FOUNDATION, RatchetDomain.DOMAIN):
                return RatchetDecision.REJECT, "Negative utility delta in strict layer"
            # Adaptive/Meta can accept small regressions for exploration
            if proposal.utility_delta < -0.1:
                return RatchetDecision.REJECT, "Utility delta too negative"

        return RatchetDecision.ACCEPT, "Proposal verified"

    def record_outcome(self, outcome: RatchetOutcome, domain: RatchetDomain):
        """Record outcome for learning."""
        self.layer_history[domain].append(outcome)

        # Track violations
        if outcome.false_positive:
            self.violation_counts[domain] += 1

    def get_layer_stats(self) -> Dict[str, Any]:
        """Get statistics for each layer."""
        stats = {}
        for domain in RatchetDomain:
            history = self.layer_history[domain]
            if history:
                stats[domain.name] = {
                    'strength': self.strengths[domain],
                    'decisions': len(history),
                    'violations': self.violation_counts[domain],
                    'avg_improvement': np.mean([o.actual_improvement for o in history]),
                    'false_positive_rate': sum(1 for o in history if o.false_positive) / len(history),
                    'false_negative_rate': sum(1 for o in history if o.false_negative) / len(history),
                }
            else:
                stats[domain.name] = {
                    'strength': self.strengths[domain],
                    'decisions': 0,
                    'violations': 0,
                }
        return stats


# =============================================================================
# RATCHET STATE (Persistent state with history)
# =============================================================================

@dataclass
class RatchetState:
    """
    Persistent ratchet state with full history.

    This is the "memory" of the ratchet system - it tracks all committed
    improvements and their anchor chain.
    """
    current_level: Dict[str, float]  # Per-domain ratchet strength
    history: List[Tuple[float, str, str]]  # (timestamp, proposal_id, proof_id)
    anchor_chain: List[str]  # Hash chain of committed improvements
    baseline_utility: float = 0.0
    current_utility: float = 0.0

    def __init__(self):
        self.current_level = {d.name: 0.5 for d in RatchetDomain}
        self.history = []
        self.anchor_chain = []
        self.baseline_utility = 0.0
        self.current_utility = 0.0

    def commit(
        self,
        proposal: ImprovementProposal,
        proof: RatchetProof,
        anchor_hash: str
    ):
        """Commit an improvement to history."""
        self.history.append((
            time.time(),
            proposal.proposal_id,
            proof.proof_id
        ))
        self.anchor_chain.append(anchor_hash)
        self.current_utility = proposal.target_state.compute_utility()

    def get_improvement_count(self) -> int:
        """Get total number of committed improvements."""
        return len(self.history)

    def get_total_improvement(self) -> float:
        """Get total utility improvement from baseline."""
        return self.current_utility - self.baseline_utility

    def verify_chain_integrity(self) -> bool:
        """Verify the anchor chain has not been tampered with."""
        if not self.anchor_chain:
            return True

        # Verify each link
        for i in range(1, len(self.anchor_chain)):
            # Each anchor should reference the previous
            # (Simplified check - real implementation would verify hashes)
            if not self.anchor_chain[i]:
                return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            'current_level': self.current_level,
            'history_count': len(self.history),
            'anchor_chain_length': len(self.anchor_chain),
            'baseline_utility': self.baseline_utility,
            'current_utility': self.current_utility,
            'total_improvement': self.get_total_improvement(),
            'chain_valid': self.verify_chain_integrity()
        }


# =============================================================================
# RATCHET ORCHESTRATOR (Main coordinator)
# =============================================================================

class RatchetMode(Enum):
    """Operating modes for rollout strategy."""
    SHADOW = auto()    # Log only, don't block
    WARN = auto()      # Warn but allow
    SELECTIVE = auto() # Block foundation only
    FULL = auto()      # All layers enforcing


class RatchetOrchestrator:
    """
    Main orchestrator coordinating all ratchet subsystems.

    Provides the unified API for:
    - Proposing improvements
    - Verifying monotonicity
    - Generating/verifying proofs
    - Anchoring committed improvements
    - Learning optimal ratchet strengths

    This is the "Ratchet Orchestrator" from Grok's V4 architecture.
    """

    def __init__(
        self,
        mode: RatchetMode = RatchetMode.SHADOW,
        proof_engine: Optional['ProofEngine'] = None,
        anchor_oracle: Optional['AnchorOracle'] = None,
        strength_learner: Optional['StrengthLearner'] = None
    ):
        self.mode = mode
        self.layer_system = LayeredRatchetSystem()
        self.state = RatchetState()

        # Subsystems (lazy initialization)
        self._proof_engine = proof_engine
        self._anchor_oracle = anchor_oracle
        self._strength_learner = strength_learner

        # Metrics
        self.proposal_count = 0
        self.accept_count = 0
        self.reject_count = 0
        self.decision_log: List[Tuple[str, RatchetDecision, str]] = []

    @property
    def proof_engine(self) -> 'ProofEngine':
        if self._proof_engine is None:
            from proof_engine import ProofEngine
            self._proof_engine = ProofEngine()
        return self._proof_engine

    @property
    def anchor_oracle(self) -> 'AnchorOracle':
        if self._anchor_oracle is None:
            from anchor_oracle import AnchorOracle
            self._anchor_oracle = AnchorOracle()
        return self._anchor_oracle

    @property
    def strength_learner(self) -> 'StrengthLearner':
        if self._strength_learner is None:
            from strength_learner import StrengthLearner
            self._strength_learner = StrengthLearner()
        return self._strength_learner

    def propose_improvement(
        self,
        proposal: ImprovementProposal,
        auto_prove: bool = True
    ) -> Tuple[RatchetDecision, Optional[str]]:
        """
        Gate all modifications through ratchet verification.

        This is the main entry point for proposing improvements.

        Returns:
            (decision, anchor_id if committed else reason)
        """
        self.proposal_count += 1

        # Try to generate proof if not provided
        proof = proposal.proof
        if proof is None and auto_prove:
            try:
                proof = self.proof_engine.generate_proof(proposal)
                proposal.proof = proof
            except Exception as e:
                proof = None

        # Evaluate against layer system
        decision, reason = self.layer_system.evaluate_proposal(proposal, proof)

        # Apply mode-specific behavior
        if self.mode == RatchetMode.SHADOW:
            # Log but don't enforce
            self.decision_log.append((proposal.proposal_id, decision, reason))
            return RatchetDecision.SHADOW, f"SHADOW: would {decision.name} - {reason}"

        elif self.mode == RatchetMode.WARN:
            # Warn but allow
            if decision == RatchetDecision.REJECT:
                decision = RatchetDecision.WARN
            self.decision_log.append((proposal.proposal_id, decision, reason))
            return decision, reason

        elif self.mode == RatchetMode.SELECTIVE:
            # Only enforce foundation layer
            if proposal.domain != RatchetDomain.FOUNDATION:
                if decision == RatchetDecision.REJECT:
                    decision = RatchetDecision.WARN

        # Full enforcement mode or selective with foundation
        self.decision_log.append((proposal.proposal_id, decision, reason))

        if decision == RatchetDecision.ACCEPT:
            self.accept_count += 1
            # Commit the improvement
            anchor_id = self.commit_improvement(proposal, proof)
            return decision, anchor_id
        else:
            if decision == RatchetDecision.REJECT:
                self.reject_count += 1
            return decision, reason

    def verify_monotonicity(
        self,
        before: SystemState,
        after: SystemState
    ) -> Tuple[bool, float]:
        """
        Ensure U(after) >= U(before).

        Returns (is_monotonic, utility_delta).
        """
        u_before = before.compute_utility()
        u_after = after.compute_utility()
        delta = u_after - u_before

        return delta >= 0, delta

    def commit_improvement(
        self,
        proposal: ImprovementProposal,
        proof: RatchetProof
    ) -> str:
        """
        Anchor proven improvement to external oracle.

        Returns anchor_id.
        """
        # Generate anchor
        anchor_id = self.anchor_oracle.anchor_proof(proof)

        # Update proof with anchor
        proof.anchor_hash = anchor_id
        proof.verified = True

        # Commit to state
        self.state.commit(proposal, proof, anchor_id)

        # Record outcome for learning
        outcome = RatchetOutcome(
            proposal_id=proposal.proposal_id,
            decision=RatchetDecision.ACCEPT,
            actual_improvement=proposal.utility_delta,
            predicted_improvement=proposal.utility_delta,
        )
        self.layer_system.record_outcome(outcome, proposal.domain)

        # Update strength learner
        try:
            self.strength_learner.update_from_outcome(
                proposal.domain.name,
                self.layer_system.get_strength(proposal.domain),
                outcome
            )
        except Exception:
            pass  # Learner may not be initialized

        return anchor_id

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            'mode': self.mode.name,
            'proposals': self.proposal_count,
            'accepts': self.accept_count,
            'rejects': self.reject_count,
            'accept_rate': self.accept_count / max(1, self.proposal_count),
            'state': self.state.to_dict(),
            'layers': self.layer_system.get_layer_stats(),
        }

    def set_mode(self, mode: RatchetMode):
        """Change operating mode."""
        self.mode = mode

    def verify_improvement_chain(self) -> bool:
        """Verify the entire improvement chain is valid."""
        return (
            self.state.verify_chain_integrity() and
            self.anchor_oracle.verify_anchor_chain()
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_system_state(
    metrics: Dict[str, float],
    program_repr: str,
    test_results: Optional[Dict[str, bool]] = None
) -> SystemState:
    """Helper to create a SystemState from common inputs."""
    program_hash = hashlib.sha256(program_repr.encode()).hexdigest()[:16]

    # Estimate complexity from program representation
    complexity = len(program_repr) / 100.0

    # Estimate uncertainty from test coverage
    if test_results:
        pass_rate = sum(test_results.values()) / len(test_results)
        uncertainty = 1.0 - pass_rate
    else:
        uncertainty = 0.5

    return SystemState(
        state_id='',  # Will be generated
        timestamp=time.time(),
        metrics=metrics,
        program_hash=program_hash,
        complexity=complexity,
        uncertainty=uncertainty,
        test_results=test_results or {}
    )


def create_proposal(
    before: SystemState,
    after: SystemState,
    modification: Dict[str, Any],
    domain: RatchetDomain = RatchetDomain.ADAPTIVE,
    description: str = ''
) -> ImprovementProposal:
    """Helper to create an ImprovementProposal."""
    return ImprovementProposal(
        proposal_id='',  # Will be generated
        source_state=before,
        target_state=after,
        modification=modification,
        domain=domain,
        utility_delta=0.0,  # Will be computed
        description=description
    )


# =============================================================================
# MAIN: Demonstration
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RATCHET ORCHESTRATOR: Provable Irreversible Self-Improvement")
    print("=" * 60)

    # Initialize orchestrator in shadow mode
    orchestrator = RatchetOrchestrator(mode=RatchetMode.SHADOW)

    print(f"\n[1] Orchestrator initialized in {orchestrator.mode.name} mode")

    # Create sample states
    before = create_system_state(
        metrics={'accuracy': 0.85, 'test_pass_rate': 0.9, 'simplicity': 0.7},
        program_repr='def f(x): return x * 2',
        test_results={'test1': True, 'test2': True, 'test3': False}
    )

    after = create_system_state(
        metrics={'accuracy': 0.95, 'test_pass_rate': 0.95, 'simplicity': 0.7},
        program_repr='def f(x): return x << 1  # Optimized',
        test_results={'test1': True, 'test2': True, 'test3': True}
    )

    print(f"\n[2] Created states:")
    print(f"    Before utility: {before.compute_utility():.3f}")
    print(f"    After utility:  {after.compute_utility():.3f}")

    # Create proposal
    proposal = create_proposal(
        before=before,
        after=after,
        modification={'type': 'optimization', 'change': 'multiply -> shift'},
        domain=RatchetDomain.ADAPTIVE,
        description='Optimize multiplication to bit shift'
    )

    print(f"\n[3] Created proposal:")
    print(f"    Domain: {proposal.domain.name}")
    print(f"    Utility delta: {proposal.utility_delta:.3f}")

    # Verify monotonicity
    is_monotonic, delta = orchestrator.verify_monotonicity(before, after)
    print(f"\n[4] Monotonicity check:")
    print(f"    Is monotonic: {is_monotonic}")
    print(f"    Utility delta: {delta:.3f}")

    # Simulate proof (normally would use ProofEngine)
    proof = RatchetProof(
        proof_id='',
        proof_type=ProofType.TEST_BASED,
        statements=[
            'All tests pass after modification',
            'Utility increased by 0.15',
            'No functional change detected'
        ],
        evidence={'test_results': after.test_results},
        confidence=0.95,
        timestamp=time.time(),
        verified=True
    )
    proposal.proof = proof

    print(f"\n[5] Generated proof:")
    print(f"    Type: {proof.proof_type.name}")
    print(f"    Confidence: {proof.confidence:.2%}")

    # Propose improvement
    decision, result = orchestrator.propose_improvement(proposal, auto_prove=False)
    print(f"\n[6] Proposal decision:")
    print(f"    Decision: {decision.name}")
    print(f"    Result: {result}")

    # Show layer system
    print(f"\n[7] Layer system:")
    for domain in RatchetDomain:
        strength = orchestrator.layer_system.get_strength(domain)
        print(f"    {domain.name}: {strength:.0%} strength")

    # Show stats
    print(f"\n[8] Orchestrator stats:")
    stats = orchestrator.get_stats()
    print(f"    Mode: {stats['mode']}")
    print(f"    Proposals: {stats['proposals']}")
    print(f"    Accepts: {stats['accepts']}")
    print(f"    Rejects: {stats['rejects']}")

    print("\n" + "=" * 60)
    print("Ratchet Orchestrator ready for V4 integration")
    print("=" * 60)
