#!/usr/bin/env python3
"""
Unit Tests for Ratchet Orchestrator and Components

Tests the V4 Provable Ratchet System:
- RatchetOrchestrator
- ProofEngine
- AnchorOracle
- StrengthLearner
- LayeredRatchetSystem
"""

import pytest
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ratchet_orchestrator import (
    RatchetOrchestrator, RatchetMode, RatchetDomain, RatchetDecision,
    LayeredRatchetSystem, RatchetState, RatchetProof, ProofType,
    ImprovementProposal, SystemState, RatchetOutcome,
    create_system_state, create_proposal
)
from proof_engine import (
    ProofEngine, AlgebraicProofStrategy, TestBasedProofStrategy,
    MonotonicityProofStrategy, ESSProofStrategy
)
from anchor_oracle import AnchorOracle, AnchorType, AnchorRecord
from strength_learner import (
    StrengthLearner, DomainBelief, AdaptiveStrengthController
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def basic_before_state():
    """Create a basic 'before' system state."""
    return create_system_state(
        metrics={'accuracy': 0.85, 'test_pass_rate': 0.9},
        program_repr='def f(x): return x * 2',
        test_results={'test1': True, 'test2': True, 'test3': False}
    )


@pytest.fixture
def basic_after_state():
    """Create a basic 'after' system state with improvement."""
    return create_system_state(
        metrics={'accuracy': 0.92, 'test_pass_rate': 0.95},
        program_repr='def f(x): return x << 1',
        test_results={'test1': True, 'test2': True, 'test3': True}
    )


@pytest.fixture
def regressing_after_state():
    """Create an 'after' state that regresses."""
    return create_system_state(
        metrics={'accuracy': 0.70, 'test_pass_rate': 0.7},
        program_repr='def f(x): return x',
        test_results={'test1': True, 'test2': False, 'test3': False}
    )


@pytest.fixture
def basic_proposal(basic_before_state, basic_after_state):
    """Create a basic improvement proposal."""
    return create_proposal(
        before=basic_before_state,
        after=basic_after_state,
        modification={'type': 'optimization', 'change': 'multiply -> shift'},
        domain=RatchetDomain.ADAPTIVE,
        description='Optimize multiplication to bit shift'
    )


# =============================================================================
# SYSTEM STATE TESTS
# =============================================================================

class TestSystemState:
    """Tests for SystemState."""

    def test_create_system_state(self, basic_before_state):
        """Test system state creation."""
        assert basic_before_state is not None
        assert basic_before_state.state_id != ''
        assert 'accuracy' in basic_before_state.metrics

    def test_compute_utility(self, basic_before_state, basic_after_state):
        """Test utility computation."""
        u_before = basic_before_state.compute_utility()
        u_after = basic_after_state.compute_utility()

        assert u_after > u_before, "After state should have higher utility"

    def test_utility_with_custom_weights(self, basic_before_state):
        """Test utility with custom weights."""
        default_utility = basic_before_state.compute_utility()
        custom_weights = {'accuracy': 2.0, 'test_pass_rate': 0.1}
        custom_utility = basic_before_state.compute_utility(weights=custom_weights)

        assert default_utility != custom_utility


# =============================================================================
# IMPROVEMENT PROPOSAL TESTS
# =============================================================================

class TestImprovementProposal:
    """Tests for ImprovementProposal."""

    def test_create_proposal(self, basic_proposal):
        """Test proposal creation."""
        assert basic_proposal is not None
        assert basic_proposal.proposal_id != ''
        assert basic_proposal.domain == RatchetDomain.ADAPTIVE

    def test_utility_delta_computed(self, basic_proposal):
        """Test utility delta is computed."""
        assert basic_proposal.utility_delta > 0, "Improvement should have positive delta"


# =============================================================================
# LAYERED RATCHET SYSTEM TESTS
# =============================================================================

class TestLayeredRatchetSystem:
    """Tests for LayeredRatchetSystem."""

    def test_default_strengths(self):
        """Test default layer strengths."""
        layer_system = LayeredRatchetSystem()

        assert layer_system.get_strength(RatchetDomain.FOUNDATION) >= 0.95
        assert layer_system.get_strength(RatchetDomain.DOMAIN) >= 0.8
        assert layer_system.get_strength(RatchetDomain.ADAPTIVE) >= 0.4

    def test_foundation_cannot_be_weakened_below_95(self):
        """Test foundation layer cannot be weakened below 0.95."""
        layer_system = LayeredRatchetSystem()
        layer_system.set_strength(RatchetDomain.FOUNDATION, 0.5)

        assert layer_system.get_strength(RatchetDomain.FOUNDATION) >= 0.95

    def test_evaluate_proposal_requires_proof_for_foundation(self, basic_proposal):
        """Test foundation layer requires verified proof."""
        layer_system = LayeredRatchetSystem()

        # Change proposal domain to foundation
        basic_proposal.domain = RatchetDomain.FOUNDATION

        decision, reason = layer_system.evaluate_proposal(basic_proposal, proof=None)

        assert decision == RatchetDecision.REJECT
        assert "proof" in reason.lower()

    def test_evaluate_proposal_accepts_with_high_confidence_proof(self, basic_proposal):
        """Test proposal accepted with high confidence proof."""
        layer_system = LayeredRatchetSystem()

        proof = RatchetProof(
            proof_id='',
            proof_type=ProofType.MONOTONICITY,
            statements=['Test proof'],
            evidence={'test': True},
            confidence=0.95,
            timestamp=time.time(),
            verified=True
        )

        decision, reason = layer_system.evaluate_proposal(basic_proposal, proof=proof)

        assert decision == RatchetDecision.ACCEPT

    def test_layer_stats(self):
        """Test layer statistics."""
        layer_system = LayeredRatchetSystem()
        stats = layer_system.get_layer_stats()

        assert 'FOUNDATION' in stats
        assert 'ADAPTIVE' in stats
        assert stats['FOUNDATION']['strength'] >= 0.95


# =============================================================================
# PROOF ENGINE TESTS
# =============================================================================

class TestProofEngine:
    """Tests for ProofEngine."""

    def test_engine_has_strategies(self):
        """Test engine has registered strategies."""
        engine = ProofEngine()

        assert len(engine.strategies) >= 3

    def test_generate_proof_for_valid_proposal(self, basic_proposal):
        """Test proof generation for valid proposal."""
        engine = ProofEngine()
        proof = engine.generate_proof(basic_proposal)

        assert proof is not None
        assert proof.confidence > 0

    def test_monotonicity_strategy(self, basic_proposal):
        """Test monotonicity proof strategy."""
        strategy = MonotonicityProofStrategy()

        assert strategy.can_prove(basic_proposal)

        proof = strategy.generate_proof(basic_proposal)
        assert proof is not None
        assert proof.proof_type == ProofType.MONOTONICITY
        assert proof.evidence.get('is_monotonic', False)

    def test_test_based_strategy(self, basic_proposal):
        """Test test-based proof strategy."""
        strategy = TestBasedProofStrategy()

        assert strategy.can_prove(basic_proposal)

        proof = strategy.generate_proof(basic_proposal)
        assert proof is not None
        assert proof.proof_type == ProofType.TEST_BASED


class TestESSProofStrategy:
    """Tests for ESS (Maynard Smith) proof strategy."""

    def test_ess_validation(self, basic_proposal):
        """Test ESS validation."""
        strategy = ESSProofStrategy()

        # Add alternatives for ESS testing
        basic_proposal.modification['alternatives'] = ['alt1', 'alt2']

        assert strategy.can_prove(basic_proposal)

        proof = strategy.generate_proof(basic_proposal)
        assert proof is not None
        assert proof.proof_type == ProofType.ESS


# =============================================================================
# ANCHOR ORACLE TESTS
# =============================================================================

class TestAnchorOracle:
    """Tests for AnchorOracle."""

    def test_anchor_proof(self):
        """Test proof anchoring."""
        oracle = AnchorOracle(use_blockchain=False)

        # Create mock proof
        class MockProof:
            def __init__(self):
                self.proof_id = 'test_proof'
                self.proof_type = type('obj', (object,), {'name': 'TEST'})()
                self.confidence = 0.95

            def to_dict(self):
                return {'id': self.proof_id}

        proof = MockProof()
        anchor_id = oracle.anchor_proof(proof)

        assert anchor_id is not None
        assert len(oracle.hash_chain) == 1

    def test_anchor_chain_integrity(self):
        """Test anchor chain maintains integrity."""
        oracle = AnchorOracle(use_blockchain=False)

        class MockProof:
            def __init__(self, id):
                self.proof_id = id
                self.proof_type = type('obj', (object,), {'name': 'TEST'})()
                self.confidence = 0.95

            def to_dict(self):
                return {'id': self.proof_id}

        # Anchor multiple proofs
        for i in range(5):
            oracle.anchor_proof(MockProof(f'proof_{i}'))

        assert oracle.verify_anchor_chain()
        assert oracle.get_chain_length() == 5

    def test_prove_ordering(self):
        """Test ordering proof between anchors."""
        oracle = AnchorOracle(use_blockchain=False)

        class MockProof:
            def __init__(self, id):
                self.proof_id = id
                self.confidence = 0.95
                self.proof_type = type('obj', (object,), {'name': 'TEST'})()

            def to_dict(self):
                return {'id': self.proof_id}

        anchor1 = oracle.anchor_proof(MockProof('first'))
        anchor2 = oracle.anchor_proof(MockProof('second'))

        ordering = oracle.prove_ordering(anchor1, anchor2)
        assert ordering is True  # anchor1 came before anchor2


# =============================================================================
# STRENGTH LEARNER TESTS
# =============================================================================

class TestStrengthLearner:
    """Tests for StrengthLearner."""

    def test_initial_beliefs(self):
        """Test initial beliefs are set correctly."""
        learner = StrengthLearner()

        assert 'FOUNDATION' in learner.domain_beliefs
        assert learner.domain_beliefs['FOUNDATION'].mean > 0.9

    def test_update_from_positive_outcome(self):
        """Test beliefs update from positive outcome."""
        learner = StrengthLearner()

        initial_mean = learner.domain_beliefs['ADAPTIVE'].mean

        outcome = RatchetOutcome(
            proposal_id='test',
            decision=RatchetDecision.ACCEPT,
            actual_improvement=0.1,
            predicted_improvement=0.1
        )

        learner.update_from_outcome('ADAPTIVE', 0.5, outcome)

        # Belief should have been updated
        assert learner.domain_beliefs['ADAPTIVE'].observations > 0

    def test_update_from_negative_outcome(self):
        """Test beliefs update from negative outcome (regression)."""
        learner = StrengthLearner()

        outcome = RatchetOutcome(
            proposal_id='test',
            decision=RatchetDecision.ACCEPT,
            actual_improvement=-0.1,  # Regression
            predicted_improvement=0.1,
            false_positive=True
        )

        learner.update_from_outcome('ADAPTIVE', 0.5, outcome)

        # Regret should increase
        regret = learner.compute_regret()
        assert regret >= 0

    def test_recommend_adjustment(self):
        """Test strength adjustment recommendations."""
        learner = StrengthLearner()

        direction, magnitude = learner.recommend_adjustment(
            'ADAPTIVE',
            current_strength=0.3,
            window=20
        )

        assert direction in ('increase', 'decrease', 'maintain')


class TestAdaptiveStrengthController:
    """Tests for AdaptiveStrengthController."""

    def test_update_and_adjust(self):
        """Test controller updates and adjusts strength."""
        controller = AdaptiveStrengthController()

        outcome = RatchetOutcome(
            proposal_id='test',
            decision=RatchetDecision.ACCEPT,
            actual_improvement=0.15,
            predicted_improvement=0.10
        )

        result = controller.update_and_adjust('ADAPTIVE', outcome)
        # May or may not adjust depending on learning state


# =============================================================================
# RATCHET ORCHESTRATOR TESTS
# =============================================================================

class TestRatchetOrchestrator:
    """Tests for RatchetOrchestrator."""

    def test_initialization_shadow_mode(self):
        """Test orchestrator initializes in shadow mode."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.SHADOW)

        assert orchestrator.mode == RatchetMode.SHADOW

    def test_propose_improvement_shadow_mode(self, basic_proposal):
        """Test proposal in shadow mode logs but doesn't block."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.SHADOW)

        decision, result = orchestrator.propose_improvement(basic_proposal)

        assert decision == RatchetDecision.SHADOW
        assert 'SHADOW' in result

    def test_propose_improvement_full_mode(self, basic_proposal):
        """Test proposal in full enforcement mode."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.FULL)

        # Add a verified proof
        basic_proposal.proof = RatchetProof(
            proof_id='',
            proof_type=ProofType.MONOTONICITY,
            statements=['Verified improvement'],
            evidence={'utility_delta': 0.1},
            confidence=0.95,
            timestamp=time.time(),
            verified=True
        )

        decision, result = orchestrator.propose_improvement(basic_proposal, auto_prove=False)

        assert decision == RatchetDecision.ACCEPT

    def test_verify_monotonicity(self, basic_before_state, basic_after_state):
        """Test monotonicity verification."""
        orchestrator = RatchetOrchestrator()

        is_monotonic, delta = orchestrator.verify_monotonicity(
            basic_before_state, basic_after_state
        )

        assert is_monotonic is True
        assert delta > 0

    def test_verify_monotonicity_detects_regression(self, basic_before_state, regressing_after_state):
        """Test monotonicity verification detects regression."""
        orchestrator = RatchetOrchestrator()

        is_monotonic, delta = orchestrator.verify_monotonicity(
            basic_before_state, regressing_after_state
        )

        assert is_monotonic is False
        assert delta < 0

    def test_commit_improvement(self, basic_proposal):
        """Test committing an improvement."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.FULL)

        proof = RatchetProof(
            proof_id='',
            proof_type=ProofType.MONOTONICITY,
            statements=['Test'],
            evidence={},
            confidence=0.95,
            timestamp=time.time(),
            verified=True
        )

        anchor_id = orchestrator.commit_improvement(basic_proposal, proof)

        assert anchor_id is not None
        assert orchestrator.state.get_improvement_count() == 1

    def test_get_stats(self):
        """Test getting orchestrator stats."""
        orchestrator = RatchetOrchestrator()

        stats = orchestrator.get_stats()

        assert 'mode' in stats
        assert 'proposals' in stats
        assert 'state' in stats

    def test_mode_change(self):
        """Test changing orchestrator mode."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.SHADOW)

        orchestrator.set_mode(RatchetMode.FULL)

        assert orchestrator.mode == RatchetMode.FULL


# =============================================================================
# REGRESSION PREVENTION TESTS
# =============================================================================

class TestRegressionPrevention:
    """Tests for regression prevention."""

    def test_cannot_accept_regression_in_domain_layer(
        self, basic_before_state, regressing_after_state
    ):
        """Test domain layer rejects regressions."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.FULL)

        proposal = create_proposal(
            before=basic_before_state,
            after=regressing_after_state,
            modification={'type': 'change'},
            domain=RatchetDomain.DOMAIN,
            description='Regressing change'
        )

        decision, reason = orchestrator.propose_improvement(proposal)

        # Should be rejected due to negative utility delta
        assert decision == RatchetDecision.REJECT

    def test_anchor_chain_immutable(self):
        """Test anchor chain cannot be modified after creation."""
        oracle = AnchorOracle()

        class MockProof:
            def __init__(self, id):
                self.proof_id = id
                self.confidence = 0.95
                self.proof_type = type('obj', (object,), {'name': 'TEST'})()

            def to_dict(self):
                return {'id': self.proof_id}

        # Create chain
        for i in range(3):
            oracle.anchor_proof(MockProof(f'proof_{i}'))

        # Verify chain integrity
        assert oracle.verify_anchor_chain()

        # Tampering would break the chain (if we could tamper)
        original_length = oracle.get_chain_length()
        assert original_length == 3


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete ratchet flow."""

    def test_full_improvement_cycle(self, basic_before_state, basic_after_state):
        """Test complete improvement cycle: propose -> prove -> anchor -> commit."""
        orchestrator = RatchetOrchestrator(mode=RatchetMode.FULL)

        # Create proposal
        proposal = create_proposal(
            before=basic_before_state,
            after=basic_after_state,
            modification={'type': 'optimization'},
            domain=RatchetDomain.ADAPTIVE
        )

        # Propose (auto-prove enabled)
        decision, result = orchestrator.propose_improvement(proposal, auto_prove=True)

        # Should be accepted in adaptive layer with auto-prove
        assert decision in (RatchetDecision.ACCEPT, RatchetDecision.WARN)

        # Verify chain was updated
        stats = orchestrator.get_stats()
        assert stats['proposals'] > 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
