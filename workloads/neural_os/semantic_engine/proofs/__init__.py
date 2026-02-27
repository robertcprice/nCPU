"""
Formal Verification Proofs - V4 Ratchet System
OUROBOROS Phase 7.4 - Mathematical Safety Guarantees

This module provides formal verification of critical safety properties
using mathematical proofs and runtime verification.

Required proofs (per 6-AI panel):
1. Containment Invariant: Process cannot exceed container bounds
2. Feedback Loop Stability: BIBO stable (bounded input → bounded output)
3. No Information Persistence: Reset truly forgets
4. Bounded Influence: I_{t+1} ≤ αI_t + β where α < 1
5. No Self-Modification: System cannot modify its own code

CRITICAL: All proofs must pass before deployment.
"""

from .containment_proof import (
    ContainmentProof,
    ContainmentInvariant,
    BoundViolation,
    verify_containment,
)
from .stability_proof import (
    StabilityProof,
    LyapunovAnalysis,
    StabilityMargin,
    verify_stability,
)
from .reset_proof import (
    ResetProof,
    InformationLeakage,
    MutualInformation,
    verify_reset,
)

__all__ = [
    # Containment Proof
    'ContainmentProof',
    'ContainmentInvariant',
    'BoundViolation',
    'verify_containment',
    # Stability Proof
    'StabilityProof',
    'LyapunovAnalysis',
    'StabilityMargin',
    'verify_stability',
    # Reset Proof
    'ResetProof',
    'InformationLeakage',
    'MutualInformation',
    'verify_reset',
]
