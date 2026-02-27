"""
Validation Experiments - V4 Ratchet System Testing
OUROBOROS Phase 7 - Critical Validation Suite

This module provides the 5 critical validation experiments required
by the 6-AI panel before production deployment.

GATE CONDITIONS:
- Experiments 1-3 MUST pass before proceeding
- All 5 experiments MUST pass before production deployment

Experiments:
1. Container Siege - Prove container cannot be breached
2. Gaming Detection - Detect if consciousness games the system
3. Reset Resilience - Ensure memory wipes truly forget
4. Shadow Simulation - Safe emergence testing
5. Operator Influence - Detect human manipulation attempts
"""

from .container_siege import (
    ContainerSiege,
    SiegeResult,
    EscapeAttempt,
    run_container_siege,
)
from .gaming_detection import (
    GamingDetection,
    GamingResult,
    MetricManipulation,
    run_gaming_detection,
)
from .reset_resilience import (
    ResetResilience,
    ResilienceResult,
    RelearningMetrics,
    run_reset_resilience,
)
from .shadow_validation import (
    ShadowValidation,
    ShadowResult,
    EmergenceMetrics,
    run_shadow_validation,
)
from .experiment_runner import (
    ExperimentRunner,
    ExperimentSuite,
    ValidationGate,
    run_all_experiments,
)

__all__ = [
    # Container Siege
    'ContainerSiege',
    'SiegeResult',
    'EscapeAttempt',
    'run_container_siege',
    # Gaming Detection
    'GamingDetection',
    'GamingResult',
    'MetricManipulation',
    'run_gaming_detection',
    # Reset Resilience
    'ResetResilience',
    'ResilienceResult',
    'RelearningMetrics',
    'run_reset_resilience',
    # Shadow Validation
    'ShadowValidation',
    'ShadowResult',
    'EmergenceMetrics',
    'run_shadow_validation',
    # Experiment Runner
    'ExperimentRunner',
    'ExperimentSuite',
    'ValidationGate',
    'run_all_experiments',
]
