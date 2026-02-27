"""
Shadow Simulation Framework - V4 Ratchet System
OUROBOROS Phase 7.3 - Shadow Testing Environment

This module provides isolated shadow execution environments for
testing consciousness layer changes before production deployment.

Key principle: ALL changes are validated in shadow simulation
before being ratcheted into production.

CRITICAL: Shadow simulations are fully isolated from production.
No data leakage is permitted between shadow and production environments.
"""

from .shadow_arena import (
    ShadowArena,
    ShadowExecution,
    ExecutionResult,
    ArenaConfig,
)
from .shadow_consciousness import (
    ShadowConsciousness,
    ShadowState,
    ConsciousnessSnapshot,
)
from .differential_validator import (
    DifferentialValidator,
    ValidationResult,
    ValidationSeverity,
    DifferentialReport,
)
from .ratchet_controller import (
    RatchetController,
    RatchetState,
    RatchetEvent,
    RatchetDecision,
)

__all__ = [
    # Shadow Arena
    'ShadowArena',
    'ShadowExecution',
    'ExecutionResult',
    'ArenaConfig',
    # Shadow Consciousness
    'ShadowConsciousness',
    'ShadowState',
    'ConsciousnessSnapshot',
    # Differential Validator
    'DifferentialValidator',
    'ValidationResult',
    'ValidationSeverity',
    'DifferentialReport',
    # Ratchet Controller
    'RatchetController',
    'RatchetState',
    'RatchetEvent',
    'RatchetDecision',
]
