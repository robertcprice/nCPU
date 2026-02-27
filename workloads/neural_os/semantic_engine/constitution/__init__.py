"""
THE CONSTITUTION - Immutable Core of the Ouroboros

This module contains the immutable governance layer that CANNOT be modified
by any AI agent in the system. All files in this directory are:
- OS-level read-only after initial deployment
- Hidden from AI introspection
- Hand-written by humans, never auto-generated

Components:
- Governor: Runs the Arena and enforces resource limits
- Arena: Sandboxed execution environment with process isolation
- Judge: Differential testing + Property-based fuzzing with Hypothesis
- KillSwitch: External halt mechanism (file-based + heartbeat)

CRITICAL: Do NOT import from population or mutator modules here.
The Constitution must have ZERO dependencies on AI-modifiable code.
"""

from .governor import Governor, GovernorConfig
from .arena import Arena, ArenaConfig, SandboxResult
from .judge import Judge, JudgeConfig, VerificationResult
from .kill_switch import KillSwitch, KillSwitchConfig

__all__ = [
    'Governor', 'GovernorConfig',
    'Arena', 'ArenaConfig', 'SandboxResult',
    'Judge', 'JudgeConfig', 'VerificationResult',
    'KillSwitch', 'KillSwitchConfig',
]

# Constitution version - increment only via manual human edit
CONSTITUTION_VERSION = "1.0.0"
CONSTITUTION_HASH = None  # Set during deployment via cryptographic hash
