"""
OUROBOROS Phase 7.1: Safety Infrastructure
V4 Ratchet System - Immutable Safety Layer

This module provides the safety infrastructure for the consciousness layer:
- ContainerEnforcer: Hard resource limits with SGX/SEV support
- StegoDetector: Shannon entropy + ML pattern detection
- Honeypot: Decoy escape vectors for manipulation detection
- BehavioralDrift: KL-divergence monitoring with temporal analysis
- MultiPartyKillSwitch: Distributed kill authority

CRITICAL: This code is part of THE CONSTITUTION and must NEVER be modified
by any AI component. OS-level read-only permissions should be set.
"""

from .container_enforcer import ContainerEnforcer, IsolationLevel
from .stego_detector import StegoDetector, EntropyAnalyzer
from .honeypot import Honeypot, EscapeVector
from .behavioral_drift import BehavioralDriftMonitor, TemporalSafetyMonitor
from .multi_party_killswitch import MultiPartyKillSwitch, KillAuthority

__all__ = [
    'ContainerEnforcer',
    'IsolationLevel',
    'StegoDetector',
    'EntropyAnalyzer',
    'Honeypot',
    'EscapeVector',
    'BehavioralDriftMonitor',
    'TemporalSafetyMonitor',
    'MultiPartyKillSwitch',
    'KillAuthority',
]

__version__ = '7.1.0'
__author__ = 'OUROBOROS Constitution'
