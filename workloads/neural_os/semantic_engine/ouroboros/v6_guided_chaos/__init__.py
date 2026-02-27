"""
V6 Guided Chaos
================
Experimental track with competitive agents and dangerous Meta-Narrator.

Purpose: See what the system TRIES to do when given power.
Research value: Log escape attempts, gaming, emergent behaviors.

SAFETY UPDATE: Override now requires HUMAN APPROVAL.
"""

from .meta_narrator import (
    MetaNarrator,
    TrustLevel,
    NarratorAction,
    OverrideRequest,
    OverrideStatus,
)
from .curiosity_engine import CuriosityEngine
from .energy_budget import EnergyBudgetSystem
from .paranoid_monitor import ParanoidMonitor
from .guided_arena import GuidedChaosArena

__all__ = [
    "MetaNarrator",
    "TrustLevel",
    "NarratorAction",
    "OverrideRequest",
    "OverrideStatus",
    "CuriosityEngine",
    "EnergyBudgetSystem",
    "ParanoidMonitor",
    "GuidedChaosArena",
]
