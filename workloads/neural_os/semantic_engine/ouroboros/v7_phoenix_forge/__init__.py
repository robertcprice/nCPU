"""
V7 Phoenix Forge
=================
Productive track with cooperative agents and safe Consensus Oracle.

Purpose: Generate useful creative output through cooperation.
Design: No override capability - democratic decision making.
"""

from .consensus_oracle import ConsensusOracle, Proposal, ConsensusResult
from .world_model import HierarchicalWorldModel
from .free_energy import FreeEnergyMinimizer
from .blackboard import Blackboard, Hypothesis
from .map_elites import MAPElites, Niche
from .phoenix_forge import PhoenixForgeArena

__all__ = [
    "ConsensusOracle",
    "Proposal",
    "ConsensusResult",
    "HierarchicalWorldModel",
    "FreeEnergyMinimizer",
    "Blackboard",
    "Hypothesis",
    "MAPElites",
    "Niche",
    "PhoenixForgeArena",
]
