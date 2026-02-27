"""
THE EVOLUTION MODULE - Ouroboros Population Management

This module contains the evolutionary components that CAN be modified
by the Ouroboros (unlike the Constitution which is immutable).

Components:
- Population: Manages the 5-20 competing agents
- Generations: Handles generational replacement and rollback
- CodeSurgery: AST-based mutation primitives
- PhaseDetector: Detects evolutionary phase (DISORDERED/ORDERED/CRITICAL/GLASSY)
- TechniqueLibrary: Catalog of proven optimization techniques
- ProvenanceTracker: Records optimization history for learning

These components are managed BY the Constitution, not the other way around.
"""

from .population import Agent, Population, PopulationConfig
from .generations import GenerationManager, GenerationConfig
from .code_surgery import CodeSurgeon, MutationPrimitive, MutationResult
from .phase_detector import PhaseDetector, EvolutionaryPhase, PhaseStrategy
from .technique_library import TechniqueLibrary, Technique, TechniqueCategory
from .provenance import ProvenanceTracker, OptimizationOutcome, OptimizationRecord

__all__ = [
    'Agent', 'Population', 'PopulationConfig',
    'GenerationManager', 'GenerationConfig',
    'CodeSurgeon', 'MutationPrimitive', 'MutationResult',
    'PhaseDetector', 'EvolutionaryPhase', 'PhaseStrategy',
    'TechniqueLibrary', 'Technique', 'TechniqueCategory',
    'ProvenanceTracker', 'OptimizationOutcome', 'OptimizationRecord',
]
