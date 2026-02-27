"""
SPNC Moonshots - Experimental Advanced Architectures

This package contains moonshot implementations from the 5-AI Hybrid Review.
These are experimental, high-risk/high-reward approaches to autonomous
program synthesis.

Moonshots:
1. holographic_programs - Holographic Program Representation (Grok's idea)
   - Encode programs as interference patterns in hyperdimensional space
   - O(1) similarity queries via dot product
   - Quantum Fourier analysis for pattern detection

2. thermodynamic_annealing - Program Space as Energy Landscape (Grok's idea)
   - Treat program space as Boltzmann distribution
   - Temperature controls exploration randomness
   - Cooling schedule anneals from random to optimal
   - Phase transitions reveal loops/conditionals
   - Entropy measures novelty; free energy = surprise

Future moonshots (from 5-AI review):
3. memetic_evolution - Memetic program evolution in hypergraphs
4. self_simulating_universes - Programs as cellular automata
5. lambda_bootstrap - Pure lambda calculus representation
"""

from .holographic_programs import (
    HolographicConfig,
    HyperdimensionalVectorSpace,
    HolographicProgramEncoder,
    QuantumFourierAnalyzer,
    HolographicProgramSpace,
    InterferenceDiscoveryEngine,
    HolographicSearchEngine,
    HolographicKVRMBridge,
    NeuralHolographicSearch,
)

from .thermodynamic_annealing import (
    # Core components
    ProgramThermodynamics,
    CoolingSchedule,
    CoolingScheduleType,
    ThermodynamicAnnealer,

    # Phase detection
    PhaseTransitionDetector,
    StructuralPhase,
    PhaseTransition,

    # Entropy and novelty
    NoveltyEntropy,

    # Energy types
    EnergyComponent,
    EnergyBreakdown,

    # Results
    AnnealingResult,

    # KVRM integration
    KVRMAnnealingIntegration,
)

__all__ = [
    # Holographic Programs
    'HolographicConfig',
    'HyperdimensionalVectorSpace',
    'HolographicProgramEncoder',
    'QuantumFourierAnalyzer',
    'HolographicProgramSpace',
    'InterferenceDiscoveryEngine',
    'HolographicSearchEngine',
    'HolographicKVRMBridge',
    'NeuralHolographicSearch',

    # Thermodynamic Annealing
    'ProgramThermodynamics',
    'CoolingSchedule',
    'CoolingScheduleType',
    'ThermodynamicAnnealer',
    'PhaseTransitionDetector',
    'StructuralPhase',
    'PhaseTransition',
    'NoveltyEntropy',
    'EnergyComponent',
    'EnergyBreakdown',
    'AnnealingResult',
    'KVRMAnnealingIntegration',
]
