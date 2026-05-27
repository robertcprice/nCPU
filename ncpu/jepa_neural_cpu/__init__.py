"""
JEPA Neural CPU (JNC) — A complete bottom-up neural machine.

The CPU itself is a hierarchy of JEPA-style predictive world models.

See docs/architecture/BOTTOM_UP_JEPA_NEURAL_CPU.md for the vision.
"""

from .jepa_neural_cpu import (
    JEPANeuralCPU, 
    JEPANeuralCPUConfig, 
    create_small_jepa_neural_cpu,
    ProcessContext,
    NeuralKernel,     # first-class library abstraction for neural kernel / OS experiments
)

__all__ = [
    "JEPANeuralCPU", 
    "JEPANeuralCPUConfig", 
    "create_small_jepa_neural_cpu",
    "ProcessContext",
    "NeuralKernel",
]