"""
NeuralCPU package - modularized from the original cpu.py monolith.

Every component of the CPU is a trained neural network running on GPU.
The NeuralCPU class is composed from mixins in separate modules:

    core.py       - NeuralCPU class, __init__, run() router
    constants.py  - OpType enum, device detection, helpers
    extractors.py - Neural bit-level extractors (MOVZ, Branch26, Branch19)
    prediction.py - Branch trace buffer, dispatcher, optimizer, branch decider
    decoder.py    - Instruction fetch/decode/operand extraction
    vectorizer.py - Loop detection and vectorization
    kernel.py     - System registers, MMU/TLB, GIC, UART, MMIO
    telemetry.py  - Execution statistics and monitoring
    training.py   - Self-supervised neural component training
    engines/      - Execution engines:
        step.py     - Legacy single-step
        fast.py     - NumPy CPU fast path (1M+ IPS)
        parallel.py - GPU parallel execution
        weave.py    - Neural weave (trained .pt model routing)
        pipeline.py - Full neural pipeline
        gpu_only.py - Zero-sync GPU execution

Usage:
    from ncpu.neural.cpu import NeuralCPU
    cpu = NeuralCPU()
"""

from .core import NeuralCPU
from .constants import OpType, device, _u64_to_s64

__all__ = ["NeuralCPU", "OpType", "device", "_u64_to_s64"]
