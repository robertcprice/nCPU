"""nCPU: Neural CPU — every CPU component replaced by trained neural networks.

Preferred entry points for the hero experience:
    python -m ncpu gpu                 # THE HERO: GPU as complete self-sufficient computer
    python -m ncpu lab / ncpu-lab      # Unified launcher

Three execution strategies:

1. Neural (ncpu.neural): Full neural CPU with 21+ trained models replacing
   ALU, decoder, register file, memory, shifts, math, and more. All
   components are PyTorch models running on GPU.

2. Model-based (ncpu.model): Uses a fine-tuned LLM to semantically decode
   assembly instructions into verified operation keys. Proves that neural
   networks can replace hardcoded instruction decode logic.

3. Tensor-based (ncpu.tensor): Full ARM64 emulation using pure GPU tensor
   operations. Registers, memory, PC, and flags all live as tensors for
   massively parallel execution.
"""

__version__ = "0.2.0"
__author__ = "Robert Price"
