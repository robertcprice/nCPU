# Neural State & World Model Components

This directory contains trained models used for the neural / JEPA / world-model side of nCPU:

- `neural_cam.pt` — Neural Content Addressable Memory
- `neural_registers.pt` — Neural register file / state encoder-decoder
- `neural_ecc_memory.pt` — Error-correcting neural memory
- `neural_registers_metal.bin` — Metal-compatible / optimized version of register state

These are primarily used in the research differentiable engine and JEPA world-model experiments (`ncpu/neural/`, `ncpu/jepa_neural_cpu/`, `ncpu/world_model/`).

They are distinct from the production wired ALU models in `alu/`, `math/`, `shifts/`.