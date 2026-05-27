# nCPU Architecture

A CPU where every component is a trained neural network.

## System Overview

nCPU has two execution modes, both GPU-native:

```
                        ┌─────────────────────────────────────────┐
                        │           ncpu.neural.NeuralCPU         │
                        │         (12K lines, real ARM64)          │
                        │                                         │
  ARM64 Binary ────────►│  Registers [32] ── torch.int64 tensor   │
                        │  Memory [1MB]   ── torch.uint8 tensor   │
                        │  Flags [4]      ── torch.float tensor   │
                        │  PC             ── torch.int64 tensor   │
                        │                                         │
                        │  ┌─────────────────────────────────┐    │
                        │  │      Neural ALU Bridge          │    │
                        │  │  ADD/SUB → arithmetic.pt        │    │
                        │  │  MUL     → multiply.pt          │    │
                        │  │  AND/OR/XOR → logical.pt        │    │
                        │  │  LSL/LSR → lsl.pt/lsr.pt        │    │
                        │  │  CMP     → neural subtraction   │    │
                        │  │  sin/cos/sqrt/exp/log/atan2     │    │
                        │  └─────────────────────────────────┘    │
                        │                                         │
                        │  Fast mode: native GPU tensor ops       │
                        │  Neural mode: all ops through .pt models│
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
  Text Assembly ───────►│            ncpu.model.CPU               │
  (MOV R0, 42)          │      (text ISA, neural execution)       │
                        │                                         │
                        │  Decode: regex parser → operation key   │
                        │  Execute: NeuralRegistry                │
                        │    └── NeuralOps (trained .pt models)   │
                        │  State: immutable CPUState dataclass    │
                        └─────────────────────────────────────────┘
```

## Execution Pipeline

### Neural Mode (default)

```
1. FETCH:   instruction_bits = memory[PC:PC+4]
2. DECODE:  Neural extractors + lookup tables → OpType, rd, rn, rm, imm
3. EXECUTE: Neural ALU Bridge → trained .pt model inference
4. COMMIT:  Write result to GPU register tensor
5. ADVANCE: PC += 4 (or branch offset)
```

Every ALU operation passes through a trained neural network:
- **ADD X0, X1, X2**: Kogge-Stone CLA (arithmetic.pt + carry_combine.pt + logical.pt) — 8 neural passes
- **MUL X0, X1, X2**: multiply.pt performs up to 16 byte-pair LUT lookups
- **AND X0, X1, X2**: logical.pt does single vectorized truth table lookup
- **LSL X0, X1, X2**: lsl.pt runs shift_decoder → index_net → validity_net (3 batched passes)
- **CMP X0, X1**: neural CLA subtraction through carry_combine.pt, flags derived from result

### Fast Mode (--fast)

Same NeuralCPU, but ALU operations use native GPU tensor arithmetic (`+`, `-`, `*`, `&`, `|`, `^`, `<<`, `>>`). Decode, memory, registers, branches — all still on GPU as tensors. Only the ALU computation path changes.

## Neural ALU Bridge

The bridge connects the 64-bit GPU-resident NeuralCPU to the 32-bit trained models:

```
NeuralCPU (torch.int64)
    │
    ├── 64→32 bit narrowing (truncate upper 32 bits)
    ├── Dispatch to trained model
    │     ├── arithmetic.pt (ADD/SUB/INC/DEC)
    │     ├── multiply.pt (MUL)
    │     ├── logical.pt (AND/OR/XOR)
    │     ├── lsl.pt / lsr.pt (SHL/SHR)
    │     └── math models (sin/cos/sqrt/exp/log/atan2)
    └── Result → Python int → assigned back to register tensor
```

## Model Wiring

| ARM64 Instruction | Model | Method | Latency |
|-------------------|-------|--------|---------|
| ADD (imm/reg) | arithmetic.pt + carry_combine.pt | `bridge.add(a, b)` | 248 us |
| SUB (imm/reg) | arithmetic.pt + carry_combine.pt | `bridge.sub(a, b)` | 246 us |
| MUL | multiply.pt | `bridge.mul(a, b)` | 21 us |
| AND (reg) | logical.pt | `bridge.and_(a, b)` | 21 us |
| ORR (reg) | logical.pt | `bridge.or_(a, b)` | 22 us |
| EOR (reg) | logical.pt | `bridge.xor_(a, b)` | 21 us |
| LSL (reg/imm) | lsl.pt | `bridge.shl(val, amt)` | 437 us |
| LSR (reg/imm) | lsr.pt | `bridge.shr(val, amt)` | 431 us |
| CMP (imm/reg) | carry_combine.pt | `bridge.cmp(a, b)` → flags | 249 us |
| sin | sincos.pt | `bridge.sin(a)` | 48 us |
| cos | sincos.pt | `bridge.cos(a)` | 48 us |
| sqrt | sqrt.pt | `bridge.sqrt(a)` | 522 us |
| exp | exp.pt | `bridge.exp_(a)` | 21 us |
| log | log.pt | `bridge.log_(a)` | 21 us |
| atan2 | atan2.pt | `bridge.atan2(y, x)` | 935 us |

## GPU-Resident State

All CPU state lives on GPU as tensors — no CPU↔GPU transfer during execution except instruction fetch:

| Component | Type | Shape | Device |
|-----------|------|-------|--------|
| Registers (X0-X30) | `torch.int64` | `[32]` | GPU |
| Memory | `torch.uint8` | `[1048576]` | GPU |
| Flags (N,Z,C,V) | `torch.float32` | `[4]` | GPU |
| Program Counter | `torch.int64` | scalar | GPU |
| Stack Pointer | `torch.int64` | scalar | GPU |
| Branch Trace Buffer | `torch.int64` | `[2048, 4]` | GPU |
| Instruction Sequence | `torch.float32` | `[256, 32]` | GPU |

## Performance Characteristics

Per-operation latency on Apple Silicon MPS (1,000 iterations, PyTorch 2.10.0):

```
O(1) Operations — Single Forward Pass (~21 us)
  ├── exp/log:     21 us   (4-layer MLP)
  ├── mul:         21 us   (batched byte-pair LUT)
  └── and/or/xor:  21 us   (vectorized truth table)

O(2) Operations — Few Passes (~48 us)
  └── sin/cos:     48 us   (5 sine-activated blocks)

O(log n) Operations — Kogge-Stone CLA (~248 us)
  └── add/sub/cmp: 248 us   (8 neural passes: 2 logical + 5 carry-combine + 1 XOR)

O(3) Operations — Batched Passes (~434 us)
  └── shl/shr:    434 us   (3 batched attention passes, vectorized)

O(n) Operations — Sequential Passes
  ├── sqrt:       522 us   (2 stages + BatchNorm batch padding)
  └── atan2:      935 us   (6 residual layers + batch padding)
```

**Key insight**: Neural MUL (21 us) is **12x faster** than neural ADD (248 us).
In conventional CPUs, MUL is typically 3-10x *slower* than ADD. The inversion
persists even with the CLA optimization because the byte-pair LUT achieves O(1)
while carry-lookahead is O(log n).

**CLA speedup**: Kogge-Stone parallel-prefix carry-lookahead reduced ADD/SUB/CMP
from ~826 us (32 ripple-carry passes) to ~248 us (8 CLA passes) — a **3.3x speedup**.

Shift operations were vectorized from 64 sequential per-bit passes (2,833 us) to 3
batched forward passes (434 us) --- a **6.5x speedup**. Shifts are now faster than sqrt.

All 22 models cold-load in **60ms**. Programs execute at ~201 us/cycle average (~4,975 IPS).

## Component Map

```
ncpu/
├── neural/                      # Full GPU neural CPU
│   ├── cpu.py                   # NeuralCPU (12K lines) — ARM64 execution engine
│   ├── neural_alu_bridge.py     # Bridge: 64-bit tensors ↔ 32-bit trained models
│   ├── memory_oracle.py         # LSTM memory access predictor
│   └── semantic_dispatcher.py   # Neural instruction dispatch
│
├── model/                       # Model-based CPU (text assembly)
│   ├── cpu.py                   # CPU orchestrator
│   ├── state.py                 # Immutable CPUState
│   ├── registry.py              # Verified operation registry
│   ├── decode.py                # Text assembly parser
│   ├── neural_ops.py            # Neural ALU operations (loads .pt models)
│   └── architectures.py         # All model class definitions
│
├── tensor/                      # Pure tensor ARM64 kernel
│   ├── cpu.py                   # Tensor CPU
│   └── kernel.py                # TensorKernel — batch execution
│
models/                          # Trained weights
├── alu/                         # 6 ALU models (incl. carry_combine)
├── shifts/                      # 4 shift/rotate models
├── register/                    # 2 register models (archived)
├── memory/                      # 3 memory models (archived)
├── decoder/                     # 1 decoder model (archived)
├── math/                        # 6 math models (wired)
└── decode_llm/                  # Qwen2.5 LoRA adapter
```
