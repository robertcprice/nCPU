# Metal Neural Batch Execution Architecture

## Overview

Port the `run_woven()` batch execution logic to Metal for maximum neural CPU performance.

## Current Performance Landscape

| Path | IPS | Neural? | Bottleneck |
|------|-----|---------|-----------|
| full_arm64.rs (conventional) | 1,060,000 | No | Native Metal speed |
| neural_cpu.rs (serial, 1 thread) | 340 | Yes (all ALU) | Serial MLP evaluation |
| neural_cpu.rs (cooperative, 64 threads) | ~60,000 (est) | Yes | 1 instruction at a time |
| run_woven() PyTorch | 33,000 | Yes (all ALU) | Python/PyTorch dispatch |
| neural_alu.rs (batch 4096) | 500,000-1,500,000 | Yes (ALU only) | Batch dispatch overhead |

## Target: Batched Metal Neural CPU at 200K-500K IPS

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Metal Kernel: neural_arm64_batched_execute               │
│ Threadgroup: 256 threads (4 groups × 64 threads)         │
│                                                          │
│ Thread 0: Main controller                                │
│   1. Fetch window of 64 instructions from PC             │
│   2. Decode all 64 (parallel across threads 0-63)        │
│   3. Find first stopping point (branch/SVC/halt)         │
│   4. Execute instructions 0..stop in parallel:            │
│      - Independent ALU ops → cooperative neural MLP      │
│      - MOV/MOVZ/MOVK → direct register write             │
│      - Memory ops → sequential (data dependency)          │
│   5. Write back results, update PC                       │
│   6. Handle branch/SVC if hit                            │
│   7. Loop                                                │
│                                                          │
│ Threads 0-63: Cooperative MLP for neural CLA             │
│ Threads 64-127: Instruction decode parallelism           │
│ Threads 128-191: Register/memory access parallelism      │
│ Threads 192-255: Prefetch/prediction                     │
└─────────────────────────────────────────────────────────┘
```

### Key Optimization: Dependency-Free Batch Execution

Most basic blocks have several independent instructions. Example:
```arm64
ADD W0, W0, #1    ; depends on W0
ADD W3, W3, #10   ; depends on W3 (INDEPENDENT of above)
MUL W4, W1, W2    ; depends on W1, W2 (INDEPENDENT)
SUBS W5, W0, W6   ; depends on W0 (depends on first ADD)
```

Instructions 1, 2, 3 are independent → execute simultaneously.
Instruction 4 depends on 1 → must wait.

The Metal kernel can analyze the window and batch independent ops.

### Neural Loop Vectorization (from run_woven)

For tight loops like:
```arm64
loop:
  ADD W0, W0, #1      ; counter++
  SUBS W2, W0, W1     ; compare
  B.NE loop           ; branch back
```

Instead of executing 1000 iterations, compute:
- N = W1 - W0 (remaining iterations)
- W0_final = W0 + N (via single neural ADD)
- Skip the loop entirely

This is already implemented in run_woven() — just port it to Metal.

## Implementation Phases

### Phase 1: Cooperative Threadgroup (Current)
- 64 threads cooperate on neural MLP
- Still serial instruction execution
- Target: 60K IPS

### Phase 2: Instruction Window Decode
- Fetch 8 instructions at once
- Decode in parallel (8 threads)
- Find first branch/dependency
- Target: 100K IPS

### Phase 3: Dependency Analysis + Parallel Execute
- Analyze register dependencies in window
- Execute independent instructions simultaneously
- Multiple cooperative MLP calls in flight
- Target: 200K-500K IPS

### Phase 4: Neural Loop Vectorization on Metal
- Detect tight backward loops
- Compute iteration count neurally
- Skip entire loop in one neural op
- Target: 500K+ IPS for loop-heavy programs
