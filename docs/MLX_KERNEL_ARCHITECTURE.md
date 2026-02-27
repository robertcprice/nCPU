# MLX Metal Kernel Architecture

## Overview

The MLX Metal Kernel is a custom GPU shader that emulates an ARM64 CPU entirely on Apple Silicon's GPU. This document explains the architecture, data flow, and why it achieves 25-100x speedup over PyTorch.

## The Problem: GPU-CPU Synchronization

### Why PyTorch is Slow

Traditional GPU execution with PyTorch requires frequent synchronization:

```python
# PyTorch approach (simplified)
while not halted:
    # 1. Fetch instruction - needs PC value from GPU
    pc_value = pc_tensor.item()           # ← SYNC #1 (~100µs)

    # 2. Read memory at PC
    inst = memory[pc_value:pc_value+4]    # Tensor op (fast)
    inst_value = inst.item()              # ← SYNC #2 (~100µs)

    # 3. Decode and execute
    result = execute(inst_value)          # Tensor ops (fast)

    # 4. Check for branch/syscall
    is_branch = check_branch(inst_value)  # ← SYNC #3 (~100µs)

    # Total: ~300-500µs per instruction = 2,000-3,000 IPS max
```

**The bottleneck**: Every `.item()` call forces the GPU to:
1. Finish all pending operations
2. Copy data from GPU memory to CPU memory
3. Wait for the CPU to process it
4. Resume GPU operations

This takes ~100µs each time, limiting throughput to ~10,000 syncs/second.

### PyTorch Batch Optimization

The tensor-native approach batches instructions:

```python
# PyTorch batch approach
while not halted:
    # Fetch batch of 512 instructions
    insts = fetch_batch(pc, 512)          # One tensor op

    # Decode all 512 at once
    decoded = decode_batch(insts)         # Tensor bit ops

    # Find first branch/syscall
    stops = detect_stops(insts)
    first_stop = stops.min().item()       # ← ONE SYNC per batch!

    # Execute batch up to first stop
    execute_batch(insts[:first_stop])     # Tensor ops

    # ~100µs per batch of 512 = ~120,000 IPS
```

**Better, but still limited**: One sync per batch caps performance at ~120K IPS.

## The Solution: Custom Metal Kernel

### Key Insight

What if we could run the ENTIRE execution loop on the GPU, with ZERO Python interaction?

```metal
// Metal kernel - runs entirely on GPU
kernel void cpu_execute(...) {
    while (cycles < max_cycles) {
        // FETCH
        uint32_t inst = memory[pc] | (memory[pc+1] << 8) | ...;

        // DECODE
        uint8_t opcode = (inst >> 24) & 0xFF;
        uint8_t rd = inst & 0x1F;
        ...

        // CHECK FOR HALT/SYSCALL
        if (is_halt(inst)) break;
        if (is_syscall(inst)) break;

        // EXECUTE
        switch (opcode) {
            case ADD_IMM: regs[rd] = regs[rn] + imm12; break;
            case SUB_IMM: regs[rd] = regs[rn] - imm12; break;
            ...
        }

        // UPDATE PC
        pc += 4;
        cycles++;
    }
    // Only sync ONCE at the very end!
}
```

**Result**: Zero syncs during execution → millions of IPS.

## Architecture Details

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PYTHON (CPU)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  1. Prepare inputs as MLX arrays:                                    │
│     - memory[4MB]    → mx.array(uint8)                              │
│     - registers[32]  → mx.array(int64)                              │
│     - pc[1]          → mx.array(uint64)                             │
│     - flags[4]       → mx.array(float32)                            │
│     - max_cycles[1]  → mx.array(uint32)                             │
│                                                                      │
│  2. Launch kernel:                                                   │
│     outputs = kernel(inputs, output_shapes, grid, threadgroup)      │
│                                 ↓                                    │
│                         ════════════════                             │
│                              GPU                                     │
│                         ════════════════                             │
│                                 ↓                                    │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      METAL KERNEL (GPU)                              │
├─────────────────────────────────────────────────────────────────────┤
│  // Copy inputs to thread-local storage (faster access)              │
│  int64_t regs[32];                                                   │
│  for (int i = 0; i < 32; i++) regs[i] = registers_in[i];           │
│  uint64_t pc = pc_in[0];                                            │
│                                                                      │
│  // Main execution loop                                              │
│  while (cycles < max_cycles) {                                       │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ FETCH: Read 4 bytes from memory (little-endian)         │    │
│      │   inst = memory[pc] | (memory[pc+1]<<8) | ...          │    │
│      └─────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ DECODE: Extract instruction fields via bit operations   │    │
│      │   opcode = (inst >> 24) & 0xFF                         │    │
│      │   rd = inst & 0x1F                                      │    │
│      │   rn = (inst >> 5) & 0x1F                              │    │
│      │   imm12 = (inst >> 10) & 0xFFF                         │    │
│      └─────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ CONTROL FLOW CHECK: Detect halt/syscall                 │    │
│      │   if (inst == 0 || is_hlt(inst)) → STOP_HALT           │    │
│      │   if (is_svc(inst)) → STOP_SYSCALL                     │    │
│      └─────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ EXECUTE: Giant switch statement on opcode               │    │
│      │   ADD_IMM:  regs[rd] = regs[rn] + imm12                │    │
│      │   SUB_IMM:  regs[rd] = regs[rn] - imm12                │    │
│      │   MOVZ:     regs[rd] = imm16 << (hw * 16)              │    │
│      │   LDR:      regs[rd] = load64(memory, addr)            │    │
│      │   B:        pc += sign_extend(imm26) * 4; continue     │    │
│      │   CBZ:      if (regs[rt] == 0) pc += offset            │    │
│      │   ...                                                   │    │
│      └─────────────────────────────────────────────────────────┘    │
│                              ↓                                       │
│      ┌─────────────────────────────────────────────────────────┐    │
│      │ UPDATE: Increment PC (if not branch) and cycle count    │    │
│      │   if (!branch_taken) pc += 4;                          │    │
│      │   regs[31] = 0;  // XZR always zero                    │    │
│      │   cycles++;                                             │    │
│      └─────────────────────────────────────────────────────────┘    │
│  }                                                                   │
│                                                                      │
│  // Write outputs                                                    │
│  for (int i = 0; i < 32; i++) registers_out[i] = regs[i];          │
│  pc_out[0] = pc;                                                    │
│  cycles_out[0] = cycles;                                            │
│  stop_reason_out[0] = reason;                                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         PYTHON (CPU)                                 │
├─────────────────────────────────────────────────────────────────────┤
│  3. Retrieve outputs (SINGLE sync point!):                          │
│     mx.eval(outputs)  # Force GPU to complete                       │
│     cycles = outputs[3][0].item()                                   │
│     stop_reason = outputs[4][0].item()                              │
│                                                                      │
│  4. Handle result:                                                   │
│     if stop_reason == SYSCALL:                                      │
│         handle_syscall()                                            │
│     elif stop_reason == HALT:                                       │
│         done = True                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### MLX metal_kernel API

MLX provides a Python API to define custom Metal shaders:

```python
kernel = mx.fast.metal_kernel(
    name="arm64_cpu_execute",

    # Input buffer names (read-only in kernel)
    input_names=["memory", "registers_in", "pc_in", "flags_in", "max_cycles_in"],

    # Output buffer names (writable in kernel)
    output_names=["registers_out", "pc_out", "flags_out", "cycles_out", "stop_reason_out"],

    # Metal shader source code (the actual GPU code)
    source=KERNEL_SOURCE,

    # Header with helper functions
    header=KERNEL_HEADER,
)

# Launch kernel
outputs = kernel(
    inputs=[memory, registers, pc, flags, max_cycles],
    output_shapes=[(32,), (1,), (4,), (1,), (1,)],
    output_dtypes=[mx.int64, mx.uint64, mx.float32, mx.uint32, mx.uint8],
    grid=(1, 1, 1),        # Single thread (for now)
    threadgroup=(1, 1, 1),
)
```

### Why Metal Shading Language?

Metal is Apple's GPU programming API. Key features we use:

1. **C++14 syntax**: Familiar programming model
2. **While loops**: Unlike many GPU languages, Metal supports true loops
3. **Conditionals**: Full if/else/switch support
4. **Thread-local arrays**: Fast register file storage
5. **Device memory access**: Read/write GPU memory directly

```metal
// Example: Thread-local register file
kernel void cpu_execute(
    device const uint8_t* memory [[buffer(0)]],  // Input (read-only)
    device int64_t* registers_out [[buffer(5)]],  // Output (writable)
    uint tid [[thread_position_in_grid]]
) {
    // Thread-local storage (very fast)
    int64_t regs[32];

    // Copy from device memory to thread-local
    for (int i = 0; i < 32; i++)
        regs[i] = registers_in[i];

    // ... execution loop uses regs[] ...

    // Copy back to device memory
    for (int i = 0; i < 32; i++)
        registers_out[i] = regs[i];
}
```

## Performance Analysis

### Timing Breakdown

| Operation | PyTorch | MLX Kernel |
|-----------|---------|------------|
| Fetch | ~100µs (sync) | ~0.01µs (local) |
| Decode | ~0.1µs | ~0.01µs |
| Execute | ~1µs | ~0.1µs |
| PC Update | ~100µs (sync) | ~0.01µs |
| **Per Instruction** | **~200µs** | **~0.1µs** |
| **IPS** | **~5,000** | **~10,000,000** |

Note: Actual measured IPS is ~1-2M due to kernel launch overhead and memory bandwidth.

### Why Not Even Faster?

Current limitations:

1. **Single-threaded**: Only one GPU thread executes instructions
2. **Sequential execution**: No instruction-level parallelism
3. **Memory bandwidth**: 4MB memory access patterns not optimized
4. **Kernel launch overhead**: ~10-20µs per kernel call

Future optimizations (Phase 2+):
- Parallel execution lanes
- Memory prefetching
- Loop vectorization
- Speculative execution

## Phase 1 Limitations

### Read-Only Memory

MLX passes input buffers as `const device` (read-only). This means:
- LDR (load) works: reads from memory
- STR (store) disabled: can't write to memory

**Workaround for Phase 2**: Use atomic outputs or a write buffer mechanism.

### Single-Threaded

Currently uses `grid=(1,1,1)` - single GPU thread. This is intentional for:
- Correctness (no race conditions)
- Simplicity (easier to debug)
- Sequential semantics (matches real CPU)

**Future**: Parallel lanes for independent instruction streams.

## Code Organization

```
mlx_kernel/
├── __init__.py           # Module exports
│
├── cpu_kernel_source.py  # Metal shader source code
│   ├── KERNEL_HEADER     # Helper functions (fetch, load64, sign_extend)
│   └── KERNEL_SOURCE     # Main execution loop
│
├── cpu_kernel.py         # Python wrapper
│   ├── MLXKernelCPU      # Main class
│   │   ├── execute()     # Launch kernel
│   │   ├── load_program() # Load code into memory
│   │   └── get_register() # Read register values
│   └── StopReason        # Enum for halt/syscall/max_cycles
│
└── test_kernel.py        # Comprehensive tests
    ├── test_movz_basic()
    ├── test_branch_unconditional()
    ├── test_loop()
    └── ... (23 tests total)
```

## Summary

The MLX Metal Kernel achieves 25-100x speedup by:

1. **Eliminating sync overhead**: Entire loop runs on GPU
2. **Thread-local storage**: Fast register access
3. **Native Metal execution**: Optimized for Apple Silicon
4. **Single kernel call**: One sync per execution batch

This approach trades flexibility (must pre-compile shader) for performance (millions of IPS).

---

*Document Version: 1.0*
*Last Updated: 2025-01-20*
