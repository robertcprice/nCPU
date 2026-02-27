# Tensor-Native Execution Architecture

## The Problem

Current execution is bottlenecked by GPU↔CPU synchronization:

```python
# Current: 500µs per instruction (2K IPS)
for each instruction:
    pc = pc.item()           # ← 103µs sync
    inst = fetch(pc).item()  # ← 390µs sync (5x .item())
    execute(inst)            # ← 17µs
```

**96% of time is sync overhead. The GPU is idle waiting for Python.**

## The Solution: Zero-Sync Tensor Execution

Keep EVERYTHING as tensors. Never call `.item()` until syscall.

```python
# Target: 1µs per instruction (1M+ IPS)
inst_batch = memory.gather(pc_batch)  # Tensor op, no sync
decoded = decode_batch(inst_batch)     # Tensor op, no sync
execute_batch(decoded, regs)           # Tensor op, no sync
pc_batch = update_pc_batch(decoded)    # Tensor op, no sync
# Only sync when syscall detected
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TENSOR STATE (All on GPU)                     │
├─────────────────────────────────────────────────────────────────┤
│  PC Tensor [B]        - B parallel program counters             │
│  Registers [B, 32]    - B copies of 32 registers                │
│  Flags [B, 4]         - B copies of NZCV flags                  │
│  Memory [4M]          - Shared memory (single copy)             │
│  Active Mask [B]      - Which lanes are still executing         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 BATCH FETCH (Tensor Gather)                      │
├─────────────────────────────────────────────────────────────────┤
│  Input:  PC tensor [B]                                          │
│  Output: Instruction tensor [B] (32-bit per lane)               │
│                                                                 │
│  byte_idx = stack([pc, pc+1, pc+2, pc+3])  # [B, 4]            │
│  bytes = memory.gather(byte_idx.flatten()) # [B*4]              │
│  insts = bytes.view(B, 4) @ [1, 256, 65536, 16M]  # [B]        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│               VECTORIZED DECODE (Bit Operations)                 │
├─────────────────────────────────────────────────────────────────┤
│  op_byte = (insts >> 24) & 0xFF           # [B]                 │
│  rd = insts & 0x1F                        # [B]                 │
│  rn = (insts >> 5) & 0x1F                 # [B]                 │
│  rm = (insts >> 16) & 0x1F                # [B]                 │
│  imm12 = (insts >> 10) & 0xFFF            # [B]                 │
│  imm16 = (insts >> 5) & 0xFFFF            # [B]                 │
│                                                                 │
│  is_add = (op_byte == 0x91) | (op_byte == 0x11)                │
│  is_sub = (op_byte == 0xD1) | (op_byte == 0x51)                │
│  is_movz = (op_byte == 0xD2) | (op_byte == 0x52)               │
│  ... (all instruction types as boolean masks)                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PARALLEL EXECUTE (Masked Operations)                │
├─────────────────────────────────────────────────────────────────┤
│  # Read operands (gather from register file)                    │
│  rn_vals = regs.gather(1, rn.unsqueeze(1)).squeeze(1)          │
│  rm_vals = regs.gather(1, rm.unsqueeze(1)).squeeze(1)          │
│                                                                 │
│  # Compute ALL possible results in parallel                     │
│  add_result = rn_vals + imm12                                   │
│  sub_result = rn_vals - imm12                                   │
│  movz_result = imm16 << (hw * 16)                              │
│  and_result = rn_vals & rm_vals                                 │
│  ...                                                            │
│                                                                 │
│  # Select correct result using masks                            │
│  result = torch.where(is_add, add_result,                       │
│           torch.where(is_sub, sub_result,                       │
│           torch.where(is_movz, movz_result,                     │
│           ...)))                                                │
│                                                                 │
│  # Scatter write to register file                               │
│  regs.scatter_(1, rd.unsqueeze(1), result.unsqueeze(1))        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              TENSOR CONTROL FLOW (Branch Handling)               │
├─────────────────────────────────────────────────────────────────┤
│  # Compute branch targets (all on GPU)                          │
│  branch_offset = sign_extend(imm26, 26) * 4                     │
│  branch_target = pc + branch_offset                             │
│                                                                 │
│  # Evaluate branch conditions                                   │
│  cond_z = (flags[:, 2] == 1)  # Zero flag                      │
│  cond_n = (flags[:, 0] == 1)  # Negative flag                  │
│  take_branch = is_cbz & cond_z | is_cbnz & ~cond_z | ...       │
│                                                                 │
│  # Update PC (tensor operation, no sync)                        │
│  new_pc = torch.where(take_branch, branch_target, pc + 4)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   SYSCALL DETECTION & SYNC                       │
├─────────────────────────────────────────────────────────────────┤
│  # Check for SVC instruction (only sync point!)                 │
│  is_svc = (insts & 0xFFE0001F) == 0xD4000001                   │
│  any_svc = is_svc.any()  # Single boolean, cheap sync          │
│                                                                 │
│  if any_svc.item():  # ← ONLY .item() call in entire loop!     │
│      # Sync state and handle syscall                            │
│      handle_syscalls(regs, memory, is_svc)                     │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Tensor-Native State

All CPU state lives on GPU as tensors:

```python
class TensorNativeCPU:
    def __init__(self, batch_size=64):
        self.B = batch_size

        # All state on GPU
        self.pc = torch.zeros(B, dtype=torch.int64, device='mps')
        self.regs = torch.zeros(B, 32, dtype=torch.int64, device='mps')
        self.flags = torch.zeros(B, 4, dtype=torch.float32, device='mps')
        self.memory = torch.zeros(4*1024*1024, dtype=torch.uint8, device='mps')
        self.active = torch.ones(B, dtype=torch.bool, device='mps')
```

### 2. Batch Execution Lanes

Run B independent execution lanes in parallel:
- Each lane has its own PC, registers, flags
- Shared memory (coherent view)
- Lanes deactivate on syscall/halt

```python
# 64 lanes executing in parallel
# If lane 5 hits syscall, only lane 5 syncs
# Other 63 lanes continue on GPU
```

### 3. Instruction Grouping

Group compatible instructions for parallel execution:

```python
# Instead of:
for inst in batch:
    if is_add(inst): do_add()
    elif is_sub(inst): do_sub()
    ...

# Do:
add_mask = is_add(batch)
sub_mask = is_sub(batch)
...
# Execute ALL adds in one tensor op
regs[add_mask, rd[add_mask]] = rn_vals[add_mask] + imm12[add_mask]
# Execute ALL subs in one tensor op
regs[sub_mask, rd[sub_mask]] = rn_vals[sub_mask] - imm12[sub_mask]
```

### 4. Speculative Execution

For short branches, execute both paths:

```python
# For CBZ with short offset:
# Execute both taken and not-taken paths
result_taken = execute_at(branch_target)
result_not_taken = execute_at(pc + 4)

# Select correct result based on condition
result = torch.where(condition, result_taken, result_not_taken)
```

### 5. Memory Coalescing

Batch memory operations:

```python
# Instead of:
for addr in addresses:
    val = memory[addr].item()

# Do:
vals = memory.gather(0, addresses)  # One GPU op for all
```

## Implementation Plan

### Phase 1: Core Tensor Engine

1. **TensorFetch**: Batch instruction fetch with gather
2. **TensorDecode**: Vectorized bit extraction
3. **TensorALU**: Parallel ALU with result selection
4. **TensorRegisters**: Scatter/gather register file

### Phase 2: Control Flow

1. **TensorBranch**: Branch target computation
2. **TensorCondition**: Flag evaluation
3. **TensorPC**: PC update with masking

### Phase 3: Memory System

1. **TensorLoad**: Batched loads with gather
2. **TensorStore**: Batched stores with scatter
3. **TensorMemcpy**: Accelerated block operations

### Phase 4: Syscall Handling

1. **SyscallDetect**: Minimal sync for detection
2. **LanePause**: Pause individual lanes
3. **StateSync**: Efficient state transfer

## Expected Performance

| Component | Current | Target | Improvement |
|-----------|---------|--------|-------------|
| Fetch | 390µs | 0.8µs | 488x |
| Decode | 0.1µs | 0.01µs | 10x |
| Execute | 17µs | 0.2µs | 85x |
| PC Update | 1.4µs | 0.01µs | 140x |
| **Total** | **507µs** | **~1µs** | **500x** |

**Target IPS: 1,000,000+ (vs current 2,000)**

## ACHIEVED RESULTS ✅

| Execution Mode | IPS | vs Single-Step |
|----------------|-----|----------------|
| Single-step GPU | 48 | 1x |
| **Batch-512** | **118,676** | **2,472x** |
| Pure tensor ops | 44,793,788 | 933,204x |

**We achieved 2,500x speedup!** See `docs/benchmarks/TENSOR_NATIVE_RESULTS.md` for details.

## Code Structure

```
neural_cpu_tensor_native.py
├── TensorState           # GPU state management
├── TensorFetch           # Batch instruction fetch
├── TensorDecode          # Vectorized decode
├── TensorALU             # Parallel ALU operations
├── TensorMemory          # Coalesced memory access
├── TensorControl         # Branch/PC handling
├── TensorSyscall         # Syscall detection & handling
└── TensorExecutor        # Main execution loop
```

## Validation Strategy

1. **Correctness**: Compare output with CPU fast path
2. **Performance**: Benchmark against current implementation
3. **Scaling**: Test with different batch sizes (16, 32, 64, 128)
4. **Edge Cases**: Branches, syscalls, memory operations

---

*This architecture eliminates the GPU↔CPU sync bottleneck by keeping ALL execution on GPU tensors until an external interaction (syscall) is required.*
