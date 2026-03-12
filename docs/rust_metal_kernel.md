# Rust Metal Kernel Architecture

The Rust Metal kernel (`FullARM64CPU`) is nCPU's primary execution backend. It runs a complete ARM64 emulator on Apple Silicon's Metal GPU with zero-copy shared memory, achieving ~500x faster compilation and ~435x faster SVC handling compared to the legacy Python MLX kernel.

## Key Design: StorageModeShared (Zero-Copy)

The Python MLX kernel uses a double-buffer architecture: `memory_in` (read-only) and `memory_out` (writable). After each kernel dispatch, the 16MB output buffer is copied back to input. With SVC traps occurring every few hundred instructions, this means **thousands of 16MB copies per program execution**.

The Rust kernel uses `MTLResourceOptions::StorageModeShared` — a single memory buffer shared between CPU and GPU with no copies. When the GPU halts for a syscall, Python reads/writes the same memory directly. Zero copies, zero overhead.

```
Python MLX (legacy):                    Rust Metal (current):
  GPU writes → memory_out                GPU writes → memory (shared)
  Copy 16MB → memory_in                  Python reads → memory (same buffer)
  Python reads → memory_in               Python writes → memory (same buffer)
  Repeat per SVC                          No copies ever
```

## Metal Buffer Layout

11 Metal buffers allocated with StorageModeShared:

| Buffer | Type | Size | Purpose |
|--------|------|------|---------|
| 0 | `uint8_t[]` | 16 MB | Main memory (program, data, stack, heap) |
| 1 | `int64_t[32]` | 256 B | General-purpose registers X0-X30, SP |
| 2 | `uint64_t` | 8 B | Program counter |
| 3 | `float[4]` | 16 B | NZCV flags |
| 4 | `uint32_t` | 4 B | Max cycles per batch |
| 5 | `uint32_t` | 4 B | Memory size |
| 6 | `int64_t[32]` | 256 B | SIMD vreg low (V0-V31 lower 64 bits) |
| 7 | `int64_t[32]` | 256 B | SIMD vreg high (V0-V31 upper 64 bits) |
| 8 | `uint32_t` | 4 B | Signal flag (RUNNING/HALT/SYSCALL/CHECKPOINT) |
| 9 | `uint64_t` | 8 B | Total cycles counter |
| 10 | `uint32_t` | 4 B | Batch count |

## GPU-Side SVC Buffer

Hot syscalls are handled entirely on GPU without trapping to Python:

| Syscall | GPU Handling |
|---------|-------------|
| `SYS_WRITE(stdout/stderr)` | Buffered at `0x3F0000` (~64KB capacity) |
| `SYS_BRK` | Heap break tracked at `SVC_BUF_BASE + 8` |
| `SYS_CLOSE(fd <= 2)` | No-op, return success |
| `SYS_EXIT` / `SYS_EXIT_GROUP` | Immediate halt |

All other syscalls trap to Python via the signal flag mechanism.

The SVC write buffer layout at `0x3F0000`:
```
[0..3]   uint32_t write_pos    (offset into data area)
[4..7]   uint32_t entry_count  (number of buffered entries)
[8..15]  uint64_t brk_addr     (heap break for SYS_BRK)
[16..]   entry data: [1B fd][2B len LE][len bytes data]
```

After each GPU dispatch, Python drains the buffer and prints buffered output.

## ARM64 ISA Coverage

139 instructions fully implemented in the Metal shader:

- **Data movement**: MOVZ, MOVN, MOVK (16/32/64-bit), LDR/STR (byte/half/word/double, pre/post-index, register offset, signed/unsigned), LDP/STP, LDXR/STXR (exclusive)
- **Arithmetic**: ADD, SUB, ADDS, SUBS, ADC, SBC, MUL, SMULL, UMULL, UDIV, SDIV, MADD, MSUB, NEG
- **Logic**: AND, ORR, EOR, BIC, ORN, EON, ANDS, BICS, MVN, TST
- **Shifts**: LSL, LSR, ASR, ROR, UBFM, SBFM, EXTR
- **Branches**: B, BL, BR, BLR, RET, B.cond (14 conditions), CBZ, CBNZ, TBZ, TBNZ
- **Comparison**: CMP, CMN, CCMP, CSEL, CSINC, CSINV, CSNEG, CSET
- **System**: NOP, HLT, BRK, SVC, DMB, DSB, ISB, MRS, MSR, CLREX
- **SIMD**: LDR/STR Q-register (128-bit load/store for musl va_list)

## Build Instructions

```bash
cd kernels/rust_metal

# Build the Rust + Metal library
maturin develop --release

# Build a wheel for the active interpreter and install it:
maturin build --release -o dist
python3 -m pip install --break-system-packages --force-reinstall dist/ncpu_metal-*.whl
```

### Dependencies

- Rust (stable), `maturin` (for PyO3 builds)
- `objc2`, `objc2-metal`, `objc2-foundation` (Metal API bindings)
- `pyo3` (Python ↔ Rust bridge)
- Apple Silicon Mac with Metal support

## Standalone Rust Launcher

The Rust crate also includes a standalone Rust launcher path with two entry
modes:

```bash
cd kernels/rust_metal
cargo run --bin ncpu_run -- --elf ../../demos/busybox.elf --rootfs -- echo hello
cargo run --bin ncpu_run -- ../../path/to/image.bin
```

Current launcher coverage:

- ELF loading and Linux stack setup via `src/elf_loader.rs`
- Boot-image validation, region/task loading, and packed rootfs mounting via `src/boot_image.rs`, `src/rootfs.rs`, and `bin/ncpu_run.rs`
- Virtual filesystem and fd management via `src/vfs.rs`
- Execution loop and Linux-persona syscall dispatch via `src/launcher.rs`
- Live `ProcessManager` integration for scheduling, fork/wait, pipes, fd snapshots, per-process memory images, and Linux `clone(220)` interception

Current limitations:

- The older fallback syscall table in `src/launcher.rs` still contains a stale `clone -> -ENOSYS` branch even though the live runtime path intercepts `clone(220)` earlier
- The native hypercall ABI in `src/native_abi.rs` exists as a module, but is not yet the default live execution path
- Boot-image mode currently boots from the packed image metadata and rootfs; extra CLI args for image mode are not wired yet
- In this workspace, `cargo check --bin ncpu_run`, `cargo check --lib`, and `cargo test process:: --lib` pass, but direct `cargo run --bin ncpu_run ...` still fails at final link time unless PyO3 can resolve Python symbols

Python is still used for higher-level orchestration and many debug tools. Framebuffer callbacks and most tracing/debug scripts remain outside this Rust path, but the runtime path above does not depend on `ncpu/os/gpu/elf_loader.py` or `ncpu/os/gpu/runner.py`.

## Python API

```python
from kernels.mlx.gpu_cpu import GPUKernelCPU, StopReasonV2

cpu = GPUKernelCPU(memory_size=16*1024*1024, quiet=True)
cpu.load_program(binary_bytes, address=0x10000)
cpu.set_pc(0x10000)
cpu.init_svc_buffer()

result = cpu.execute(max_cycles=100_000)
# result.cycles, result.elapsed_seconds, result.stop_reason, result.ips

# Register access
cpu.get_register(0)      # X0
cpu.set_register(0, 42)

# Memory access
cpu.read_memory(addr, size)  # → bytes
cpu.write_memory(addr, data)
cpu.read_memory_64(addr)     # → int (little-endian)

# Bulk operations
regs = cpu.get_registers_numpy()   # → np.ndarray(int64, 32)
cpu.set_registers_numpy(regs)
n, z, c, v = cpu.get_flags()
cpu.set_flags(n=True, z=False, c=False, v=False)

# SVC buffer
entries = cpu.drain_svc_buffer()  # → [(fd, data), ...]
```

## Backend Selection

Set `NCPU_GPU_BACKEND` environment variable:

| Value | Behavior |
|-------|----------|
| `auto` (default) | Try Rust Metal, fall back to Python MLX |
| `rust` | Force Rust Metal (fail if unavailable) |
| `mlx` | Force Python MLX (useful for debugging) |

```bash
NCPU_GPU_BACKEND=mlx python3 -m pytest tests/  # Test with legacy backend
```

## Performance

| Metric | Rust Metal | Python MLX | Speedup |
|--------|-----------|------------|---------|
| Compilation | ~44ms | ~22s | **~500x** |
| IPS (sustained) | ~1,916,190 | ~4,400 | **~435x** |
| SVC overhead | 0 bytes copy | 16MB copy | **Zero-copy** |
| cc.c self-compile | ~2s | ~15min | **~450x** |

## GPU-Native Debugging Toolkit

The Metal shader includes a comprehensive debugging infrastructure checked every GPU cycle at zero overhead:

Shell-facing command docs live in [`docs/gpu_debugging_toolkit.md`](/Users/bobbyprice/projects/nCPU/docs/gpu_debugging_toolkit.md).

### Instruction Trace Buffer (0x3B0000)
- 4096-entry circular buffer, 56 bytes/entry
- Captures: PC, instruction word, x0-x3, NZCV flags, SP
- Enabled/disabled via sentinel value (0xFFFFFFFF = disabled)
- Zero overhead when disabled (single conditional check)

### Breakpoints (Debug control block at 0x3A0000)
- Up to 4 PC breakpoints, checked every cycle before instruction execution
- **Conditional breakpoints**: Fire only when PC matches AND register equals a value
- Zero-cost: check is inside the Metal shader, no GPU-CPU round-trip
- Signal: STOP_BREAKPOINT=4 → Rust Signal::Breakpoint → Python StopReasonV2.BREAKPOINT

### Memory Watchpoints (offset 92 in debug control block)
- Up to 4 memory watchpoints using shadow-comparison
- After each instruction, compares 8-byte value at watched address against stored shadow
- If changed: records (wp_index, addr, old_val, new_val), fires STOP_WATCHPOINT=5
- Shadow auto-updates on trigger (prevents re-trigger on resume)

### Debug Control Block Layout (200 bytes at 0x3A0000)
```
Offset 0:   Breakpoints  (36B)  — num_bp(4B) + 4 × bp_pc(8B)
Offset 36:  Conditions   (52B)  — num_cond(4B) + 4 × {reg_idx(4B) + reg_value(8B)}
Offset 92:  Watchpoints  (108B) — num_wp(4B) + 4 × {addr(8B) + shadow(8B)} + hit_info(28B)
```

### Python API
```python
cpu.enable_trace() / cpu.disable_trace()
cpu.read_trace()   → [(pc, inst, x0, x1, x2, x3, flags, sp), ...]
cpu.set_breakpoint(index, pc)
cpu.set_conditional_breakpoint(index, pc, reg, value)
cpu.set_watchpoint(index, addr)
cpu.read_watchpoint_info()  → (wp_idx, addr, old_val, new_val) or None
cpu.clear_breakpoints() / cpu.clear_watchpoints() / cpu.clear_all_debug()
```

## File Layout

```
kernels/rust_metal/
  Cargo.toml              # Rust package config
  bin/ncpu_run.rs         # Standalone Rust launcher (ELF or boot image)
  src/
    lib.rs                # PyO3 module registration
    full_arm64.rs         # 3,150 lines — complete Metal shader + Rust host
    boot_image.rs         # Boot image format, builder, validation, rootfs packing
    elf_loader.rs         # Rust ELF loader + Linux stack/auxv setup
    vfs.rs                # In-memory VFS and fd table
    rootfs.rs             # Rootfs loader for packed boot images
    process.rs            # Multi-process state manager used by the live standalone launcher path
    native_abi.rs         # Native hypercall ABI module (not yet default live path)
    launcher.rs           # Rust execution loop + live Linux-persona runtime, plus one stale fallback syscall table to clean up
    continuous.rs         # Original ~60-instruction kernel
    ...                   # Other experimental modules
kernels/mlx/
  gpu_cpu.py              # Unified factory (auto/rust/mlx selection)
  rust_runner.py          # Python wrapper for FullARM64CPU
  cpu_kernel_v2.py        # Legacy Python MLX kernel
```
