# GPU Debugging Toolkit

The Alpine GPU shell exposes a finished 26-command GPU-native debugging and
analysis toolkit on top of deterministic Metal execution, the 4096-entry trace
buffer, GPU-side breakpoints/watchpoints, preserved post-execution state, and
shared ELF symbol loading.

## Status

- 26 shell-facing commands
- verified by dedicated GPU trace/debugging tests plus BusyBox shell integration
- deterministic post-mortem analysis instead of ptrace-style live instrumentation
- symbol-aware profiling and stack reconstruction when unstripped ELF symbols are available

## Command Surface

| Command | Purpose |
|---------|---------|
| `gpu-xray CMD` | Post-execution register and memory forensics |
| `gpu-replay CMD` | Deterministic replay proof |
| `gpu-diff A -- B` | Cross-execution state diff |
| `gpu-freeze CMD` | Snapshot full machine state |
| `gpu-thaw [id]` | Inspect saved snapshots |
| `gpu-timing-proof CMD` | Prove stable cycle counts across runs |
| `gpu-strace CMD` | Zero-overhead syscall tracing |
| `gpu-trace CMD` | Instruction-level execution trace |
| `gpu-break ADDR CMD` | PC breakpoint debugging |
| `gpu-watch ADDR CMD` | Memory watchpoint debugging |
| `gpu-history CMD` | Time-travel execution history |
| `gpu-coverage CMD` | Instruction-type coverage analysis |
| `gpu-taint ADDR CMD` | Data-flow tracking from a watched address |
| `gpu-bisect ADDR EXPECTED CMD` | Find the first bad write |
| `gpu-step CMD` | Single-step the first instructions |
| `gpu-profile CMD` | Instruction mix, hotspots, and call targets |
| `gpu-stack CMD` | Reconstruct BL/RET call stacks |
| `gpu-heat CMD` | Instruction frequency heatmap |
| `gpu-diff-input A -- B` | Diff traces for different inputs |
| `gpu-asm CMD` | Disassemble trace entries to ARM64 |
| `gpu-sanitize CMD` | Zero-overhead memory safety analysis |
| `gpu-fuzz CMD [--rounds N]` | Crash-oriented fuzzing with trace capture |
| `gpu-reverse REG CMD` | Reverse data-flow from a final register value |
| `gpu-const-time A -- B` | Constant-time verification |
| `gpu-map CMD` | Visualize memory layout and access regions |
| `gpu-entropy FILE` | Shannon entropy analysis |

`gpu-help` in the Alpine shell prints the same command surface grouped by use
case.

## Trace Model

- Trace entries capture state before instruction execution.
- Each entry contains `pc`, `inst`, `x0-x3`, `NZCV`, and `sp`.
- `gpu-reverse` can fully track `x0-x3`; for higher registers it falls back to
  write-site discovery because the trace does not capture all 31 registers.
- Breakpoints trace the matching instruction but stop before executing it.

## Symbol Annotations

- `gpu-profile` and `gpu-stack` can annotate PCs with function names when ELF
  function symbols are available.
- Symbol loading now comes from the shared ELF layer, not a one-off helper in a
  single analysis script.
- The shipped [`demos/busybox.elf`](/Users/bobbyprice/projects/nCPU/demos/busybox.elf)
  is stripped, so raw PCs are expected unless you provide an unstripped sibling
  such as `busybox.debug`, `busybox.debug.elf`, `busybox.unstripped`, or
  `busybox.unstripped.elf`.

## Practical Notes

- The shell-facing implementations live in
  [`demos/alpine_gpu.py`](/Users/bobbyprice/projects/nCPU/demos/alpine_gpu.py).
- The standalone write-up lives in
  [`paper/gpu_debugging_toolkit_paper.md`](/Users/bobbyprice/projects/nCPU/paper/gpu_debugging_toolkit_paper.md).
- The underlying GPU trace/debug primitives are documented in
  [`docs/rust_metal_kernel.md`](/Users/bobbyprice/projects/nCPU/docs/rust_metal_kernel.md).
- The low-level ELF loading and symbol parsing live in
  [`ncpu/os/gpu/elf_loader.py`](/Users/bobbyprice/projects/nCPU/ncpu/os/gpu/elf_loader.py).
- The Rust backend is packaged as `ncpu_metal`, matching the repo and avoiding
  the older legacy package name.
