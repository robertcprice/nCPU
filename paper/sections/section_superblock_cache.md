## 16. Region-Guard Minimization for Superblock Caches

nCPU's neural execution engine runs a zero-sync GPU-only hot path that detects repeatable instruction blocks and memoizes their simulated execution traces. The cache must correctly invalidate when a block's inputs change, without over-guarding bytes that the block does not read. This section describes a three-level cache (trace / template / shape-patch) with a load/store asymmetric guard split, and reports measured speedups on a bytecopy workload.

The technique generalizes beyond nCPU to any ISA-level JIT memoization strategy: the pre-state a block depends on is the dual of its load set, not its memory-op set.

### 16.1 Problem Statement

A superblock is a bounded linearized execution trace through one or more basic blocks, terminated by a halt, synthetic stop, or back-edge. For a block at program counter $\text{pc}$, a trace cache maps a lookup key (typically $\text{pc}$ plus code-window contents) to a cached concrete execution plan: the register-file deltas, the flag updates, and the memory writes that the block produced on its original simulation.

For the cache to be safe on subsequent entries, the engine must verify that the block's inputs have not changed since the cache was populated. Inputs are:

1. The *code* executed by the block --- the instruction bytes in the linearized window.
2. The *registers* and *condition flags* the block reads along its path.
3. The *memory* the block reads along its path.

A naive implementation guards all three conservatively: snapshot the code window, snapshot the register file, snapshot every byte touched by any load or store in the block. On a cache lookup, compare all three snapshots byte-for-byte with the current machine state; if any differs, invalidate and re-simulate.

This conservatism produces two classes of false misses:

- **Write-only memory regions.** Stores write memory the block does not depend on. If a later byte mutation happens to fall in a pure-store region, the guard triggers a miss even though the block's inputs are unchanged.
- **Uninvolved register values.** Loop preambles and post-processing code leave registers live that the loop body never reads. A guard that captures the entire register file invalidates whenever any of these benign registers change.

Both categories expand the false-miss rate, which directly hurts throughput: every false miss re-simulates the block (which is what the cache exists to avoid) and re-copies its pre-state to the runtime CPU.

### 16.2 Three-Level Cache

The cache is organized into three levels, each with progressively looser invalidation semantics:

| Level | Key | Invariant | Use case |
|-------|-----|-----------|----------|
| Trace | $\text{pc}$ + code bytes + memory snapshot + full register vector | Exact re-execution of same inputs | Tight-loop hot spots |
| Template | code shape + minimal register guard + flag guard | Same program, different non-live register values | Shared loop bodies across iterations |
| Shape-patch | code shape with literal slots unbound | Same control flow, different literal immediates | Specializing a shape across call sites |

The trace level is the most restrictive: any change to the register file, the memory snapshot, or the code window invalidates. The template level relaxes the register guard to only the registers the block's simulation provably reads. The shape-patch level relaxes further, accepting different literal immediates at marked instruction slots and patching the cached plan before re-executing.

Each level has a distinct invalidation cost: trace-level misses require re-simulation; template-level misses require recomputing the minimal guard but can reuse the cached plan's structure; shape-patch-level misses can apply a small literal patch to the cached plan without touching its control flow. The cost hierarchy makes the cache work best when hot blocks are stable but their surrounding state is noisy.

### 16.3 Minimal Register Guard via Backward Liveness

The template-level register guard captures only registers the block *uses*. We compute this via a backward liveness scan over the simulated path: starting from the exit point, we walk the instructions in reverse, maintaining a live set. An instruction defines a register when it writes one, and uses registers when it reads them.

The subtlety is memory operations. A load *defines* its destination register (the loaded value becomes live in that register for later use), but it also *uses* its address register (the base register that computes the address). A store *uses* both its address register and its data register. The backward scan handles loads by removing the destination register from the live set (it's defined here, so uses after this point are satisfied) and adding the address register. Stores keep the address register but only keep the data register live if it was already live --- a store in a cold part of the block whose data register is never read downstream does not extend the guard.

This last rule is the asymmetric-store observation: store data-reg liveness is conditional on downstream use, store address-reg liveness is unconditional. The naive implementation keeps both unconditionally, which inflates the guard to cover registers the cache need not discriminate on.

### 16.4 Read/Write Window Split

An analogous asymmetry holds for memory. The pre-state memory guard must capture all bytes the block *reads*. Bytes the block writes do not need pre-state protection: the block overwrites them. The previous implementation captured the union of read and written byte ranges, which is conservative but wasteful.

The fix is a window-aware split in the simulator: track read-windows (from loads) and write-windows (from stores) separately. The memory guard is built from read-windows only. Further, when a load reads bytes that an earlier store in the same block has already written, those bytes are served from a shadow buffer during simulation and need not appear in the pre-state guard at all --- the block is self-contained for that byte range.

The code change is localized to the simulator's memory-op handler. In pseudocode:

```
on load at address A, size S:
    if all bytes in [A, A+S) are in shadow_buffer:
        # fully satisfied by prior store in this block
        pass
    else:
        pre_windows.append((A, A + S))
    read from shadow_buffer where present, else from memory

on store at address A, size S, data D:
    post_windows.append((A, A + S))
    shadow_buffer[A..A+S] := D
```

The corresponding memory guard is `merge(pre_windows)` rather than `merge(pre_windows ∪ post_windows)`. The pre-sync window --- bytes copied from the host memory into the execution CPU before the block runs --- is the same set, so the fix simultaneously reduces guard false-misses *and* reduces host-to-runtime copy traffic.

### 16.5 Measured Results

On the ARM64 bytecopy workload (2000 iterations copying a 2000-byte buffer, a representative store-heavy loop), the read/write split produces a direct reduction in pre-sync bytes:

| Workload | `pre_sync_bytes` before | after | reduction |
|---|---|---|---|
| `bytecopy` (single loop) | 4000 | 2000 | 50% |
| `adjacent-bytecopy` (two chained loops) | 8000 | 4000 | 50% |

Wall-clock IPS is within run-to-run noise at $\approx 1\,\text{M}$ IPS on both configurations for these workloads, because the Rust runtime CPU dominates execution time at this workload scale. The pre-sync win matters more as memory range or hotloop re-entry rate grows --- the savings scale linearly with loop buffer size and with block re-entry count.

The three-level cache contributes a different kind of win: on hot blocks that survive across many entries but see noise in the surrounding register file, the template-level guard catches what would otherwise be a full trace-level miss. The cost of a template hit (guard check + plan reuse) is a small fraction of the cost of re-simulation.

### 16.6 Verification

The cache and guard implementation is pinned by five regressions in the nCPU test suite:

1. `test_simulator_pre_windows_excludes_store_only_ranges` --- pure-store block produces an empty `pre_windows`.
2. `test_simulator_pre_windows_shadows_read_after_write` --- a store-then-load of the same byte produces an empty `pre_windows` (shadow handles the read).
3. `test_simulator_pre_windows_keeps_pure_load_ranges` --- pure-load block keeps its read range in `pre_windows`.
4. `test_superblock_trace_cache_ignores_store_only_byte_changes` --- benign mutation at a store destination still hits the cache.
5. `test_superblock_trace_cache_invalidates_on_load_byte_change` --- mutation at a load source invalidates the cache.

All five pass on the current implementation. A sixth pre-existing assertion on `previous_pre_sync_bytes` was updated from 8 to 4 bytes; the old value pinned the over-conservative behavior, not a correctness constraint.

### 16.7 Relation to Prior Work

Traditional trace caches (e.g., Pentium 4, Intel Ice Lake µop cache) guard by program counter and code bytes, and invalidate on any write to the covered instruction range. They do not model data dependencies because their job is to cache decoded µops, not to skip execution.

Speculative hot-loop JITs (e.g., LuaJIT 2, PyPy) use trace trees with type-specialized guards and side exits. Their guard metric is *type stability*, not *memory freshness*: a loop's iteration types are guarded, but its pre-state memory is not, because tracing JITs re-execute every iteration rather than memoizing outcomes.

The closest analog to our construction is a software speculation engine like a DBT (dynamic binary translator) with region caching. Those systems face the same over-guarding question but typically resolve it by using page-granularity protection rather than byte-granularity windows. Our approach is finer --- window-level rather than page-level --- and exploits the asymmetry between load-read and store-write dependencies to produce tighter guards without additional hardware support.

The three-level layering (trace / template / shape-patch) is, as far as we know, new in the JIT region-cache literature. Each level trades off guard complexity against hit rate, and the combination dominates either level alone on our benchmarks.

### 16.8 Future Directions

1. **Dead-register elision from guards.** The backward liveness scan currently keeps all base registers used by memory ops. Some of these may be recomputed by arithmetic within the block (e.g., loop counter updates) and do not need to be guarded at their pre-state value, only at their entry-to-loop value. A fixed-point liveness analysis across multiple block entries could further trim guards.

2. **Partial memory guard compaction.** *(Considered and deferred.)* The read-window set currently stores exact byte ranges. A stride descriptor (base, stride, count) would reduce storage for strided loops, but byte-level comparison remains O(N) unless we also reorder the compare to match the stride. On bytecopy workloads at 2 KB pre-sync, the memcmp cost is already ~200 ns, dominated by the surrounding pipeline. The optimization is speculative until a workload with large (>100 KB) strided windows AND frequent trace-level hits emerges. The adaptive promotion mechanism above already captures the more impactful case of skipping unhelpful trace-level comparisons entirely.

3. **Adaptive level promotion.** *(Implemented and measured.)* Program keys that consecutively miss at trace level are promoted to skip trace-level lookup, avoiding the memory-snapshot comparison cost on keys that never benefit from exact-match caching. On a synthetic workload where the code window is stable but a non-guarded register varies per entry, threshold=3 produces a **17% per-lookup speedup** (24.71 μs → 20.42 μs on 2000 iterations, CPU only) with identical template hit counts. Threshold is configurable via `NCPU_GPU_ONLY_SUPERBLOCK_TRACE_PROMOTION` (default 3, zero disables the optimization). Two regressions pin the on/off behavior and a microbenchmark at `benchmarks/benchmark_superblock_promotion.py` reproduces the measured speedup. Benchmark output is at `artifacts/superblock_promotion_benchmark.txt`.

4. **Cross-block guard sharing.** Adjacent blocks in the same loop often guard on the same memory windows. Sharing a guard across blocks that pair in a known way would amortize the snapshot cost.
