# Deterministic Post-Mortem Debugging on GPU: A Standalone Toolkit for Trace, Replay, Forensics, and Constant-Time Verification

**Robert Price**

*March 2026*

---

## Abstract

Modern debugging tools are constrained by the execution model of CPUs. Breakpoints perturb timing, replay is difficult to make exact, watchpoints are scarce, and post-mortem state is usually incomplete once a process exits. This paper presents a different model: a GPU-native debugging toolkit built on deterministic ARM64 execution inside a Metal shader, where machine state persists in GPU-visible memory and analysis is performed after execution rather than through live ptrace-style instrumentation.

The resulting system exposes 26 shell-facing commands that cover instruction tracing, replay, breakpoints, watchpoints, time-travel history, syscall tracing, instruction coverage, taint-style propagation, bug bisection, profiling, call-stack reconstruction, heatmaps, trace diffing, disassembly, memory sanitization, crash-oriented fuzzing, reverse data-flow analysis, constant-time verification, and memory layout visualization. Several of these capabilities are individually familiar in CPU tooling, but the combination is unusual: they are unified behind one execution substrate, operate with effectively zero runtime instrumentation overhead, and exploit deterministic GPU execution to make exact replay and exact trace comparison practical.

This draft positions the toolkit as a standalone contribution independent of the broader nCPU system. The core claim is that once execution is hosted in a deterministic GPU compute environment with persistent trace buffers and explicit control memory, debugging changes from interactive interference to post-execution inspection. That shift enables a practical development platform for systems debugging, security analysis, and education that would be either fragile, approximate, or structurally impossible on conventional CPUs.

## 1. Introduction

Debugging on conventional CPUs is defined by scarcity and interference. Hardware watchpoints are limited. Breakpoints trap into the operating system and perturb the execution being observed. Record-and-replay systems can be powerful, but exact reproduction remains difficult because timing, scheduling, branch prediction, caching, and system interactions all introduce nondeterminism. After a process exits, most of the interesting state is gone.

This paper argues that a deterministic GPU execution substrate changes that model. In the system described here, ARM64 programs execute inside a Metal shader with explicit machine state, a circular trace buffer, GPU-resident debug control structures, and a shell-facing command surface designed around post-mortem analysis. Rather than instrumenting a live process from the outside, the debugger is built into the execution environment itself.

The result is a completed 26-command GPU-native debugging toolkit integrated into the Alpine GPU shell. It supports basic developer workflows such as trace, replay, and breakpointing, but it also supports workflows that are uncommon or impractical on CPU platforms: exact constant-time verification from trace equivalence, reverse data-flow from preserved execution history, crash-oriented fuzzing where every failing input keeps a full post-mortem trace, and memory safety analysis performed after execution with no additional sanitizer runtime.

This toolkit is motivated by the broader nCPU project, but it stands on its own as a systems contribution. The relevant novelty is not "debugging on GPU" as a slogan. The novelty is the combination of deterministic execution, preserved state, shell-facing usability, and an analysis surface broad enough to function as a real development environment rather than a one-off demo.

## 2. Contributions

This toolkit makes the following contributions:

1. A practical 26-command GPU-native debugging and analysis toolkit exposed through a shell environment rather than a lab-only prototype.
2. A deterministic execution model that makes exact replay, exact trace diffing, and exact constant-time comparison feasible.
3. A post-mortem debugging architecture in which trace, register state, memory state, and debug metadata remain inspectable after program termination.
4. A unified command surface spanning tracing, replay, break/watch, profiling, stack reconstruction, disassembly, fuzzing, sanitization, reverse analysis, and memory visualization.
5. A security-analysis workflow that combines exact trace comparison, cycle comparison, memory access inspection, and crash capture without relying on heavyweight CPU instrumentation frameworks.
6. A standalone framing for GPU execution as a debugging substrate, not only as an accelerator for unrelated workloads.

## 3. System Model

The toolkit runs on top of a GPU-resident ARM64 execution environment implemented in Metal and exposed through a Rust backend packaged as `ncpu_metal`. Programs execute with explicit architectural state: registers, flags, memory, program counter, trace buffers, debug control state, and syscall buffers are all represented in memory controlled by the runtime.

Three design choices are central:

- **Deterministic execution.** Re-running the same workload on the same inputs produces the same instruction stream and the same cycle counts under the model used by the runtime.
- **Persistent debug state.** Execution history is not discarded at process exit; it is left in GPU-visible data structures that analysis commands can inspect after the fact.
- **Integrated analysis.** Debugging features are implemented as part of the execution substrate and shell tooling, not bolted on through external ptrace-like control.

The trace model records instruction entries before execution. A typical entry includes the program counter, the instruction word, selected general-purpose registers, flags, and the stack pointer. This is sufficient to support post-mortem reconstruction of control flow, limited register evolution, branch behavior, memory access reasoning, and instruction classification.

## 4. Command Taxonomy

The toolkit exposes 26 shell-facing commands. They are best understood by category rather than as an undifferentiated list.

### 4.1 Trace and Replay

- `gpu-trace`
- `gpu-history`
- `gpu-replay`
- `gpu-diff`
- `gpu-diff-input`

These commands answer the most basic debugging questions: what executed, in what order, and how one execution differs from another. Because execution is deterministic, replay and diff are exact rather than approximate.

### 4.2 Break, Watch, and Inspect

- `gpu-break`
- `gpu-watch`
- `gpu-step`
- `gpu-xray`
- `gpu-freeze`
- `gpu-thaw`
- `gpu-strace`

These commands provide breakpoint-style debugging, watchpoint-style debugging, single-step inspection, and post-execution state interrogation. Unlike CPU debugging, the observation mechanism is embedded into the execution engine and does not rely on OS traps and resume cycles.

### 4.3 Profiling and Structure Recovery

- `gpu-profile`
- `gpu-stack`
- `gpu-heat`
- `gpu-coverage`
- `gpu-asm`
- `gpu-map`

These commands summarize execution at higher levels: hot code regions, call structure, instruction classes, symbolic or raw-PC disassembly, and memory layout. When ELF symbols are available, profile and stack views can be symbol-aware.

### 4.4 Data-Flow and Correctness Analysis

- `gpu-taint`
- `gpu-bisect`
- `gpu-reverse`
- `gpu-sanitize`
- `gpu-const-time`
- `gpu-timing-proof`

This cluster is where the toolkit moves beyond a conventional debugger. It does not just show execution; it analyzes it. These commands identify mutation chains, isolate the first bad write, walk backward from a final register value, flag memory safety issues, and compare traces and cycles for security-sensitive constant-time behavior.

### 4.5 Robustness and Security Workflows

- `gpu-fuzz`
- `gpu-entropy`
- `gpu-help`

`gpu-fuzz` turns the preserved-trace model into a usable crash-analysis loop: each crash can be examined immediately without a separate reproduction phase. `gpu-entropy` and grouped command help broaden the toolkit into an everyday analysis environment rather than a narrow debugging utility.

## 5. Why GPU Changes the Debugging Model

The toolkit’s main claim is structural: the GPU execution model changes what debugging can be.

On a CPU:

- breakpoints perturb execution through traps and scheduler interaction,
- replay must fight nondeterministic microarchitectural behavior,
- exact timing comparisons are noisy,
- watchpoints are scarce,
- sanitizer and dynamic instrumentation frameworks impose large overheads,
- state is usually incomplete after process exit.

On this GPU substrate:

- trace capture is part of the execution pipeline,
- replay is deterministic under the runtime’s model,
- the same program and input can be compared exactly against another run,
- break/watch logic is checked in the shader rather than through external traps,
- post-execution analysis can inspect preserved state instead of reconstructed logs,
- correctness and security properties can be studied from exact traces.

This does not mean GPU debugging replaces all CPU tooling. It means a deterministic GPU runtime can serve as a distinct class of debugging platform, with different strengths and different failure modes.

## 6. Representative Workflows

### 6.1 Post-Mortem Crash Analysis

`gpu-fuzz` runs randomized inputs and captures failures along with preserved trace history. Instead of reproducing a crash under a separate debugger, the failing execution can be inspected immediately with `gpu-trace`, `gpu-asm`, `gpu-xray`, or `gpu-reverse`.

### 6.2 Root-Cause Isolation

`gpu-bisect` and `gpu-watch` narrow down where an incorrect value was introduced. `gpu-reverse` then traces backward from a final register state to the earlier writes that produced it.

### 6.3 Constant-Time Verification

`gpu-const-time` and `gpu-timing-proof` compare runs with different inputs using exact cycle counts and trace sequences. This is valuable for cryptographic code because CPU noise often makes similar analysis inconclusive.

### 6.4 Program Understanding

`gpu-profile`, `gpu-stack`, `gpu-heat`, and `gpu-asm` support static-dynamic understanding of unfamiliar binaries, especially when paired with symbol data from unstripped ELF files.

## 7. Evaluation Summary

This draft intentionally avoids inventing unpublished numbers. The final version should fill in the following measured results from the repository’s validated test and demo runs.

### 7.1 Implementation Scope

- Number of shell-facing commands: `26`
- Underlying execution substrate: ARM64 on Metal via `ncpu_metal`
- Trace model: fixed-size circular buffer with per-instruction architectural snapshots
- Integration surface: Alpine GPU shell plus Python/Rust runtime APIs

### 7.2 Verification Scope

Fill in:

- dedicated debugging-tool test count,
- shell-level integration test count,
- total repository test count at submission time,
- commands demonstrated end-to-end against real ELF workloads such as BusyBox.

### 7.3 Qualitative Outcomes

The final paper should summarize:

- deterministic replay behavior,
- exact trace diff usefulness,
- post-mortem debugging without rerun/reproduction,
- symbol-aware profiling and stack reconstruction when ELF symbols are present,
- examples where reverse analysis or constant-time verification shortened diagnosis time.

## 8. Limitations

The toolkit is strong, but it is not magic.

- The trace does not capture all 31 general-purpose registers per instruction, so some reverse analyses are stronger for low registers than for arbitrary architectural state.
- Symbol-aware output depends on unstripped ELF symbol availability.
- The execution model is deterministic by construction, which is useful for debugging but does not model every source of behavior seen on real out-of-order CPUs.
- The environment is currently specialized to the project’s ARM64-on-Metal runtime rather than a drop-in debugger for arbitrary host-native processes.
- Some capabilities, especially memory safety and constant-time analysis, are currently phrased as shell tools and engineering workflows rather than as fully formal verification systems.

These are acceptable limitations for a first standalone paper because the goal is to demonstrate a new debugging substrate, not to claim universal replacement of mature CPU debuggers.

## 9. Relationship to the Broader nCPU Project

The broader nCPU project includes neural arithmetic, a neural operating system layer, GPU-native ELF execution, and a complete shell environment. This paper deliberately narrows scope. It treats the GPU debugging toolkit as an independent systems contribution with its own motivation, architecture, and evaluation criteria.

That separation is useful for publication. The main nCPU paper argues for an end-to-end computational stack. This paper argues for a debugging and analysis platform enabled by deterministic GPU execution. The audiences overlap, but the contribution boundaries are different enough to justify a separate manuscript.

## 10. Future Work

Several directions would strengthen a full publication:

1. Extend trace capture to more registers and richer memory-event metadata.
2. Add automatic symbol resolution and source-level annotation where debug data is available.
3. Quantify crash-analysis turnaround versus conventional CPU fuzzing plus debugger workflows.
4. Formalize the constant-time analysis claim with a clearer security model and threat assumptions.
5. Compare zero-overhead post-mortem sanitization against CPU sanitizers on representative memory bugs.
6. Explore whether the same debugging model generalizes beyond ARM64-on-Metal to other deterministic accelerator runtimes.

## 11. Conclusion

This toolkit demonstrates that debugging on GPU can be more than an exotic trace viewer. When execution is deterministic, architectural state is explicit, and post-execution history is preserved, debugging changes character. Replay becomes exact. Trace diffing becomes exact. Crash analysis can begin immediately. Security checks such as constant-time comparison become more credible because they are not drowned in host CPU noise.

The standalone value of the work is the combination of these properties in a finished command surface. The broader nCPU project may have motivated the implementation, but the GPU-native debugging toolkit deserves to be evaluated as its own systems artifact and, in that form, it is substantial enough to support a dedicated paper.

File changed: `/Users/bobbyprice/projects/nCPU/paper/gpu_debugging_toolkit_paper.md`
