# Bridge Demo — A Program No Human Wrote, Running on the Computer Made of Neural Networks

**ROADMAP Rung 8.** One artifact that fuses the two halves of nCPU: the
program-synthesis pillar (nsynth) and the GPU-as-computer pillar (the
rust_metal ARM64 Metal kernel with its self-hosting C compiler).

## What happens

```
I/O examples ──► nsynth (Rust gradient synthesizer)     "no human wrote it"
                     │  Mog program
                     ▼
              Mog → C rewrite (deterministic, ~20 lines)
                     │  C source
                     ▼
   self-hosting cc.c, RUNNING ON the Metal GPU          "the GPU computer
   (ncpu_metal.run_elf + GPU VFS)                        compiles it itself"
                     │  raw ARM64 binary in GPU VFS
                     ▼
              minimal-ELF wrap (2 PT_LOAD segments)
                     │
                     ▼
   execution ON the same rust_metal Metal kernel        "and runs it"
   (results printed via real write() syscalls)
                     │
                     ▼
   verification against oracle + local clang + Mog→Python transpile
```

The synthesized function is never touched by a human at any stage. The only
handwritten code in the GPU binary is a generic driver (`print_num` + `main`)
that calls the synthesized function on each test input and prints the result —
it contains zero knowledge of the algorithm.

## What it proves

1. **End-to-end machine authorship.** nsynth was given only ten I/O pairs
   (`1→1, 2→5, 3→14, …`) and produced, via gradient-based synthesis
   (`synth_gradient`, ~9s fresh with the solved-program cache disabled), a
   correct `sum_of_squares` loop program in Mog. No template for this
   function was consulted; verification inside nsynth gates the output.
2. **The GPU computer is a real computer.** The same self-hosting C compiler
   described in paper §16 (`ncpu/os/gpu/programs/tools/cc.c`, ~4,200 lines of
   freestanding C) runs as an ELF on the rust_metal kernel, reads the source
   from the GPU virtual filesystem, and emits ARM64 machine code into that
   filesystem — 171,715 GPU cycles for this program.
3. **The synthesized code executes natively on the GPU substrate.** The
   GPU-compiled binary runs on the same Metal kernel (3,195 cycles), printing
   results through genuine ARM64 `write` syscalls trapped by the kernel.
4. **The outputs are correct beyond the training distribution.** GPU outputs
   match a closed-form oracle on all six training inputs, both nsynth
   holdouts (7→140, 10→385), **and two inputs the synthesizer never saw in
   any form** (12→650, 20→2870). They also match a local clang build of the
   identical C source and nsynth's own Mog→Python transpile run locally.

## Verified result (2026-06-12, Apple Silicon, Metal)

```
GPU outputs: [1, 5, 14, 30, 55, 91, 140, 385, 650, 2870]
checks: gpu_vs_oracle OK, gpu_vs_local_clang OK, gpu_vs_python_transpile OK
synthesis: synth_gradient, 9.41s (fresh, cache disabled)
GPU compile: 171,715 cycles (0.375s) — self-hosting cc on Metal
GPU execute: 3,195 cycles (0.021s) — clean EXIT(0)
```

Artifacts: [`artifacts/bridge_demo_result.json`](../artifacts/bridge_demo_result.json)
(problem, examples, Mog source, C source, GPU stats, outputs, expected,
per-check booleans, sha256 of the GPU-compiled binary) and
[`artifacts/bridge_demo_transcript.txt`](../artifacts/bridge_demo_transcript.txt).

## Reproduce

Prerequisites (macOS + Apple Silicon):

```bash
# 1. nsynth binary
cd nsynth && cargo build --release && cd ..

# 2. rust_metal Python module (either)
pip install kernels/rust_metal/dist/ncpu_metal-0.1.0-cp312-abi3-macosx_11_0_arm64.whl
# or: cd kernels/rust_metal && maturin develop --release

# 3. ARM64 cross-compiler (compiles cc.c itself — Layer 1 of the meta stack)
brew install aarch64-elf-gcc

# 4. clang (ships with Xcode CLT) for the local cross-check
```

Run:

```bash
python3 demos/bridge/synthesized_on_gpu.py          # fresh synthesis (~1 min)
python3 demos/bridge/synthesized_on_gpu.py --quick  # allow nsynth solved-cache
```

Exit code 0 + `RESULT: MATCH` means every check passed. Any mismatch is
recorded honestly in the JSON artifact (`"match": false`) and the script
exits 1.

## Implementation notes

- **Why wrap the cc output in an ELF?** cc.c emits a raw position-fixed image
  (code at `0x10000`, optional `NCCD`-tagged data section for `0x50000`) for
  the original MLX-kernel shell flow. The rust_metal loader consumes static
  `ET_EXEC` ELFs, so `wrap_raw_in_elf()` adds a 64-byte ELF header plus one
  or two `PT_LOAD` program headers around the unmodified GPU-compiled bytes.
- **Why not the raw `MetalCPU` class for execution?** Its basic shader does
  not trap `SVC`, so a program's exit/write syscalls are skipped. The full
  kernel behind `run_elf` (`kernels/rust_metal/src/core/full_arm64.rs`)
  handles syscalls properly — and gives the demo real stdout capture.
- **Mog → C** is a ~20-line rewrite: `fn f(a: i64) -> i64 {` →
  `long f(long a) {`, `x: i64 = e;` → `long x = e;`, and parenthesizing
  `while`/`if` conditions. Everything else in the scalar-i64 Mog subset is
  already valid C. The pass fails closed if any Mog token survives.
- **Honesty controls:** synthesis runs with `NSYNTH_CACHE_PATH=""` by default
  so every run is a genuine fresh synthesis; the unseen inputs (12, 20) are
  absent from both the example set and the holdout set passed to nsynth;
  expected values come from the closed form `n(n+1)(2n+1)/6`, not from any
  stage of the pipeline.
