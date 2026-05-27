# nCPU Trained Model Index

Every ALU operation in the neural CPU is a trained neural network.

## ALU (`alu/`)

| File | Architecture | Size | Accuracy | Latency | Status |
|------|-------------|------|----------|---------|--------|
| `arithmetic.pt` | Sequential(3→128→64→2), sigmoid | 38K | 100% | 248 us | **WIRED** — ADD/SUB/INC/DEC via Kogge-Stone CLA (8 neural passes) |
| `carry_combine.pt` | Sequential(4→64→32→2), sigmoid | 13K | 100% | — | **WIRED** — Carry-combine operator for parallel-prefix addition |
| `multiply.pt` | Parameter[256,256,16] byte-pair LUT | 4.0M | 100% | 22 us | **WIRED** — MUL via batched byte-pair lookup (O(1)) |
| `divide.pt` | Sequential(3→64→32→2), sigmoid | 13K | Trained | — | LOADED — architecture ready, not exposed in ISA |
| `logical.pt` | Parameter[7,4] truth tables | 1.7K | 100% | 22 us | **WIRED** — AND/OR/XOR via vectorized truth table lookup (O(1)) |
| `compare.pt` | Linear(3→3), sigmoid | 2.0K | 100% | — | LOADED — CMP uses neural subtraction via CLA instead |

**Performance insight**: MUL (22 us) is **12x faster** than ADD (248 us). Even with Kogge-Stone carry-lookahead (3.3x faster than ripple-carry), the byte-pair LUT's O(1) lookup beats O(log n) parallel-prefix carry.

## Shifts & Rotates (`shifts/`)

| File | Architecture | Size | Accuracy | Latency | Status |
|------|-------------|------|----------|---------|--------|
| `lsl.pt` | shift_decoder(64→768→64) + index_net(128→768→64) + validity_net(128→384→1), temp=0.01 | 5.6M | 100% | 437 us | **WIRED** — SHL via vectorized attention-based shifting (3 batched passes) |
| `lsr.pt` | Same architecture as lsl.pt | 5.6M | 100% | 431 us | **WIRED** — SHR via vectorized attention-based shifting (3 batched passes) |
| `asr.pt` | Same architecture | 2.8M | 100% | — | ARCHIVED — ASR handled natively by NeuralCPU |
| `rol.pt` | Same architecture | 2.6M | 100% | — | ARCHIVED — ROL handled natively by NeuralCPU |

**Performance insight**: Shifts were vectorized from 64 sequential per-bit passes (2,833 us) to 3 batched passes (434 us) --- a 6.5x speedup. Now comparable to sqrt.

## Math Functions (`math/`)

| File | Architecture | Size | Accuracy | Latency | Status |
|------|-------------|------|----------|---------|--------|
| `sincos.pt` | 5× SinCosBlock(Linear+sin activation) → Linear(512,2) | 4.0M | Trained | 47 us | **WIRED** — sin/cos via bridge.sin()/cos() |
| `atan2.pt` | 6 residual layers × (Linear(512,512)+BatchNorm+ReLU) → Linear(512,2) | 6.1M | Trained | 1,055 us | **WIRED** — atan2 via bridge.atan2() |
| `sqrt.pt` | Two-stage: initial(1→256→1) + refine(2→256→1), BatchNorm | 536K | Trained | 524 us | **WIRED** — sqrt via bridge.sqrt() |
| `exp.pt` | 4-layer MLP: 1→256→256→256→1, ReLU | 521K | Trained | 21 us | **WIRED** — exp via bridge.exp_() |
| `log.pt` | 4-layer MLP: 1→256→256→256→1, ReLU | 521K | Trained | 22 us | **WIRED** — log via bridge.log_() |
| `doom_trig.pt` | Buffer[8192] sine + cosine tables, fixed-point | 66K | 100% | — | ARCHIVED — lookup table, not a neural network |

## Register File (`register/`)

| File | Architecture | Size | Status |
|------|-------------|------|--------|
| `register_file.pt` | RegisterBase + SPSwitch + XZRDetector + WHandler + FlagSelector | 241K | ARCHIVED — NeuralCPU uses GPU tensor indexing (faster and correct) |
| `register_vsa.pt` | Vector Symbolic Architecture (complex, not reconstructed) | 7.7M | ARCHIVED — trained for different system |

## Memory Operations (`memory/`)

| File | Architecture | Size | Status |
|------|-------------|------|--------|
| `stack.pt` | AddrArith + MemAddr + OpNet(65→128→64→2) | 374K | ARCHIVED — NeuralCPU handles memory natively as GPU tensor slicing |
| `pointer.pt` | AddrArith + MemAddr(64→128→256) | 306K | ARCHIVED — trained for different ISA |
| `function_call.pt` | AddrArith + ReturnAddrNet + TargetSelector + RetProcessor | 196K | ARCHIVED — trained for different ISA |

## Decoder (`decoder/`)

| File | Architecture | Size | Status |
|------|-------------|------|--------|
| `arm64_decoder.pt` | Transformer: BitEmbed + PosEmbed → SelfAttn(8-head) → FieldExtractor → 7 DecoderHeads | 6.5M | ARCHIVED — NeuralCPU uses optimized neural extractors + lookup tables |

## Decoder LLM (`decode_llm/`)

Qwen2.5-Coder-1.5B with LoRA adapter for semantic instruction decode.
100% accuracy on 33,750-step training set. **WIRED** in ncpu.model real mode.

## Research (`research/`)

Contains experimental / research models that are **not** part of the production wired neural ALU.

- `research/predictors/` — Microarchitectural neural predictors (branch, hazard, dependency, instruction scheduler, direct_alu). These are opt-in research components for the differentiable / neural CPU surface. See `research/predictors/README.md`.
- `research/neuros/` — Neural OS ("neuros") models (assembler, cache, compiler, mmu, scheduler, security, tlb, watchdog, etc.). Used by `ncpu/os/neuros/` and `ncpu/neural/metal_neural_os*`.
- `neural/` — Neural state / world model components (CAM, registers, ECC memory). Used by JEPA and differentiable research paths.

These models are distinct from the production models in `alu/`, `math/`, and `shifts/`.

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| **WIRED** | 13 | Active in neural ALU pipeline (8 ALU + 5 math) |
| LOADED | 2 | Weights loaded, architecture reconstructed, not in dispatch path |
| ARCHIVED | 8 | Architecture reconstructed, available for future work |

**Total production models**: 23 (~61 MB, 135 MB with decode_llm)
**Architecture classes reconstructed**: 11 of 12 orphaned models (all except register_vsa.pt)
**Neural ALU bridge**: Always on by default (off in fast mode)
**Model load time**: 60ms total (all 23 models)

## Performance Tiers

| Tier | Latency | Operations | Strategy |
|------|---------|-----------|----------|
| O(1) Fast | ~22 us | exp, log, mul, and, or, xor | Single-pass lookup/MLP |
| O(1) Medium | ~48 us | sin, cos | Multi-block deep network |
| O(log n) CLA | ~248 us | add, sub, cmp | Kogge-Stone parallel-prefix carry (8 neural passes) |
| O(3) Batched | ~434 us | shl, shr | 3 batched attention passes (vectorized) |
| O(n) Sequential | ~522 us | sqrt | 2 stages + BatchNorm batch padding |
| O(n) Heavy | ~935 us | atan2 | 6 sequential passes + batch padding |

**Fastest**: exp (21 us) | **Slowest**: atan2 (935 us) | **Spread**: ~45x
