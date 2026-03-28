# nCPU Model Reference

Comprehensive documentation for all 24 trained neural network models.

## 1. Neural Full Adder (`models/alu/arithmetic.pt`)

**Purpose**: Bit-serial binary addition — the fundamental building block for ADD, SUB, INC, DEC.

**Architecture**:
```
Input:  [bit_a, bit_b, carry_in]  — 3 floats
Hidden: Linear(3, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 2)
Output: sigmoid → [sum_bit, carry_out] — 2 floats
```

**How it works**: Called 32 times in sequence, once per bit position (LSB first). Each call produces one sum bit and a carry-out that feeds into the next iteration. This is exactly how a hardware ripple-carry adder works, but every gate is a neural network.

**SUB implementation**: Two's complement via complement-and-add. `a - b = a + (~b) + 1`. The complement is computed by flipping bits (`1.0 - bits_b`), and carry-in starts at 1.0.

**Training**: Supervised on all 2^3 = 8 input combinations. 100% accuracy after training.

**Wiring**: `NeuralALUBridge.add()`, `.sub()`, `.cmp()` — also used for INC (add 1) and DEC (sub 1).

**Latency**: ~826 us mean (32 sequential forward passes, ~26 us per pass). This makes addition the 3rd slowest operation — counterintuitively slower than multiplication (22 us) because carry propagation is irreducibly sequential.

---

## 2. Neural Multiplier LUT (`models/alu/multiply.pt`)

**Purpose**: 32-bit integer multiplication via byte-pair lookup.

**Architecture**:
```
Single parameter: nn.Parameter([256, 256, 16])
No layers — this is a learned lookup table.
Each entry: sigmoid(table[a_byte, b_byte]) → 16 bits representing the product.
```

**How it works**: Decompose each 32-bit operand into 4 bytes. For each non-zero byte pair (a_byte, b_byte), look up the 16-bit product in the table. Accumulate results with appropriate bit shifts. Up to 16 lookups per multiplication.

**Optimization**: All non-zero byte pairs gathered into a single tensor operation — batch lookup instead of 16 sequential calls.

**Training**: Supervised on all 256 × 256 = 65,536 byte pairs. 100% accuracy.

**Wiring**: `NeuralALUBridge.mul()`.

**Latency**: ~22 us mean (single batched gather). The fastest ALU operation — **38x faster than addition** because byte-pair products are independent (no carry chain).

---

## 3. Neural Logical (`models/alu/logical.pt`)

**Purpose**: Bitwise AND, OR, XOR (plus NOT, NAND, NOR, XNOR).

**Architecture**:
```
Single parameter: nn.Parameter([7, 4])
7 operations × 4 truth table entries.
Index: ops[op_idx, bit_a * 2 + bit_b] → sigmoid → threshold at 0.5
```

**How it works**: Each operation has a 4-entry truth table. For each of 32 bits, compute index = bit_a × 2 + bit_b, look up the truth table entry, apply sigmoid, threshold. All 32 bits processed in one vectorized step.

**Operations**: AND=0, OR=1, XOR=2, NOT=3, NAND=4, NOR=5, XNOR=6.

**Training**: Supervised on truth tables. 100% accuracy (trivially).

**Wiring**: `NeuralALUBridge.and_()`, `.or_()`, `.xor_()`.

**Latency**: ~22 us mean (single vectorized lookup for all 32 bits). Tied with MUL for fastest operation.

---

## 4. Neural Shift Left (`models/shifts/lsl.pt`)

**Purpose**: Logical shift left — trained to route bits to their correct shifted positions.

**Architecture** (three decomposed sub-networks):
```
shift_decoder:  Linear(64, 768) → ReLU → Linear(768, 768) → ReLU → Linear(768, 64)
index_net:      Linear(128, 768) → ReLU → Linear(768, 768) → ReLU → Linear(768, 64)
validity_net:   Linear(128, 384) → ReLU → Linear(384, 1)
temperature:    nn.Parameter (learned, value ≈ 0.01)
```

**How it works** (all 64 output bit positions computed in parallel):
1. Binary-encode shift amount → shift_decoder → softmax encoding (1 forward pass)
2. Build [64, 128] batch: 64×64 identity matrix (one-hot positions) concatenated with expanded shift encoding
3. index_net([64, 128]) → [64, 64] logits → softmax(logits / temperature, dim=1) → batched attention (1 forward pass)
4. validity_net([64, 128]) → [64, 1] sigmoid gate for all positions (1 forward pass)
5. Output bits = batched attention_weighted_sum × validity_gate

**Temperature convention**: `temperature ≈ 0.01` means DIVIDE by it (`softmax(logits / 0.01)`), which sharpens the distribution to near-hard attention. This is critical — multiplying by 0.01 would flatten the distribution.

**Training**: Supervised on exhaustive shift combinations. 100% accuracy for amounts 0-31.

**Wiring**: `NeuralALUBridge.shl()`.

**Latency**: ~465 us mean (3 batched forward passes: 1x shift_decoder + 1x index_net + 1x validity_net, all 64 output positions computed simultaneously). Previously 2,833 us with 64 sequential per-bit passes; vectorization yielded a 6.1x speedup. Now comparable to sqrt (~524 us) and faster than addition (~826 us).

---

## 5. Neural Shift Right (`models/shifts/lsr.pt`)

Same architecture as lsl.pt, trained for logical right shift.

**Wiring**: `NeuralALUBridge.shr()`.

**Latency**: ~462 us mean (3 batched forward passes, same architecture as LSL).

---

## 6. Neural Compare (`models/alu/compare.pt`)

**Architecture**: `Linear(3, 3)` with sigmoid activation.

**Status**: LOADED but not used in the dispatch path. CMP is implemented via neural subtraction through arithmetic.pt instead (same as real CPUs: CMP = SUB without storing the result). Flags are derived from the subtraction result: N = (result < 0), Z = (result == 0), C = unsigned(a >= b).

---

## 7. Neural Divider (`models/alu/divide.pt`)

**Architecture**: Same as arithmetic.pt but with `hidden_dim=64`:
```
Linear(3, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 2)
```

**Status**: LOADED, architecture reconstructed, not exposed in ISA.

---

## 8. Neural SinCos (`models/math/sincos.pt`)

**Purpose**: Approximate sin(x) and cos(x) simultaneously.

**Architecture**:
```
5 × SinCosBlock(Linear(dim, 512) → sin activation)
Final: Linear(512, 2)
Input: [batch, 1] (angle in radians)
Output: [batch, 2] (sin, cos)
```

**Checkpoint format**: `{model: state_dict, max_err: float, epoch: int}`

**Status**: **WIRED** via `NeuralALUBridge.sin()`, `.cos()`. Uses fixed-point convention (input ÷ 1000 = radians).

**Latency**: ~47 us mean (2 forward passes through the deep sine-activated network).

---

## 9. Neural Sqrt (`models/math/sqrt.pt`)

**Purpose**: Two-stage square root approximation with Newton-style refinement.

**Architecture**:
```
Stage 1 (initial): Linear(1,256) → BatchNorm → ReLU → Linear(256,256) → BatchNorm → ReLU → Linear(256,1)
Stage 2 (refine):  Linear(2,256) → BatchNorm → ReLU → Linear(256,256) → BatchNorm → ReLU → Linear(256,1)
```

Refine stage takes `[x, initial_estimate]` as input.

**Note**: BatchNorm uses `track_running_stats=False`, requiring batch_size ≥ 2 even in eval mode.

**Checkpoint format**: `{model: state_dict, rel_err: float, abs_err: float, epoch: int}`

**Status**: **WIRED** via `NeuralALUBridge.sqrt()`.

**Latency**: ~524 us mean (two-stage forward pass + BatchNorm batch padding overhead).

---

## 10. Neural Exp (`models/math/exp.pt`)

**Architecture**:
```
Linear(1, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, 1)
```

**Checkpoint format**: `{model: state_dict, error: float}`

**Status**: **WIRED** via `NeuralALUBridge.exp_()`.

**Latency**: ~21 us mean (single-pass 4-layer MLP). Tied for fastest operation.

---

## 11. Neural Log (`models/math/log.pt`)

Same architecture as NeuralExp.

**Status**: **WIRED** via `NeuralALUBridge.log_()`.

**Latency**: ~22 us mean (single-pass 4-layer MLP).

---

## 12. Neural Atan2 (`models/math/atan2.pt`)

**Purpose**: Two-argument arctangent approximation.

**Architecture**:
```
Encoder: Linear(6, 512) → BatchNorm → ReLU
6 residual layers: Linear(512, 512) → BatchNorm → ReLU (with skip connections)
Output: Linear(512, 2)
Input: [sin_a, cos_a, y≥0, x≥0, y<0, x<0] (6 features)
Output: [angle_sin, angle_cos] (2 outputs, converted to angle via atan2)
```

**Note**: Uses `track_running_stats=False` BatchNorm — requires batch_size ≥ 2.

**Checkpoint format**: Direct state_dict (not wrapped in dict).

**Status**: **WIRED** via `NeuralALUBridge.atan2()`.

**Latency**: ~1,055 us mean (6 residual layers + BatchNorm batch padding with duplicate input).

---

## 13. DOOM Trig LUT (`models/math/doom_trig.pt`)

**Purpose**: DOOM-compatible fixed-point trigonometry lookup table.

**Architecture**: Not a neural network — `register_buffer` tensors:
```
sine_table:   [8192] fixed-point values (scale 65536)
cosine_table: [8192] fixed-point values
```

**Status**: ARCHIVED — lookup table, not neural inference.

---

## 14. Neural Register File (`models/register/register_file.pt`)

**Architecture**: RegisterBase(5→64→32) + SPSwitch(6→64→32→1) + XZRDetector(5→32→16→1) + WHandler(64→96→64 + 128→96→32) + FlagSelector(4→16)

**Status**: ARCHIVED — NeuralCPU uses GPU tensor indexing for registers, which is both faster and correct. The register model was trained for a different ISA.

---

## 15. Neural Stack (`models/memory/stack.pt`)

**Architecture**: AddrArith(FullAdder) + MemAddr(64→128→256, learned temperature) + OpNet(65→128→64→2)

**Status**: ARCHIVED — NeuralCPU handles stack via native GPU memory tensor operations.

---

## 16. Neural Pointer (`models/memory/pointer.pt`)

**Architecture**: AddrArith(FullAdder) + MemAddr(64→128→256, learned temperature)

**Status**: ARCHIVED — NeuralCPU handles LDR/STR via GPU tensor memory slicing.

---

## 17. Neural Function Call (`models/memory/function_call.pt`)

**Architecture**: AddrArith + ReturnAddrNet(64→128→64→64) + TargetSelector(64→128→64) + RetProcessor(64→64→64)

**Status**: ARCHIVED — NeuralCPU handles BL/RET via GPU register/memory operations.

---

## 18. Neural ARM64 Decoder (`models/decoder/arm64_decoder.pt`)

**Architecture**: Transformer-based decoder:
```
Encoder:    BitEmbed(2, 64) + PosEmbed(32, 64) → Linear(128, 256)
Extractor:  SelfAttention(256, 8 heads) → FieldAttention(256, 8 heads, 6 queries)
Heads:      category(10), operation(128), rd(32), rn(32), rm(32), imm(26), flags(3)
Refine:     Linear(1536, 512) → ReLU → Linear(512, 256)
```

**Status**: ARCHIVED — NeuralCPU uses optimized neural extractors + lookup tables that are faster for real-time decode.

---

## 19-22. Shift/Rotate Variants

- **ASR** (`shifts/asr.pt`): Arithmetic shift right. Same NeuralShiftNet architecture. ARCHIVED.
- **ROL** (`shifts/rol.pt`): Rotate left. Same architecture. ARCHIVED.
- **Register VSA** (`register/register_vsa.pt`): Vector Symbolic Architecture. Too complex to reconstruct. ARCHIVED.
- **Decode LLM** (`decode_llm/`): Qwen2.5-Coder-1.5B LoRA. **WIRED** in real mode.

## Architecture Reconstruction Summary

11 of 12 orphaned models have been fully reconstructed in `ncpu/model/architectures.py`, verified with `strict=True` weight loading. Only `register_vsa.pt` remains unrecoverable due to its complex VSA architecture.
