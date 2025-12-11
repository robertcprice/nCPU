# KVRM-CPU Decode LLM Training Results

## Model Overview

- **Base Model**: Qwen/Qwen2.5-Coder-1.5B
- **Training Method**: LoRA (Low-Rank Adaptation)
- **Training Data**: 50,000 assembly instruction → JSON decode pairs
- **Checkpoint**: `models/decode_llm/checkpoint-2500`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Target Modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Learning Rate | 3e-4 |
| Batch Size | 4 |
| Max Sequence Length | 128 |
| Warmup Ratio | 0.05 |
| Weight Decay | 0.01 |

## Training Progress

| Step | Epoch | Training Loss | Eval Loss | Notes |
|------|-------|---------------|-----------|-------|
| 100 | 0.01 | 2.2209 | - | Initial loss |
| 500 | 0.04 | 0.4055 | 0.3984 | Rapid convergence |
| 1000 | 0.09 | 0.3951 | 0.3861 | Stabilizing |
| 1500 | 0.13 | 0.3900 | 0.3964 | Slight overfit |
| 2000 | 0.18 | 0.3857 | 0.3923 | Good progress |
| 2500 | 0.22 | 0.3834 | 0.3838 | Current checkpoint |

Training was stopped at ~8% completion (2500/33750 steps).

## Validation Results

### At Step 2500 (Current Checkpoint)

**Built-in Validation (100% accuracy on 7 test cases)**:
- MOV R3, 42 → OP_MOV_REG_IMM ✅
- ADD R0, R1, R2 → OP_ADD ✅
- CMP R3, R4 → OP_CMP ✅
- JMP loop → OP_JMP ✅
- HALT → OP_HALT ✅
- sub r5 r3 r1 → OP_SUB ✅
- FOO R1, R2 → OP_INVALID ✅

### Extended Inference Tests (80% accuracy on 15 test cases)

| Instruction | Expected | Actual | Status |
|------------|----------|--------|--------|
| MOV R3, 42 | OP_MOV_REG_IMM | OP_MOV_REG_IMM | ✅ |
| MOV R0, R1 | OP_MOV_REG_REG | OP_MOV_REG_REG | ✅ |
| ADD R0, R1, R2 | OP_ADD | OP_INVALID | ❌ |
| SUB R5, R3, R1 | OP_SUB | OP_SUB | ✅ |
| MUL R0, R1, R2 | OP_MUL | OP_MUL | ✅ |
| CMP R3, R4 | OP_CMP | OP_CMP | ✅ |
| JMP 10 | OP_JMP | OP_JMP | ✅ |
| JZ 5 | OP_JZ | OP_JZ | ✅ |
| JNZ 20 | OP_JNZ | OP_JNZ | ✅ |
| HALT | OP_HALT | OP_HALT | ✅ |
| NOP | OP_NOP | OP_NOP | ✅ |
| INC R0 | OP_INC | OP_INVALID | ❌ |
| DEC R1 | OP_DEC | OP_INVALID | ❌ |
| add r0, r1, r2 | OP_ADD | OP_ADD | ✅ |
| BADOP R1 | OP_INVALID | OP_INVALID | ✅ |

## Known Limitations

With only 8% of training completed:

1. **ADD Instruction (uppercase)**: Model sometimes outputs OP_INVALID for uppercase ADD, but handles lowercase "add" correctly
2. **INC/DEC Instructions**: Not yet learned by the model at this checkpoint
3. **Training Recommendations**: Continue training to at least 50% completion for better coverage

## Usage

### Mock Mode (Recommended for Development)

```python
from kvrm_cpu.decode_llm import DecodeLLM

decoder = DecodeLLM(mock_mode=True)
result = decoder.decode("MOV R3, 42")
print(result.key)  # OP_MOV_REG_IMM
print(result.params)  # {'dest': 'R3', 'value': 42}
```

### Real Mode (Trained Model)

```python
from kvrm_cpu.decode_llm import DecodeLLM

decoder = DecodeLLM(
    mock_mode=False,
    model_path="models/decode_llm/checkpoint-2500"
)
decoder.load()  # Loads Qwen2.5-Coder-1.5B + LoRA adapter

result = decoder.decode("SUB R5, R3, R1")
print(result.key)  # OP_SUB
print(result.params)  # {'dest': 'R5', 'src1': 'R3', 'src2': 'R1'}
```

## Model Architecture

```
Qwen/Qwen2.5-Coder-1.5B (1.56B params)
    └── LoRA Adapters (18.5M trainable params, 1.18% of total)
        └── Trained on CPU instruction decode task
```

## Files

- `models/decode_llm/checkpoint-2500/adapter_config.json` - LoRA configuration
- `models/decode_llm/checkpoint-2500/adapter_model.safetensors` - Trained weights (~74MB)
- `models/decode_llm/checkpoint-2500/trainer_state.json` - Training state and metrics
- `training/train_decode.py` - Training script
- `data/cpu_decode_train.jsonl` - Training data (50K samples)

## Model Selection Rationale

**Why Qwen2.5-Coder-1.5B over TinyLlama-1.1B?**

1. **Code-Specialized**: Qwen2.5-Coder is trained specifically on code, making it better suited for parsing assembly instructions
2. **JSON Generation**: Better at generating structured JSON output due to code training
3. **Modern Architecture**: Uses latest transformer optimizations
4. **Strong Performance**: Achieves state-of-the-art results at its size class for code-related tasks

## Continuing Training

To resume training from the current checkpoint:

```bash
cd /Users/bobbyprice/projects/kvrm-cpu
./venv/bin/python training/train_decode.py \
    --data data/cpu_decode_train.jsonl \
    --output models/decode_llm \
    --epochs 3 \
    --batch-size 4
```

Training time estimate (on Apple M-series):
- Full 3 epochs: ~9-10 hours
- Current progress: ~1.5 hours (8% complete)
