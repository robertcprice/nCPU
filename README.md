# KVRM-CPU

A model-native CPU emulator demonstrating the KVRM (Key-Value Response Mapping) paradigm at the lowest level of computing.

## Overview

KVRM-CPU replaces traditional hardcoded instruction decode logic with a semantic LLM-based decoder that emits verified registry keys. This proves that even the most fundamental computing operations can be handled through the KVRM pattern.

```
Traditional CPU:  MEMORY → FETCH → DECODE → EXECUTE → STATE
                                    ↓
                             [Hardcoded Silicon]

KVRM-CPU:         MEMORY → FETCH → DECODE_LLM → KEY → REGISTRY → EXECUTE → STATE
                                      ↓          ↓        ↓
                                [Semantic LLM] [JSON]  [Verified]
```

## Features

- **LLM-Based Decode**: Uses fine-tuned Qwen2.5-Coder-1.5B with LoRA for instruction decoding
- **Verified Registry**: Only pre-defined operations can execute - no arbitrary code execution
- **Immutable State**: Functional programming approach with immutable CPU state
- **Full ISA**: Supports MOV, ADD, SUB, MUL, CMP, JMP, JZ, JNZ, JS, JNS, INC, DEC, HALT, NOP
- **Dual Mode**: Run with trained LLM or mock (rule-based) decoder

## Installation

```bash
git clone https://github.com/yourusername/kvrm-cpu.git
cd kvrm-cpu
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Mock Mode (No GPU Required)

```python
from kvrm_cpu import KVRMCPU

cpu = KVRMCPU(mock_mode=True)
program = """
    MOV R0, 10
    MOV R1, 20
    ADD R2, R0, R1
    HALT
"""
result = cpu.load_and_run(program)
print(f"R2 = {result.registers['R2']}")  # R2 = 30
```

### Real LLM Mode

```python
from kvrm_cpu import KVRMCPU

cpu = KVRMCPU(mock_mode=False, model_path="models/decode_llm/checkpoint-2500")
cpu.load()  # Loads the trained model

program = """
    MOV R0, 5
    MOV R1, 1
loop:
    SUB R0, R0, R1
    CMP R0, R1
    JNZ loop
    HALT
"""
result = cpu.load_and_run(program)
```

## Trained Model

### Model Details

| Property | Value |
|----------|-------|
| Base Model | Qwen/Qwen2.5-Coder-1.5B |
| Training Method | LoRA (r=16, alpha=32) |
| Trainable Params | 18.5M (1.18% of total) |
| Training Data | 50K instruction-decode pairs |
| Checkpoint | models/decode_llm/checkpoint-2500 |

### Performance

At 8% training completion (2500/33750 steps):
- **Mock Mode**: 100% accuracy (rule-based)
- **LLM Mode**: 80% accuracy on extended test suite

See [Training Results](docs/TRAINING_RESULTS.md) for detailed metrics.

## Supported Instructions

| Instruction | Description | Example |
|-------------|-------------|---------|
| MOV Rd, imm | Load immediate | MOV R0, 42 |
| MOV Rd, Rs | Copy register | MOV R1, R0 |
| ADD Rd, Rs1, Rs2 | Addition | ADD R0, R1, R2 |
| SUB Rd, Rs1, Rs2 | Subtraction | SUB R3, R4, R5 |
| MUL Rd, Rs1, Rs2 | Multiplication | MUL R0, R1, R2 |
| CMP Rs1, Rs2 | Compare (sets flags) | CMP R0, R1 |
| JMP addr/label | Unconditional jump | JMP loop |
| JZ addr/label | Jump if zero | JZ done |
| JNZ addr/label | Jump if not zero | JNZ loop |
| JS addr/label | Jump if sign | JS negative |
| JNS addr/label | Jump if not sign | JNS positive |
| INC Rd | Increment | INC R0 |
| DEC Rd | Decrement | DEC R1 |
| HALT | Stop execution | HALT |
| NOP | No operation | NOP |

## Running Tests

```bash
# Run all unit tests (mock mode)
python -m pytest tests/ -v

# Run model inference tests (requires GPU/MPS)
python -m pytest tests/test_model_inference.py -v
```

## Training

To train or continue training the decode LLM:

```bash
# Generate training data (if needed)
python training/generate_cpu_data.py

# Train the model
python training/train_decode.py --epochs 3 --batch-size 4
```

## Project Structure

```
kvrm-cpu/
├── src/kvrm_cpu/
│   ├── __init__.py
│   ├── cpu.py          # Main KVRM-CPU class
│   ├── state.py        # Immutable CPU state
│   ├── registry.py     # Verified operation registry
│   └── decode_llm.py   # LLM-based instruction decoder
├── training/
│   ├── train_decode.py # Training script
│   └── generate_cpu_data.py # Data generation
├── models/
│   └── decode_llm/     # Trained model checkpoints
├── programs/
│   ├── sum.asm         # Sum 1 to N
│   ├── fibonacci.asm   # Fibonacci sequence
│   └── multiply.asm    # Multiplication via addition
├── tests/              # Unit and integration tests
└── docs/               # Documentation
```

## Documentation

- [Architecture](docs/architecture.md) - Detailed system design
- [Training Results](docs/TRAINING_RESULTS.md) - Model training metrics

## License

MIT
