# KVRM-CPU Architecture

A model-native CPU emulator demonstrating the KVRM paradigm at the lowest level of computing.

## Overview

KVRM-CPU replaces traditional hardcoded instruction decode logic with a semantic LLM-based decoder that emits verified registry keys. This proves that even the most fundamental computing operations can be handled through the KVRM (Key-Value Response Mapping) pattern.

### Traditional CPU vs KVRM-CPU

```
Traditional CPU:  MEMORY → FETCH → DECODE → EXECUTE → STATE
                                     ↓
                              [Hardcoded Silicon]
                              Bit patterns → operations

KVRM-CPU:         MEMORY → FETCH → DECODE_LLM → KEY → REGISTRY → EXECUTE → STATE
                                       ↓          ↓        ↓
                                 [Semantic LLM] [JSON]  [Verified]
                                 Instructions → {"op": "OP_ADD", ...}
                                               Primitives Only
```

## Core Components

### 1. CPUState (`state.py`)

Immutable state representation for the CPU:

```python
@dataclass
class CPUState:
    registers: Dict[str, int]  # R0-R7 (8 general purpose)
    pc: int                    # Program counter
    flags: Dict[str, bool]     # ZF (zero), SF (sign)
    memory: List[str]          # Program instructions
    labels: Dict[str, int]     # Label → address mapping
    halted: bool               # Execution stopped
    cycle_count: int           # Total cycles executed
```

**Registers:**
- `R0-R7`: 8 general-purpose 32-bit signed integers
- `PC`: Program counter (implicit)
- `FLAGS`: ZF (zero flag), SF (sign flag)

### 2. Registry (`registry.py`)

The verified execution layer. Only pre-defined operations can execute:

| Key | Parameters | Description |
|-----|------------|-------------|
| `OP_MOV_REG_IMM` | dest, value | Load immediate into register |
| `OP_MOV_REG_REG` | dest, src | Register-to-register copy |
| `OP_ADD` | dest, src1, src2 | Addition |
| `OP_SUB` | dest, src1, src2 | Subtraction |
| `OP_MUL` | dest, src1, src2 | Multiplication |
| `OP_CMP` | src1, src2 | Compare (sets flags) |
| `OP_JMP` | addr | Unconditional jump |
| `OP_JZ` | addr | Jump if zero flag set |
| `OP_JNZ` | addr | Jump if zero flag not set |
| `OP_HALT` | - | Stop execution |
| `OP_NOP` | - | No operation |
| `OP_INVALID` | raw | Error handler |

Each primitive is a pure function: `(state, params) → new_state`

### 3. Decode LLM (`decode_llm.py`)

The semantic decoder that transforms instructions into verified keys:

```
Input:  "ADD R3, R1, R2"
Output: {"key": "OP_ADD", "params": {"dest": "R3", "src1": "R1", "src2": "R2"}}
```

**Modes:**
- **Mock Mode**: Rule-based parser for development/testing
- **Real Mode**: Trained TinyLlama-1.1B with LoRA adapters

### 4. CPU Orchestrator (`cpu.py`)

The main execution engine:

```python
class KVRMCPU:
    def load_program(source: str)      # Parse and load assembly
    def step() -> TraceEntry           # Single cycle execution
    def run(max_cycles) -> List[...]   # Run until HALT
    def get_register(reg) -> int       # Read register value
    def dump_registers() -> dict       # All register values
    def get_trace() -> List[TraceEntry]  # Execution history
```

## Execution Flow

```
1. FETCH:    instruction = memory[PC]
2. DECODE:   key, params = decode_llm.decode(instruction, labels)
3. VALIDATE: assert key in VALID_REGISTRY_KEYS
4. EXECUTE:  new_state = registry[key](state, params)
5. TRACE:    record(cycle, instruction, key, params, pre_state, post_state)
6. REPEAT:   until HALT or max_cycles
```

## Instruction Set Architecture (ISA)

### Data Movement
```asm
MOV R0, 42      ; Load immediate value
MOV R1, R0      ; Copy register to register
```

### Arithmetic
```asm
ADD R3, R1, R2  ; R3 = R1 + R2
SUB R3, R1, R2  ; R3 = R1 - R2
MUL R3, R1, R2  ; R3 = R1 * R2
```

### Comparison
```asm
CMP R1, R2      ; Sets ZF if R1 == R2, SF if R1 < R2
```

### Control Flow
```asm
JMP loop        ; Unconditional jump to label
JZ done         ; Jump if ZF set (values equal)
JNZ loop        ; Jump if ZF not set (values not equal)
```

### Special
```asm
HALT            ; Stop execution
NOP             ; No operation
```

### Labels
```asm
loop:           ; Define label at current address
    ADD R0, R0, R1
    JMP loop    ; Reference label
```

## Training Pipeline

### Data Generation (`training/generate_cpu_data.py`)

Generates 50,000 training samples with augmentations:

| Category | Count | Examples |
|----------|-------|----------|
| MOV reg, imm | 10k | "MOV R3, 42", "mov r1, 0x1F" |
| MOV reg, reg | 5k | "MOV R1, R2" |
| ADD/SUB/MUL | 15k | "ADD R3 R1 R2", "sub r0, r5, r4" |
| CMP | 5k | "CMP R1, R2" |
| JMP/JZ/JNZ | 10k | "JMP loop", "jz 10" |
| HALT/NOP | 2k | "HALT", "nop" |
| Invalid | 3k | "FOO R1", "" |

**Augmentations:**
- Random capitalization: ADD, add, Add
- With/without commas: ADD R3 R1 R2, ADD R3, R1, R2
- Whitespace variations: extra spaces, tabs
- Label vs numeric addresses

### Model Training (`training/train_decode.py`)

**Configuration:**
- Base model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- LoRA: r=16, alpha=32, dropout=0.05
- Target modules: q_proj, k_proj, v_proj, o_proj
- Epochs: 3
- Batch size: 4 (MPS-optimized)
- Learning rate: 3e-4

**Prompt Format:**
```
### Context:
ADD R3, R1, R2

### Key:
{"key": "OP_ADD", "params": {"dest": "R3", "src1": "R1", "src2": "R2"}}
```

## Benchmark Results

Comparison of Traditional CPU emulator vs KVRM-CPU:

| Mode | Avg Time/Run | Overhead | Correctness |
|------|--------------|----------|-------------|
| Traditional | ~0.1ms | 1x | 100% |
| KVRM Mock | ~0.2ms | 2x | 100% |
| KVRM Real | ~50ms (uncached) | 500x | >95% |

The overhead is the cost of semantic understanding - acceptable for verified, auditable computing.

## KVRM Benefits

1. **Semantic Understanding**: Instructions decoded by meaning, not bit patterns
2. **Full Auditability**: Every decode decision is traceable
3. **Natural Language Input**: "Add R3, R1, R2" parsed semantically
4. **Verification Layer**: Decoded keys checked against verified registry
5. **Extensibility**: New instructions via training, not silicon changes
6. **Error Explanations**: Invalid instructions get semantic error messages

## Example Programs

### Sum 1-10 (`programs/sum_1_to_10.asm`)
```asm
    MOV R0, 0       ; sum = 0
    MOV R1, 1       ; counter = 1
    MOV R2, 11      ; limit
    MOV R3, 1       ; increment
loop:
    ADD R0, R0, R1  ; sum += counter
    ADD R1, R1, R3  ; counter++
    CMP R1, R2      ; compare to limit
    JNZ loop        ; continue if not equal
    HALT            ; R0 = 55
```

### Fibonacci (`programs/fibonacci.asm`)
```asm
    MOV R0, 0       ; fib(0)
    MOV R1, 1       ; fib(1)
    MOV R2, 10      ; iterations
    MOV R3, 0       ; counter
    MOV R4, 1       ; constant 1
loop:
    MOV R5, R1      ; temp = current
    ADD R1, R0, R1  ; current = prev + current
    MOV R0, R5      ; prev = temp
    ADD R3, R3, R4  ; counter++
    CMP R3, R2
    JNZ loop
    HALT            ; R1 = 89
```

## Usage

### CLI
```bash
# Run with mock decoder
python main.py --program programs/sum_1_to_10.asm --mode mock

# Run with trained model
python main.py --program programs/fibonacci.asm --mode real --model models/decode_llm

# Show execution trace
python main.py --program programs/multiply.asm --trace
```

### Python API
```python
from kvrm_cpu import KVRMCPU

# Mock mode (no GPU needed)
cpu = KVRMCPU(mock_mode=True)
cpu.load_program(source_code)
trace = cpu.run()
print(cpu.dump_registers())

# Real mode (with trained model)
cpu = KVRMCPU(mock_mode=False, model_path="models/decode_llm")
cpu.load()
cpu.load_program(source_code)
trace = cpu.run()
cpu.unload()
```

### Gradio Demo
```bash
python demo/gradio_app.py
# Opens web interface at http://localhost:7861
```

## Project Structure

```
kvrm-cpu/
├── src/kvrm_cpu/
│   ├── __init__.py      # Public API
│   ├── state.py         # CPUState dataclass
│   ├── registry.py      # Verified primitives (12 ops)
│   ├── decode_llm.py    # Semantic decoder
│   └── cpu.py           # Main orchestrator
├── training/
│   ├── generate_cpu_data.py  # 50k samples
│   └── train_decode.py       # LoRA fine-tuning
├── programs/
│   ├── sum_1_to_10.asm
│   ├── fibonacci.asm
│   └── multiply.asm
├── benchmarks/
│   └── cpu_benchmark.py      # Traditional vs KVRM
├── tests/
│   ├── test_state.py
│   ├── test_registry.py
│   ├── test_decode.py
│   └── test_programs.py
├── demo/
│   └── gradio_app.py         # Web interface
├── main.py                   # CLI entry point
└── requirements.txt
```

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
datasets>=2.14.0
accelerate>=0.24.0
gradio>=4.0.0
pytest>=7.0.0
tqdm>=4.66.0
```

## Theoretical Foundation

KVRM-CPU demonstrates that the KVRM paradigm applies at every level of computing:

1. **Vector Operations** (kvrm-vector): `push`, `pop`, `sort` → semantic intents
2. **CPU Operations** (kvrm-cpu): `ADD`, `MOV`, `JMP` → verified keys
3. **OS Operations** (future): `open`, `read`, `exec` → semantic syscalls

The key insight: **Any operation that can be verified can be semantically decoded.**

This creates a foundation for:
- Auditable computing where every decision is traceable
- Extensible systems where new operations are trained, not coded
- Natural language interfaces to low-level systems
- Verified execution where only approved operations run
