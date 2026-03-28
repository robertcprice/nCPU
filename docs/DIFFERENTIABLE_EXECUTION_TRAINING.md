# Differentiable Execution as a Training Signal for Code Generation Models

## Abstract

We propose injecting nCPU's differentiable CPU into the training loop of a code-generating language model so that **execution error gradients flow directly back into the model's weights**. Instead of the standard generate-then-verify loop (REINFORCE, rejection sampling, or external tool-use), the model receives dense, per-operation gradient signal from actually running the code it produces — through a fully differentiable execution engine.

This is only possible because nCPU provides:
1. A differentiable ALU (100% accurate integer arithmetic with gradient flow)
2. A differentiable execution engine (soft PC, soft register addressing, Gumbel-softmax opcodes)
3. A differentiable compilation pipeline (source tokens → neural compiler → execution → loss)
4. A proven coprocessor injection mechanism (drop-in MLP replacement, ~113K params)

No other system offers all four. This document describes the architecture, implementation plan, and expected outcomes.

---

## 1. The Problem with Current Code Model Training

### 1.1 REINFORCE / Rejection Sampling (Sparse Signal)

Current code LM training (CodeRL, RLTF, etc.) works like:
```
model generates code → execute externally → binary pass/fail → REINFORCE gradient
```

The gradient says "this program was wrong" but NOT "the addition on line 5 should have been subtraction." The signal is sparse: one scalar reward for an entire program.

### 1.2 External Tool Use (No Gradient)

Tool-augmented models (Toolformer, etc.) can call a Python interpreter during inference, but:
- No gradient flows through the external tool
- The model treats execution as a black box
- Learning happens through prompt engineering, not weight updates

### 1.3 What We Want (Dense Signal)

```
model generates tokens → parse to nCPU ISA → differentiable execution → MSE on expected output
                    ↑                                                          |
                    └──── gradients flow ALL the way back ────────────────────┘
```

Every arithmetic operation, every register write, every conditional branch provides gradient signal. The model learns not just "this program is wrong" but "this specific operation computed the wrong value, and here's the direction to fix it."

---

## 2. Architecture

### 2.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                │
│                                                                         │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────┐               │
│  │ Code LM  │───>│ Code-to-ISA  │───>│ Differentiable  │               │
│  │ (frozen  │    │   Parser     │    │    Engine        │               │
│  │  +copro) │    │              │    │ (execute_soft)   │               │
│  └──────────┘    └──────────────┘    └────────┬────────┘               │
│       ↑                                        │                        │
│       │          ┌──────────────┐              ▼                        │
│       │          │  Execution   │    ┌─────────────────┐               │
│       └──────────│    Loss      │<───│ Expected Output  │               │
│     gradients    │  (dense MSE) │    │  (test cases)    │               │
│                  └──────────────┘    └─────────────────┘               │
│                                                                         │
│  Three gradient paths:                                                  │
│    Path A: Loss → Engine → Parser → LM token logits → coprocessor     │
│    Path B: Loss → Engine → SoftProgram params (program optimization)   │
│    Path C: Loss → Engine → DiffCompiler → source embeddings           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Roles

**Code LM (Qwen2.5/3.5 with nCPU coprocessor)**
- Generates code tokens given a task prompt
- Coprocessor layers (NCPUCoprocessorMLP) provide differentiable arithmetic inside the forward pass
- Backbone frozen; only coprocessor params + optional LoRA update

**Code-to-ISA Parser**
- Extracts arithmetic/logic subexpressions from generated Python code
- Maps them to nCPU's 14-opcode ISA (MOV_IMM, ADD, SUB, MUL, AND, OR, XOR, CMP, BEQ, BNE, BGT, HALT)
- Produces either FixedProgram (hard, gradient through immediates only) or SoftProgram (full gradient)
- Also handles variable tracking: Python variables → nCPU registers

**Differentiable Engine**
- DifferentiableEngine.execute_soft() runs the parsed program
- Soft PC, Gumbel-softmax opcodes, soft register addressing
- Full gradient flow through every operation
- execute_soft_batched() for efficient batch training

**Execution Loss**
- MSE between engine output registers and expected values from test cases
- Per-operation auxiliary losses (did the ADD produce the right sum?)
- Optional: loss on intermediate register states (trace-level supervision)

### 2.3 Three Training Modes

**Mode 1: Coprocessor + Execution Loss (Simplest)**
- LM generates code as usual (teacher-forced or sampled)
- Parse arithmetic subexpressions from generated tokens
- Execute on differentiable engine
- Execution MSE added to standard LM loss
- Gradients update coprocessor params

**Mode 2: Differentiable Compilation (End-to-End)**
- Use DifferentiableCompiler to map LM hidden states directly to SoftPrograms
- No text parsing step — hidden states → program → execution → loss
- Full gradient from execution through compilation into LM hidden states
- Most ambitious; requires the DifferentiableCompilationPipeline

**Mode 3: Program Optimization Feedback (Inference-Time)**
- At inference, LM generates a program
- ProgramOptimizer backprops through execution to find optimal constants
- Optimized constants fed back as context for the LM's next generation
- Hybrid: dense execution gradient at inference, standard training otherwise

---

## 3. Code-to-ISA Parser Design

### 3.1 Supported Python Patterns

| Python Pattern | nCPU ISA Translation |
|---|---|
| `x = 5` | `MOV_IMM R_x, 5` |
| `z = x + y` | `ADD R_z, R_x, R_y` |
| `z = x - y` | `SUB R_z, R_x, R_y` |
| `z = x * y` | `MUL R_z, R_x, R_y` |
| `z = x & y` | `AND R_z, R_x, R_y` |
| `z = x \| y` | `OR R_z, R_x, R_y` |
| `z = x ^ y` | `XOR R_z, R_x, R_y` |
| `if x == y:` | `CMP R_x, R_y` + `BEQ target` |
| `if x > y:` | `CMP R_x, R_y` + `BGT target` |
| `for i in range(n): total += x` | Loop unroll or bounded iteration |
| `return z` | Register R_z holds the result + `HALT` |

### 3.2 Variable-to-Register Mapping

- nCPU has 8 registers (R0–R7)
- Python variables mapped in order of first assignment
- Function arguments get R0, R1, ... (matching calling convention)
- Return value expected in R0 by default
- If > 8 variables, spill to register reuse with lifetime analysis

### 3.3 What We DON'T Parse

- String operations (no nCPU equivalent)
- Object/class operations
- System calls, I/O
- Recursion (requires stack, which DifferentiableEngine doesn't model)
- Floating-point (could use NeuralFloatALU but adds complexity)

The parser focuses on **integer arithmetic and logic** — the operations where the differentiable engine is 100% accurate and the gradient signal is most meaningful.

### 3.4 Soft vs Hard Parsing

**Hard parse (FixedProgram):** AST → deterministic instruction sequence. Gradients flow through immediates only. Simpler, faster, good for Mode 1.

**Soft parse (SoftProgram):** AST → initial opcode logits, register logits. The program structure is a differentiable approximation that gradient descent can adjust. Required for Mode 2.

---

## 4. Execution Loss Design

### 4.1 Primary Loss: Output MSE

```python
def execution_loss(engine_result: ExecutionResult, expected: dict[int, float]) -> Tensor:
    loss = 0.0
    for reg_idx, expected_val in expected.items():
        actual = engine_result.registers[reg_idx]
        loss += (actual - expected_val) ** 2
    return loss / len(expected)
```

### 4.2 Trace-Level Loss (Optional, Stronger Signal)

If we have intermediate expected states (e.g., "after line 3, x should be 15"):

```python
def trace_loss(engine_result: ExecutionResult, expected_trace: list[dict]) -> Tensor:
    loss = 0.0
    for step, expected_regs in expected_trace:
        if step < len(engine_result.register_trace):
            actual = engine_result.register_trace[step]
            for reg_idx, val in expected_regs.items():
                loss += (actual[reg_idx] - val) ** 2
    return loss
```

### 4.3 Correctness-Weighted Loss

Not all test cases are equally informative. Weight by difficulty:

```python
def weighted_execution_loss(results, expected_list, weights):
    total = 0.0
    for result, expected, w in zip(results, expected_list, weights):
        total += w * execution_loss(result, expected)
    return total / sum(weights)
```

### 4.4 Combined Training Loss

```
L_total = L_lm + α * L_execution + β * L_aux_copro + γ * L_trace
```

- `L_lm`: standard next-token prediction loss (teacher-forced)
- `L_execution`: MSE from differentiable execution on test cases
- `L_aux_copro`: load-balancing loss from coprocessor router
- `L_trace`: optional intermediate-state supervision
- α, β, γ: hyperparameters (α=1.0, β=1.0, γ=0.1 as starting points)

---

## 5. Training Data

### 5.1 Arithmetic Code Problems

Simple Python functions with known input → output:

```python
{
    "prompt": "Write a function that returns x * y + z",
    "test_cases": [
        {"inputs": {"x": 3, "y": 7, "z": 2}, "output": 23},
        {"inputs": {"x": 5, "y": 4, "z": 1}, "output": 21},
    ],
    "isa_translation": "MOV R0, #x\nMOV R1, #y\nMOV R2, #z\nMUL R3, R0, R1\nADD R0, R3, R2\nHALT"
}
```

### 5.2 Variable Tracking Problems

```python
{
    "prompt": "x = 5\nx = x + 3\ny = x * 2\nWhat is y?",
    "expected_trace": [
        (0, {0: 5}),      # After MOV R0, 5
        (1, {0: 8}),      # After ADD R0, R0, 3
        (2, {1: 16}),     # After MUL R1, R0, 2
    ],
    "final_output": {"R1": 16}
}
```

### 5.3 Loop Problems (Bounded)

```python
{
    "prompt": "Sum of integers from 1 to n",
    "test_cases": [
        {"inputs": {"n": 5}, "output": 15},
        {"inputs": {"n": 10}, "output": 55},
    ]
}
```

### 5.4 Data Sources

- **Synthetic generator:** parametric arithmetic/logic functions (10K–100K samples)
- **GSM8K extraction:** math word problems with arithmetic steps
- **HumanEval subset:** problems with pure arithmetic solutions
- **CodeContests arithmetic subset:** competitive programming with integer math
- **Existing CodeArithmeticGenerator:** already in `ncpu/coprocessor/code_arithmetic_data.py`

---

## 6. Implementation Plan

### Phase 1: Code-to-ISA Parser (`ncpu/execution_training/code_parser.py`)
- Python AST → nCPU instruction sequence
- Variable-to-register allocation
- Support for assignment, arithmetic, comparison, simple conditionals
- Unit tests with known programs

### Phase 2: Execution Loss Module (`ncpu/execution_training/execution_loss.py`)
- DifferentiableEngine wrapper that takes parsed programs + test cases
- Returns scalar loss with full gradient graph
- Batch support via execute_soft_batched
- Handles parse failures gracefully (fallback to LM-only loss)

### Phase 3: Training Loop (`ncpu/execution_training/train.py`)
- Extends existing coprocessor training harness
- Adds execution loss to the standard LM loss
- Supports all three training modes
- Periodic evaluation on held-out test cases
- Saves coprocessor weights + training metrics

### Phase 4: Evaluation (`ncpu/execution_training/evaluate.py`)
- Arithmetic accuracy (like existing evaluate_arithmetic_accuracy)
- Code correctness (generate → execute → check)
- Execution-grounded accuracy (parse → differentiable execute → check)
- Comparison: baseline vs coprocessor vs coprocessor+execution-training

### Phase 5: Integration + Paper
- End-to-end tests
- Scaling sweep across model sizes
- Paper section / standalone report
- Update README with new training mode

---

## 7. Expected Outcomes

### 7.1 What Should Improve

- **Arithmetic accuracy in generated code:** The model gets direct gradient signal when its code computes wrong answers, instead of just "wrong output."
- **Variable tracking:** Execution traces teach the model that `x = x + 3` means R0 changes from 5 to 8 — the gradient encodes this transition.
- **Constant selection:** ProgramOptimizer-style gradient flow helps the model choose the right constants (e.g., loop bounds, offsets).

### 7.2 What Might Not Change

- **Code style / structure:** Execution loss doesn't care about code readability.
- **Non-arithmetic bugs:** Type errors, API misuse, etc. are outside nCPU's scope.
- **Long programs:** DifferentiableEngine max_steps limits execution length.

### 7.3 Quantitative Targets

| Metric | Baseline (copro only) | Target (copro + exec training) |
|---|---|---|
| Synthetic arithmetic accuracy | ~65% (Qwen3.5-2B) | 85%+ |
| Code arithmetic accuracy | ~50% | 70%+ |
| HumanEval+ pass@1 | baseline | +2-5% (arithmetic subset) |
| Variable tracking accuracy | ~40% | 65%+ |

### 7.4 The Unique Contribution

**No other system provides dense execution gradients to a language model.** CodeRL uses REINFORCE (sparse). Toolformer uses tool output as context (no gradient). Self-play uses success/failure (binary). We use MSE through every operation (dense, directional, per-instruction).

---

## 8. Relationship to Existing nCPU Modules

| Existing Module | Role in This System |
|---|---|
| `ncpu/differentiable/execution.py` | Core engine: DifferentiableEngine, SoftProgram, FixedProgram |
| `ncpu/differentiable/program_optimizer.py` | Mode 3: inference-time constant optimization |
| `ncpu/differentiable/diff_compiler.py` | Mode 2: end-to-end differentiable compilation |
| `ncpu/coprocessor/inject.py` | Injects coprocessor into transformer MLP layers |
| `ncpu/coprocessor/train.py` | Base training harness we extend |
| `ncpu/coprocessor/soft_alu.py` | Differentiable ALU used by both engine and coprocessor |
| `ncpu/coprocessor/code_arithmetic_data.py` | Existing code-pattern dataset generator |
| `ncpu/self_optimizing/task_local_fast_weights.py` | Future: fast-weight adaptation during execution-grounded inference |

---

## 9. Research Context

### 9.1 Related Work

- **CodeRL (Le et al., 2022):** REINFORCE on code generation with unit test feedback. Sparse signal.
- **RLTF (Liu et al., 2023):** Fine-grained RL with line-level feedback. Closer to dense, but still discrete.
- **Toolformer (Schick et al., 2023):** Self-supervised tool use. No execution gradient.
- **AlphaCode (Li et al., 2022):** Massive sampling + filtering. No gradient from execution.
- **Self-Debugging (Chen et al., 2023):** LM explains and fixes its own code. Textual, not gradient-based.
- **Looped Transformers as Programmable Computers (Giannou et al., 2023):** Repeated transformer passes implement computation. Relevant to our differentiable execution.

### 9.2 What's New Here

1. **Differentiable execution as a loss function** — not just a verifier, but a gradient source
2. **Per-operation gradient signal** — not per-program, not per-line, but per-ALU-operation
3. **Coprocessor integration** — the same neural ALU that provides training gradients also augments the model at inference time
4. **Three-mode training** — from simple execution loss augmentation to full end-to-end differentiable compilation

---

## 10. File Layout

```
ncpu/execution_training/
├── __init__.py
├── code_parser.py          # Python AST → nCPU ISA
├── execution_loss.py       # Differentiable execution loss computation
├── train.py                # Training loop (extends coprocessor trainer)
├── evaluate.py             # Evaluation harness
├── data.py                 # Training data generation + loading
└── README.md               # Quick-start guide

docs/
└── DIFFERENTIABLE_EXECUTION_TRAINING.md  # This document
```
