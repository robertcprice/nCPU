# Differentiable Execution Training

Train code generation models with **dense execution gradients** instead of sparse pass/fail rewards.

## What This Does

Standard code model training: model generates code → execute externally → binary pass/fail → sparse gradient.

This module: model generates code → **parse to nCPU ISA** → **execute on differentiable CPU** → **MSE loss on every register** → **dense gradients flow back through every ALU operation**.

The model learns not just "this program was wrong" but "the addition on line 5 should have been subtraction."

## Quick Start

```bash
# Smoke test (no LM required, tests the full parse → execute → gradient pipeline)
python -m ncpu.execution_training.train --synthetic-only --steps 200

# Full training with a real model
python -m ncpu.execution_training.train \
    --model Qwen/Qwen3.5-0.8B \
    --steps 2000 \
    --exec-loss-weight 1.0 \
    --output-dir training_results/exec_training/

# Run tests
python -m pytest tests/test_execution_training.py -v
```

## Architecture

```
Code LM (frozen + coprocessor) → Code-to-ISA Parser → DifferentiableEngine → Execution Loss
         ↑                                                                          |
         └──── gradients flow ALL the way back ────────────────────────────────────┘
```

### Components

| Module | Role |
|--------|------|
| `code_parser.py` | Python AST → nCPU 14-opcode ISA (assignments, arithmetic, loops, conditionals) |
| `execution_loss.py` | Differentiable execution loss with full gradient flow |
| `data.py` | Training data generators (arithmetic, variable tracking, loop problems) |
| `evaluate.py` | Evaluation harness measuring parse rate, exec accuracy, loss |
| `train.py` | Training loop combining LM loss + execution loss + coprocessor aux loss |

### Supported Python Patterns

| Python | nCPU Translation |
|--------|-----------------|
| `x = 5` | `MOV_IMM R_x, 5` |
| `z = x + y` | `ADD R_z, R_x, R_y` |
| `z = x * y` | `MUL R_z, R_x, R_y` |
| `z = x & y` | `AND R_z, R_x, R_y` |
| `if x > y:` | `CMP R_x, R_y` + `BGT target` |
| `for i in range(N):` | Loop unroll (bounded) |
| `x += 3` | `ADD R_x, R_x, R_temp` |

### Training Loss

```
L_total = L_lm + α * L_execution + β * L_aux_copro + γ * L_trace
```

- `L_lm`: standard next-token prediction
- `L_execution`: MSE from differentiable execution on test cases
- `L_aux_copro`: coprocessor load-balancing (from nCPU router)
- `L_trace`: optional intermediate register state supervision

### Three Modes

1. **Coprocessor + Execution Loss** (default): Parse reference code, execute differentiably, train coprocessor
2. **Differentiable Compilation**: Map LM hidden states directly to programs (end-to-end)
3. **Program Optimization Feedback**: Inference-time constant tuning via ProgramOptimizer

## Test Results

33/33 tests passing:
- Parser: assignment, arithmetic, multiplication, subtraction, complex expressions, augmented assignment, multi-step, function parse, loop unroll, bitwise ops, if-statements, assembly output, program conversion
- Execution loss: correct programs, wrong programs, soft gradient flow, batched loss, parsing integration, fallback handling
- Data: all three generators, combined dataset, parseability validation
- Evaluation: reference evaluation, summary formatting
- Gradient flow: end-to-end soft program gradients, gradient direction verification, optimization convergence

## Relation to Existing nCPU Modules

This builds on:
- `ncpu/differentiable/` — DifferentiableEngine, SoftProgram, FixedProgram
- `ncpu/coprocessor/` — NCPUCoprocessorMLP injection, router, soft ALU
- `ncpu/coprocessor/train.py` — base training harness pattern

See `docs/DIFFERENTIABLE_EXECUTION_TRAINING.md` for the full architecture document.
