# Differentiable Execution as a Training Signal for Code Generation

## Abstract

We present a method for training code generation models using **dense execution gradients** from a differentiable CPU. Instead of the standard binary pass/fail reward from external test execution, our approach parses generated Python code into a 14-opcode differentiable ISA, executes it on a fully differentiable engine with soft register addressing and Gumbel-softmax opcode selection, and backpropagates the execution error through every ALU operation directly into the model's weights. This provides per-operation gradient signal — the model learns not just "this program was wrong" but "this specific arithmetic operation computed the wrong value, and here's the direction to fix it."

The system builds on nCPU's proven differentiable infrastructure: a neural ALU with 100% integer arithmetic accuracy (exhaustively verified), a differentiable execution engine with full gradient flow through control flow, and a coprocessor injection mechanism that adds ~113K trainable parameters to any HuggingFace transformer.

## 1. Introduction

Training code generation models currently relies on one of three paradigms:

**REINFORCE / rejection sampling** generates code, executes it against test cases, and uses the binary pass/fail as a reward signal. The gradient is sparse: one scalar for an entire program. CodeRL [1], RLTF [2], and AlphaCode [3] all operate in this regime.

**Tool-augmented generation** lets the model call an external interpreter during inference (Toolformer [4], code interpreters in GPT-4), but no gradient flows through the tool. The model treats execution as a black box.

**Self-debugging** has the model explain and fix its own code textually [5], with no execution-derived gradient signal.

All three approaches share a fundamental limitation: **execution is opaque to the training process**. The model never receives dense, directional feedback about *which* operations within its code are wrong and *how* to fix them.

We propose a fourth paradigm: **differentiable execution as a training signal**. By parsing generated code into a differentiable ISA and executing it on a fully differentiable CPU, we produce MSE losses at every register, at every instruction step, with gradients that flow through every ALU operation — all the way back into the model's weights.

## 2. Method

### 2.1 Code-to-ISA Parser

The parser converts Python code (via AST analysis) into nCPU's 14-opcode differentiable ISA:

| Opcode | Python Pattern | Description |
|--------|---------------|-------------|
| MOV_IMM | `x = 5` | Load immediate to register |
| MOV_REG | `y = x` | Register-to-register move |
| ADD | `z = x + y` | Integer addition |
| SUB | `z = x - y` | Integer subtraction |
| MUL | `z = x * y` | Integer multiplication |
| AND | `z = x & y` | Bitwise AND |
| OR | `z = x \| y` | Bitwise OR |
| XOR | `z = x ^ y` | Bitwise XOR |
| CMP | `if x > y:` | Compare (sets flags) |
| BEQ/BNE/BGT | conditional | Branch on flags |
| HALT | return/end | Stop execution |

The parser handles assignments, binary arithmetic, augmented assignments (`+=`, `*=`), comparisons, `if`/`else`, bounded `for` loops (unrolled), and `while` loops. Python variables are mapped to 8 nCPU registers (R0–R7) with a temp register pool for subexpressions.

Two program representations are produced:

- **FixedProgram**: hard instruction encoding, gradients flow only through differentiable immediate values
- **SoftProgram**: Gumbel-softmax over opcodes, softmax attention over registers — full gradient flow through the program structure itself

### 2.2 Differentiable Execution

The DifferentiableEngine executes programs with three key mechanisms for gradient flow:

1. **Soft PC (program counter)**: a probability distribution over instruction positions, advanced via softmax blending rather than hard incrementing
2. **Soft register addressing**: weighted reads/writes across all registers via softmax attention weights
3. **Parallel ALU evaluation**: all operations computed simultaneously, weighted by opcode probabilities

This means every execution step is a differentiable function of the program parameters, register state, and flags. The entire execution trace exists in one PyTorch computation graph.

### 2.3 Execution Loss

The loss has three components:

**Output loss (L_output)**: MSE between the engine's final register state and the expected values from test cases.

$$L_{\text{output}} = \frac{1}{|T|} \sum_{(r, v) \in T} (R_r - v)^2$$

where T is the set of (register, expected_value) targets.

**Trace loss (L_trace)**: MSE on intermediate register states at specific instruction steps. This provides per-instruction supervision.

$$L_{\text{trace}} = \frac{1}{|S|} \sum_{(t, r, v) \in S} (R_r^{(t)} - v)^2$$

where S is the set of (step, register, expected_value) trace targets.

**Structure loss (L_structure)**: penalties for non-halting programs and excessive step counts.

The combined training loss:

$$L = L_{\text{LM}} + \alpha L_{\text{output}} + \beta L_{\text{aux}} + \gamma L_{\text{trace}}$$

where L_LM is the standard next-token prediction loss and L_aux is the coprocessor load-balancing loss from the router.

### 2.4 Three Training Modes

**Mode 1: Coprocessor + Execution Loss.** The simplest mode. Reference code from the training data is parsed and executed. The execution loss is added to the standard LM loss. The nCPU coprocessor layers (injected into transformer MLP sublayers) receive execution-grounded gradients alongside the standard language modeling gradients.

**Mode 2: Differentiable Compilation.** A compilation bridge projects LM hidden states into the DifferentiableCompiler's embedding space via a learned linear projection. The compiler produces a SoftProgram, which is executed. Gradients flow end-to-end: execution → compilation → LM hidden states.

**Mode 3: Generated Code Training.** The model generates code (non-differentiable sampling), which is parsed and executed. The execution loss serves as a reward signal for REINFORCE policy gradient on the generation logits, with an adaptive EMA baseline for variance reduction. The coprocessor still receives direct execution gradients via the SoftProgram path.

### 2.5 Trace-Level Supervision

The TraceGenerator simulates Python code line-by-line, recording variable state after each statement, then maps these states to expected nCPU register values at each instruction. This provides the maximum gradient signal: instead of one loss at the end of execution, every instruction has a target.

## 3. Implementation

The system is implemented as `ncpu/execution_training/` with 8 modules:

| Module | Lines | Purpose |
|--------|-------|---------|
| `code_parser.py` | ~600 | Python AST → nCPU ISA with temp register pooling |
| `execution_loss.py` | ~400 | Differentiable loss with NaN/Inf clamping |
| `data.py` | ~450 | Three generators: arithmetic, variable tracking, loops |
| `trace_data.py` | ~350 | Per-instruction trace generation via Python simulation |
| `compilation_bridge.py` | ~500 | LM hidden states → DiffCompiler → execution → loss |
| `generated_code_training.py` | ~600 | REINFORCE + execution loss on generated code |
| `evaluate.py` | ~400 | Multi-dimensional evaluation harness |
| `train.py` | ~600 | Training loop with combined loss, CLI |

96 tests verify the full pipeline: parser correctness, gradient flow through soft programs, loss computation, data generation quality, trace supervision, compilation bridge, REINFORCE baseline, and end-to-end optimization convergence.

### 3.1 Key Design Decisions

**Temp register pooling.** The parser allocates temporary registers for subexpressions and releases them after use, preventing register exhaustion on complex expressions. This achieves 100% parse rate on generated training data.

**NaN/Inf clamping.** Soft execution can produce overflow on programs with multiplication chains. Register values are clamped to [-1e6, 1e6] in the engine, and losses are clamped to 1e6 with NaN replacement.

**SoftProgram initialization from parsed code.** When converting parsed code to a SoftProgram, the opcode and register logits are initialized with a strong bias (5.0) toward the parsed instructions. This gives gradient descent a good starting point while preserving full differentiability.

## 4. Unique Contribution

No other system provides dense execution gradients to a language model:

| Approach | Signal Type | Gradient Source |
|----------|------------|----------------|
| CodeRL [1] | Sparse (pass/fail) | REINFORCE estimator |
| RLTF [2] | Line-level | Fine-grained RL |
| AlphaCode [3] | Sparse (filtering) | No execution gradient |
| Toolformer [4] | Context (tool output) | No gradient through tool |
| Self-Debugging [5] | Textual | No execution gradient |
| **This work** | **Dense, per-operation** | **MSE through differentiable CPU** |

The combination of (1) a proven differentiable CPU with 100% arithmetic accuracy, (2) a code parser that handles real Python patterns, and (3) a coprocessor that injects differentiable arithmetic into transformer forward passes, creates a system where execution-derived gradients directly update the model's computation pathway.

## References

[1] Le et al. "CodeRL: Mastering Code Generation through Pretrained Models and Deep Reinforcement Learning." NeurIPS 2022.

[2] Liu et al. "RLTF: Reinforcement Learning from Unit Test Feedback." arXiv 2023.

[3] Li et al. "Competition-Level Code Generation with AlphaCode." Science 2022.

[4] Schick et al. "Toolformer: Language Models Can Teach Themselves to Use Tools." NeurIPS 2023.

[5] Chen et al. "Teaching Large Language Models to Self-Debug." arXiv 2023.

[6] Giannou et al. "Looped Transformers as Programmable Computers." ICML 2023.
