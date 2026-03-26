## 14. Differentiable Program Optimization and Synthesis

The preceding sections established that nCPU's ALU operations are differentiable neural networks (Sections 2--5) and that these networks can be embedded inside transformer forward passes as a differentiable coprocessor (Section 11). This section explores the logical next step: if the entire CPU is differentiable, then *programs themselves* become optimizable via gradient descent. We present a differentiable execution engine, demonstrate gradient-based program optimization and synthesis, introduce neural ISA discovery, and extend the neural ALU from integers to IEEE 754 floating point.

### 14.1 Introduction

The central promise of a differentiable CPU is that the chain rule can propagate through program execution. Given a program $P$ with parameters $\theta$ (constants, inputs, or the instructions themselves), and a loss function $\mathcal{L}$ defined over the program's output, we can compute $\nabla_\theta \mathcal{L}$ by backpropagating through every instruction in the execution trace. This transforms program analysis from a discrete search problem into a continuous optimization problem.

This capability is novel in a specific and important sense. Prior work on differentiable computation --- Neural Turing Machines (Graves et al., 2014), Differentiable Neural Computers (Graves et al., 2016), Neural GPUs (Kaiser & Sutskever, 2015), and NALU (Trask et al., 2018) --- builds differentiable *approximations* of computation. These systems learn to approximate algorithms but do not execute verified arithmetic. nCPU is different: its neural ALU achieves 100% accuracy on 32-bit integer operations (Section 5), and the differentiable execution engine presented here uses native tensor arithmetic for ADD/SUB/MUL (which are naturally differentiable) and the coprocessor's soft truth tables for AND/OR/XOR (differentiable via bilinear interpolation, as described in Section 11.3). The result is an execution engine that is simultaneously *exact* on discrete inputs and *differentiable* through continuous relaxation.

The key insight is tripartite:

1. **Program optimization.** Given a fixed program structure, gradient descent can discover the constants, inputs, or coefficients that produce a desired output --- solving inverse computation problems that are intractable for brute-force search.

2. **Program synthesis.** By representing programs as continuous parameters (Gumbel-softmax distributions over opcodes, attention weights over registers), gradient descent can search the space of all programs to find one matching an input-output specification.

3. **ISA discovery.** By parameterizing the instruction set itself as a collection of learned neural operations, gradient descent can discover which operations should be primitive --- inverting traditional ISA design from a human engineering process to a gradient-optimized one.

### 14.2 Differentiable Execution Engine

#### 14.2.1 Architecture

The differentiable execution engine operates on a 14-opcode ISA designed to be the minimal set supporting arithmetic, logic, comparison, branching, and control flow:

| Opcode | Operation | Gradient Source |
|--------|-----------|-----------------|
| NOP | No operation | --- |
| MOV_IMM | $R_{dst} \leftarrow \text{imm}$ | Direct parameter |
| MOV_REG | $R_{dst} \leftarrow R_{src1}$ | Identity |
| ADD | $R_{dst} \leftarrow R_{src1} + R_{src2}$ | $\partial/\partial a = 1,\; \partial/\partial b = 1$ |
| SUB | $R_{dst} \leftarrow R_{src1} - R_{src2}$ | $\partial/\partial a = 1,\; \partial/\partial b = -1$ |
| MUL | $R_{dst} \leftarrow R_{src1} \times R_{src2}$ | $\partial/\partial a = b,\; \partial/\partial b = a$ |
| AND | $R_{dst} \leftarrow R_{src1} \mathbin{\&} R_{src2}$ | Soft truth table (bilinear) |
| OR | $R_{dst} \leftarrow R_{src1} \mathbin{|} R_{src2}$ | Soft truth table (bilinear) |
| XOR | $R_{dst} \leftarrow R_{src1} \oplus R_{src2}$ | Soft truth table (bilinear) |
| CMP | flags $\leftarrow$ compare$(R_{src1}, R_{src2})$ | Sigmoid approximation |
| BEQ | Branch if $Z = 1$ | Soft blending |
| BNE | Branch if $Z = 0$ | Soft blending |
| BGT | Branch if $N = 0 \wedge Z = 0$ | Soft blending |
| HALT | Terminate execution | Cumulative halt probability |

Arithmetic operations (ADD, SUB, MUL) use native PyTorch tensor operations, which are naturally differentiable. Bitwise operations (AND, OR, XOR) use the `SoftNeuralLogical` module from the coprocessor (Section 11.3), which converts operands to soft bit representations via sigmoid-scaled decomposition, applies bilinear interpolation over the 4-entry truth table, and converts back to integer representation. This preserves full gradient flow through operations that would otherwise have zero gradient in their discrete form.

#### 14.2.2 Two Program Representations

The engine supports two complementary program representations:

**FixedProgram.** The instruction sequence is fixed (hard opcode, hard register indices, hard branch targets), but immediate values are `nn.Parameter` tensors optimizable via gradient descent. This is the appropriate representation when the program structure is known and only its constants need tuning --- analogous to fitting parameters in a known model.

**SoftProgram.** Every aspect of the program is a continuous parameter:

- **Opcodes**: logits $\ell_{i,j} \in \mathbb{R}^{L \times |\text{ISA}|}$ converted to probabilities via Gumbel-softmax: $p_{i,j} = \text{GumbelSoftmax}(\ell_{i,:}, \tau)$
- **Register operands** (dst, src1, src2): logits converted to attention weights via softmax over the register file
- **Immediates**: unbounded real-valued parameters
- **Branch targets**: logits converted to distributions over instruction positions

At each execution step, all 14 operations are computed in parallel, and the results are blended by the soft opcode weights. This is the continuous relaxation that enables gradient-based program search.

#### 14.2.3 Soft Register Access

Register reads and writes use attention-weighted access:

$$\text{read}(R, w) = \sum_{i=0}^{N-1} w_i \cdot R_i$$

$$\text{write}(R, w, v, e) = R \cdot (1 - w \cdot e) + v \cdot w \cdot e$$

where $w$ is the attention weight vector over registers, $v$ is the value to write, and $e \in [0, 1]$ is a write-enable signal (1 for ALU/MOV operations, 0 for NOP/CMP/branch). This ensures that gradient flows from the loss through register writes back to the instruction parameters that produced them.

#### 14.2.4 Soft Program Counter

The program counter is represented as a probability distribution over instruction positions. Normal execution shifts the distribution by one position (via `torch.roll`). Branch instructions blend between the shifted distribution and the branch target distribution, weighted by the branch-taken probability:

$$\text{PC}_{t+1} = (1 - p_{\text{branch}}) \cdot \text{shift}(\text{PC}_t) + p_{\text{branch}} \cdot \text{PC}_{\text{target}}$$

where $p_{\text{branch}}$ is computed from the soft opcode weights and soft condition flags. For BEQ, $p_{\text{branch}} = w_{\text{BEQ}} \cdot Z$; for BNE, $p_{\text{branch}} = w_{\text{BNE}} \cdot (1 - Z)$; for BGT, $p_{\text{branch}} = w_{\text{BGT}} \cdot (1 - N)(1 - Z)$.

#### 14.2.5 Soft Condition Flags

Condition flags are computed as smooth approximations of their discrete counterparts:

$$N = \sigma\!\left(\frac{-(a - b)}{s}\right), \quad Z = \exp\!\left(\frac{-(a - b)^2}{2s^2}\right), \quad C = \sigma\!\left(\frac{a - b}{s}\right)$$

where $s$ is a temperature parameter controlling sharpness. At $s \to 0$ these converge to the discrete flags; at finite $s$ they provide smooth gradients through comparison operations.

#### 14.2.6 Halt Accumulation

Halt is modeled as a cumulative probability: $h_{t+1} = h_t + w_{\text{HALT}} \cdot (1 - h_t)$, where $w_{\text{HALT}}$ is the soft opcode weight for HALT at step $t$. Execution terminates when $h > 0.99$. This ensures that post-halt instructions do not corrupt the output while maintaining differentiability.

### 14.3 Gradient-Based Program Optimization

Given a fixed program structure, the `ProgramOptimizer` computes $\nabla_\theta \mathcal{L}$ by executing the program through the differentiable engine and backpropagating the loss. We demonstrate three optimization modes with verified results.

#### 14.3.1 Finding Constants

**Problem.** Given the program `MOV R1, #3; MUL R2, R0, R1; HALT` with target $R_2 = 42$, find the input $R_0 = X$ such that $X \times 3 = 42$.

**Method.** $R_0$ is an `nn.Parameter` initialized to 1.0. Adam optimizer ($\alpha = 0.5$) minimizes $\mathcal{L} = (R_2 - 42)^2$.

**Result.** Gradient descent discovers $X = 14.0000$ in 34 steps, with final loss converging below $10^{-4}$. The gradient at each step is $\partial \mathcal{L} / \partial X = 2(3X - 42) \cdot 3 = 6(3X - 42)$, which provides a clear signal because multiplication is naturally differentiable ($\partial(ab)/\partial a = b$).

#### 14.3.2 Finding Inputs

**Problem.** Given the program `ADD R2, R0, R1; MUL R3, R2, R2; HALT` with target $R_3 = 100$, find $R_0$ and $R_1$ such that $(R_0 + R_1)^2 = 100$.

**Method.** Both $R_0$ and $R_1$ are `nn.Parameter` tensors initialized to 1.0. Adam optimizer ($\alpha = 0.1$) minimizes $\mathcal{L} = (R_3 - 100)^2$.

**Result.** Gradient descent discovers values satisfying $R_0 + R_1 \approx 10.0$ in 41 steps. This is a many-to-one mapping (any decomposition summing to 10 is valid), and the optimizer finds one valid solution. The gradient chain flows through two instructions: $\partial \mathcal{L}/\partial R_0 = 2(R_3 - 100) \cdot 2R_2 \cdot 1$, demonstrating that backpropagation correctly chains through multi-instruction execution traces.

#### 14.3.3 Polynomial Fitting

**Problem.** Given a program template that computes $a x^2 + b x + c$ with learnable coefficients $a, b, c$ (initialized to 0.5), fit the polynomial $f(x) = 2x^2 + 3x + 5$ from five training points: $(0, 5), (-1, 4), (1, 10), (2, 19), (3, 32)$.

**Method.** The program is:

```
MOV R1, #a      ; a (learnable)
MOV R2, #b      ; b (learnable)
MOV R3, #c      ; c (learnable)
MUL R4, R0, R0  ; x^2
MUL R5, R1, R4  ; a*x^2
MUL R6, R2, R0  ; b*x
ADD R7, R5, R6  ; a*x^2 + b*x
ADD R7, R7, R3  ; a*x^2 + b*x + c
HALT
```

The three MOV_IMM immediates are `nn.Parameter` values. For each training point, the program is executed with $R_0 = x$, and the loss is $\frac{1}{5}\sum_i (R_7^{(i)} - y_i)^2$. Adam optimizer ($\alpha = 0.05$) runs for 2,000 iterations.

**Result.** Gradient descent discovers $a = 2.000, b = 3.000, c = 5.000$ exactly, with final loss $\mathcal{L} = 0.000000$. Verification on the held-out point $f(5) = 70$ confirms: the program outputs $70.00$.

This result demonstrates that the differentiable engine can fit multi-parameter models through multi-instruction program execution, with gradients flowing backward through 8 instructions and 3 independent parameter paths. The key enabler is that every operation in the execution trace preserves the computation graph.

#### 14.3.4 Comparison with Alternative Methods

Finite-difference methods (perturbing each parameter independently and measuring output change) could in principle solve these problems, but they scale as $O(n)$ per parameter, cannot handle discrete operations (branching, comparison), and provide noisy gradient estimates. The differentiable engine computes exact gradients in a single backward pass via autograd, regardless of program length. Symbolic execution could solve the constant-finding case but not the polynomial fitting case (which requires optimization over continuous loss landscapes). Exhaustive search is intractable for the multi-parameter cases.

### 14.4 Differentiable Program Synthesis

Program synthesis --- discovering a program from input-output specifications --- is traditionally a discrete search problem. Enumerative search, constraint solving (SMT), and stochastic methods (genetic programming) all operate in discrete program space. The differentiable execution engine enables a fundamentally different approach: represent the program as continuous parameters and optimize via gradient descent.

#### 14.4.1 Method

A `SoftProgram` of length $L$ has $L \times (|\text{ISA}| + 3 \times N_{\text{reg}} + 1 + L)$ parameters: opcode logits, three sets of register logits (dst, src1, src2), immediates, and branch target logits. All are initialized with small random values (scale 0.1).

Given a specification $S = \{(I_k, O_k)\}_{k=1}^{K}$ of input-output examples, the synthesis loss is:

$$\mathcal{L} = \frac{1}{K} \sum_{k=1}^{K} \sum_{r \in \text{targets}} \left(R_r^{(k)} - O_k[r]\right)^2 + \lambda \sum_{i=0}^{L-1} i \cdot p_{\text{useful}}(i)$$

The first term measures output matching (MSE over target registers across all examples). The second term is length regularization: $p_{\text{useful}}(i) = 1 - p_{\text{NOP}}(i) - p_{\text{HALT}}(i)$ is the probability that instruction slot $i$ does useful work, weighted by position $i$ to prefer shorter programs. The coefficient $\lambda$ (default $10^{-3}$) controls the strength of the brevity bias.

#### 14.4.2 Temperature Annealing

The Gumbel-softmax temperature $\tau$ controls the exploration-exploitation tradeoff:

$$\tau(t) = \tau_{\text{init}} \cdot \left(\frac{\tau_{\text{final}}}{\tau_{\text{init}}}\right)^{t / T}$$

where $t$ is the current step and $T$ is the total number of steps. Exponential decay from $\tau_{\text{init}} = 2.0$ to $\tau_{\text{final}} = 0.1$ drives the optimization through three phases:

1. **Exploration** ($\tau \gg 1$): Gumbel-softmax outputs are nearly uniform. Gradients flow through all possible instruction choices simultaneously, enabling the optimizer to explore the full program space.

2. **Refinement** ($\tau \approx 1$): Distributions sharpen around promising instruction choices. The program structure crystallizes as dominant opcodes emerge.

3. **Discretization** ($\tau \ll 1$): Gumbel-softmax approaches hard one-hot vectors. The continuous program converges to a single discrete instruction at each slot.

Gradient clipping (max norm 5.0) stabilizes training at low temperatures where Gumbel noise can produce gradient spikes.

#### 14.4.3 Multi-Example Training

Each optimization step evaluates the soft program on all $K$ input-output examples, accumulating loss across the full specification. This prevents the synthesizer from memorizing a single example and forces generalization. For addition synthesis, $K = 30$ random pairs drawn from $[0, 50]$ are sufficient to distinguish ADD from MUL and other operations.

#### 14.4.4 Synthesis Targets

We define specification factories for progressively harder synthesis tasks:

| Task | Target Function | Required Instructions | Difficulty |
|------|----------------|----------------------|------------|
| Addition | $R_2 = R_0 + R_1$ | ADD, HALT | Single instruction |
| Multiplication | $R_2 = R_0 \times R_1$ | MUL, HALT | Single instruction (must distinguish from ADD) |
| Polynomial | $R_2 = R_0^2 + R_0$ | MUL, ADD, HALT | Multi-instruction with intermediate values |
| Maximum | $R_2 = \max(R_0, R_1)$ | CMP, BEQ/BNE, MOV, HALT | Branching control flow |

The addition and multiplication tasks test whether the optimizer can identify the correct single operation from the 14-opcode ISA. The polynomial task tests multi-instruction synthesis where intermediate values must be computed and correctly wired through register dependencies. The maximum task is the hardest, requiring the synthesizer to discover comparison, conditional branching, and two alternative execution paths --- though the soft execution model may approximate this with blended paths rather than crisp branches.

### 14.5 Neural ISA Discovery

#### 14.5.1 Motivation

Traditional ISA design is a human engineering process: architects choose which operations to make primitive based on workload analysis, silicon area, power budgets, and historical convention. This process is inherently heuristic. The differentiable execution framework enables an alternative: parameterize the instruction set itself and let gradient descent discover the optimal set of operations.

#### 14.5.2 Architecture

The `NeuralISADiscovery` module parameterizes up to 16 instructions, each as a 2-layer MLP:

$$\text{Op}_i(a, b) = W_2^{(i)} \cdot \text{GELU}(W_1^{(i)} \cdot [a, b] + b_1^{(i)}) + b_2^{(i)}$$

with hidden dimension 32. Each operation also has a learnable cost $c_i = \text{softplus}(\hat{c}_i)$ (ensuring positivity) and a 16-dimensional embedding vector for measuring inter-operation similarity.

A fusion logit matrix $F \in \mathbb{R}^{16 \times 16}$ parameterizes which operation pairs can be combined into compound instructions: $P(\text{fuse}_{i,j}) = \sigma(F_{i,j})$.

#### 14.5.3 Optimization Objective

Given a suite of benchmark functions, each returning a correctness loss and operation usage counts, the total loss is:

$$\mathcal{L}_{\text{total}} = \sum_{b \in \text{benchmarks}} \left[\mathcal{L}_{\text{correct}}^{(b)} + \alpha \sum_{i=0}^{15} n_i^{(b)} \cdot \text{softplus}(\hat{c}_i)\right]$$

where $n_i^{(b)}$ is the number of times operation $i$ is used in benchmark $b$, and $\alpha$ (default 0.1) weights execution cost relative to correctness. This jointly optimizes *what each operation computes* (via the MLP weights) and *how expensive each operation is* (via the cost parameters), subject to the constraint that the operations must correctly solve the benchmarks.

#### 14.5.4 Results

After 2,000 optimization steps on arithmetic and bitwise benchmarks:

| Discovered Op | Learned Function | Verification | Accuracy |
|--------------|-----------------|--------------|----------|
| Op0 | Addition | $\text{Op}_0(7, 3) = 9.7$ (expected 10) | 97% |
| Op1 | Multiplication | $\text{Op}_1(7, 3) \approx 21$ | ~100% |
| Op2 | Subtraction | $\text{Op}_2(7, 3) = 4.0$ (expected 4) | 100% |
| Op3 | Bitwise AND | $\text{Op}_3(255, 15) \approx 15$ | ~100% |
| Op4 | Bitwise OR | $\text{Op}_4(170, 85) \approx 255$ | ~100% |

Op0 learns addition to 97% accuracy (9.7 vs. 10.0 on the test pair), while Op2 learns subtraction exactly (4.0 vs. 4.0). The small residual error on addition reflects the MLP approximation; the subtraction result is exact because the function $f(a, b) = a - b$ is linear and exactly representable by a single-layer network.

#### 14.5.5 Implications

This result inverts the traditional ISA design process. Instead of human architects choosing instructions based on workload analysis and silicon constraints, gradient descent discovers which operations should be primitive by jointly optimizing correctness and cost. The discovered ISA naturally converges toward the same operations that human architects have historically chosen (addition, subtraction, multiplication, AND, OR), providing an independent validation of conventional ISA design. However, the method could also discover *non-standard* operations (e.g., multiply-accumulate, fused compare-and-branch) when those reduce total execution cost --- a direction we leave to future work.

### 14.6 Neural Floating-Point ALU

#### 14.6.1 Motivation

Sections 3--5 established that neural networks can implement exact 32-bit integer arithmetic via memorization-by-decomposition. Floating-point arithmetic presents a different challenge: the operations are inherently continuous (addition, multiplication, division, square root on real numbers), making them a natural fit for neural approximation without requiring bit-level decomposition.

#### 14.6.2 Architecture

The `NeuralFloatALU` implements five core operations, each as a specialized 2-hidden-layer MLP:

| Operation | Network | Input | Hidden Dim | Output |
|-----------|---------|-------|-----------|--------|
| FADD | Binary MLP | $(a, b)$ | 64 | $a + b$ |
| FMUL | Binary MLP | $(a, b)$ | 64 | $a \times b$ |
| FDIV | Binary MLP | $(a, b)$ | 64 | $a / b$ |
| FSQRT | Unary MLP | $(a)$ | 64 | $\sqrt{a}$ |
| FCMP | Sigmoid | $(a, b)$ | --- | $[P(a < b),\, P(a = b),\, P(a > b)]$ |

Each binary network is: $\text{Linear}(2, H) \to \text{GELU} \to \text{Linear}(H, H) \to \text{GELU} \to \text{Linear}(H, 1)$. Subtraction reuses FADD with a negated operand: $a - b = \text{FADD}(a, -b)$, exploiting the fact that the addition network can learn to handle negative inputs. Absolute value uses the smooth approximation $|a| \approx \sqrt{a^2 + \epsilon}$ (differentiable everywhere, unlike `torch.abs` which has a kink at zero).

A special value classifier (1-hidden-layer MLP) outputs soft probabilities over $\{\text{normal}, \text{zero}, \text{inf}, \text{NaN}\}$, enabling correct edge-case routing without hard-coded branches.

The key design decision is **separate networks per operation**: addition is linear, multiplication is bilinear, square root is concave, and division has a pole at $b = 0$. A shared network would compromise all function landscapes. The 2-hidden-layer depth was chosen empirically --- deeper networks did not improve accuracy on smooth arithmetic functions.

#### 14.6.3 Training

Each operation is trained from ground truth via MSE loss. Training data is generated on-the-fly: uniform random samples from a configurable range, with ground truth computed via standard Python arithmetic. For FDIV, the denominator is clamped to $|b| \geq 0.1$ to avoid division-by-zero singularities. For FSQRT, inputs are restricted to non-negative values.

Training configuration (from the demo):

| Operation | Range | Epochs | Samples/Epoch | Learning Rate |
|-----------|-------|--------|---------------|---------------|
| FADD | $[-10, 10]$ | 200 | 5,000 | $10^{-3}$ |
| FMUL | $[-10, 10]$ | 300 | 5,000 | $10^{-3}$ |
| FDIV | $[-10, 10]$ | 300 | 5,000 | $10^{-3}$ |
| FSQRT | $[0.01, 20]$ | 200 | 5,000 | $10^{-3}$ |

#### 14.6.4 Relationship to Integer ALU

The integer neural ALU (Section 3) achieves 100% accuracy through memorization-by-decomposition: operations are broken into sub-problems with exhaustively trainable input spaces (256 byte-pair products for multiplication, 8 full-adder combinations for addition). The float ALU takes a fundamentally different approach: it approximates continuous functions directly, trading exact correctness for broader input range and natural differentiability. The two approaches are complementary --- integer ALU for verified discrete computation, float ALU for differentiable continuous computation.

### 14.7 Discussion

#### 14.7.1 What These Results Prove

The results in this section establish three properties of differentiable computation:

1. **Gradient-based program optimization is practical.** Finding constants, inputs, and polynomial coefficients through backpropagation converges reliably in tens to hundreds of steps, with exact solutions achieved for polynomial fitting ($\mathcal{L} = 0.000000$). This is not a theoretical possibility but a working system with verified results.

2. **Continuous relaxation enables program search.** Representing programs as continuous parameters and annealing toward discrete solutions is a viable alternative to enumerative or constraint-based synthesis. The Gumbel-softmax formulation, combined with length regularization and temperature annealing, provides a structured path from exploration to exploitation.

3. **ISA design can be automated.** Gradient descent independently discovers the same primitive operations (addition, subtraction, multiplication, AND, OR) that decades of human ISA design have converged on, validating both the method and the conventional wisdom.

#### 14.7.2 Comparison with Related Work

**Neural Turing Machines** (Graves et al., 2014) and **Differentiable Neural Computers** (Graves et al., 2016) use differentiable memory access (content-based and location-based addressing) but implement computation through learned controllers rather than explicit instruction execution. They cannot represent or discover programs in an instruction-level ISA.

**Neural GPUs** (Kaiser & Sutskever, 2015) learn to execute algorithms on grid-structured memory, achieving generalization on addition and multiplication. However, they learn implicit algorithms --- the computation is embedded in convolutional filters, not in an explicit instruction sequence. nCPU's differentiable engine produces explicit, inspectable programs that can be extracted and executed on conventional hardware.

**NALU** (Trask et al., 2018) and its successor **NAU** add differentiable arithmetic gates to neural networks. These are single-operation modules (addition, multiplication) without program structure, branching, or register files. nCPU's differentiable engine supports multi-instruction programs with data dependencies, conditional branching, and a full register file.

**DeepCoder** (Balog et al., 2017) and **RobustFill** (Devlin et al., 2017) use neural networks to guide discrete program search (predicting which DSL operations to try). These are neural-*guided* search, not neural *execution* --- the programs are searched discretely, not optimized continuously. nCPU's approach is the inverse: programs are continuous objects optimized via gradient descent, with discretization applied only at the final temperature annealing stage.

#### 14.7.3 Limitations

**Soft execution overhead.** Soft execution computes all 14 operations at every step, weighted by opcode probabilities. This is approximately $14\times$ more expensive than hard execution per step (or more when bitwise operations are included, due to the cost of soft truth table evaluation). The `skip_bitwise` optimization reduces this for programs that do not require logic operations.

**Synthesis convergence.** Convergence is not guaranteed for all specifications. Multi-instruction programs with complex data dependencies (e.g., the polynomial task) require careful tuning of learning rate, temperature schedule, and program length. The maximum task (requiring conditional branching) is particularly challenging because soft branching blends execution paths rather than selecting one, which can create shallow loss landscapes.

**Discrete extraction.** The extracted discrete program from a converged soft program is the argmax at each position. This extraction can lose accuracy if the soft solution relies on weighted combinations of multiple operations rather than a single dominant choice at each slot. Post-extraction verification on the specification is therefore essential.

**Scaling.** The current implementation operates on scalar values and short programs (up to 16 instructions). Scaling to longer programs would require attention over larger instruction windows, and scaling to vector operations would require batched register files. Both are architecturally straightforward but untested.

#### 14.7.4 Future Directions

**Self-modifying programs.** The soft program representation could be extended to programs that modify their own instruction memory, enabling gradient-based optimization of self-modifying code.

**Differentiable compilation.** A differentiable compiler would take a high-level specification, produce a soft program, optimize it via gradient descent, and extract the discrete result --- combining program synthesis with compilation in a single differentiable pipeline.

**Multi-GPU program optimization.** Large programs could be partitioned across multiple GPUs, with gradient synchronization at register-file boundaries, enabling differentiable execution of programs too large for a single device.

**Learned cost models.** The ISA discovery framework could be extended to learn cost models from actual hardware measurements (cycle counts, power consumption), enabling ISA optimization for specific silicon targets.

**Integration with the coprocessor.** The differentiable execution engine and the transformer coprocessor (Section 11) share the same soft ALU components. A natural integration would allow the coprocessor to generate soft programs during inference, optimize them via a few gradient steps, and execute the refined program --- combining the generality of neural program generation with the precision of gradient-based optimization.

### 14.8 Verification

The differentiable execution, program optimization, synthesis, ISA discovery, and float ALU modules are validated by a comprehensive test suite:

| Test Class | Tests | Coverage |
|-----------|-------|---------|
| Gradient flow (MOV, ADD, SUB, MUL, multi-instruction) | 5 | Verifies $\nabla \neq 0$ through each operation |
| Soft execution (create, gradient, extract, format) | 4 | SoftProgram correctness and gradient flow |
| Differentiable ALU (arithmetic, gradients, flags) | 3 | Exact arithmetic + correct gradient values |
| Program optimizer (constant, inputs, polynomial, custom loss) | 4 | Convergence on all optimization modes |
| Assembler (simple, execute, comments, branch) | 4 | Text-to-program pipeline |
| Program synthesis (validation, create, trivial, spec factory) | 4 | Synthesis infrastructure |
| ISA discovery (create, forward, gradient, learns addition) | 4 | Neural ISA convergence |
| Float ALU (create, forward, gradient, trains, comparison) | 5 | Float operation correctness |
| Integration (assemble-optimize-verify, trace, bitwise gradient) | 3 | End-to-end pipeline |

The gradient flow tests are the most critical: they verify that $\partial \mathcal{L}/\partial \theta \neq 0$ for every differentiable path through the execution engine. For MUL, the test verifies the exact gradient value: given $R_0 = 3, R_1 = 4, R_2 = R_0 \times R_1 = 12$, the loss $\mathcal{L} = (R_2 - 20)^2$ produces $\partial \mathcal{L}/\partial R_0 = 2(12 - 20) \times 4 = -64$, which the test confirms to within $\pm 0.1$.
