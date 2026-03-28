# Execution-Guided Diffusion for Code Generation
## Technical Specification v1.0

**Project:** nCPU-Guided Code Diffusion (EGDC)
**Date:** 2026-03-27
**Status:** Draft

---

## 1. Abstract

We propose EGDC (Execution-Guided Diffusion for Code), a system that combines
masked discrete diffusion language modeling with differentiable program execution
to generate provably-correct code. Unlike autoregressive code models that generate
tokens left-to-right with no execution awareness, and unlike prior diffusion code
models that are semantically blind, EGDC uses nCPU's differentiable execution
engine as classifier guidance during the denoising process. Every refinement step
is informed by "does this code actually execute correctly against the specification?"
via differentiable gradient signals.

This is the first system to combine differentiable execution with diffusion-based
code generation. No prior work provides gradient flow from an execution engine
back through the denoising process.

---

## 2. Background & Prior Art

### 2.1 Discrete Diffusion for Code
- **MDLM** (Sahoo et al., NeurIPS 2024): Masked diffusion LM. Forward process
  randomly masks tokens; reverse process learns to unmask. Simplified
  Rao-Blackwellized objective = mixture of masked LM losses. SOTA among
  diffusion LMs on language benchmarks.
- **DiffuCoder** (Apple, 2025): 7B masked diffusion model for code. Coupled-GRPO
  for RL. Shows diffusion models can match AR on code tasks.
- **Mercury Coder** (Inception Labs, 2025): Commercial diffusion LLM, 10x faster
  than AR, 88-90% HumanEval.
- **LLaDA** (2025): 8B diffusion LM matching LLaMA3 on code/math.

### 2.2 Execution-Guided Generation
- **EG-CFG** (Lavon et al., NeurIPS 2026): Line-by-line execution during AR
  generation via CFG. 99.4% HumanEval. But AR-only, discrete traces only.
- **DiffusionCoder** (2025): Neural execution predictor + Langevin dynamics
  for discrete diffusion. +8.5% from execution guidance. Uses proxy, not
  true differentiable execution.
- **AlphaCode** (DeepMind): Generate millions, filter by execution. Brute force.

### 2.3 Differentiable Execution
- **∂4** (Bosnjak et al., 2017): Differentiable Forth interpreter with soft
  memory, soft PC. Limited to short traces.
- **TerpreT** (Gaunt et al., 2016): Gradient-based program synthesis. Key finding:
  constraint solvers dominate pure gradient methods.
- **NLI** (Macfarlane et al., ICLR 2026): Learns its own language with
  Gumbel-Softmax. Gradient-based test-time refinement.

### 2.4 The Gap
No prior work combines:
1. Masked discrete diffusion for code generation
2. Differentiable execution engine providing gradient-based guidance
3. True backpropagation through an execution engine into the denoising process

This is what EGDC provides.

---

## 3. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    EGDC SYSTEM OVERVIEW                       │
│                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │ Specification│───▶│  Masked      │───▶│  Generated     │  │
│  │ (NL + tests) │    │  Diffusion   │    │  Code Program  │  │
│  └─────────────┘    │  Denoiser    │    └────────────────┘  │
│                     │  (Transformer)│                        │
│                     └──────┬───────┘                        │
│                            │ x_t (partially unmasked)       │
│                            ▼                                │
│                     ┌──────────────┐                        │
│                     │  Token-to-   │                        │
│                     │  SoftProgram │                        │
│                     │  Bridge      │                        │
│                     └──────┬───────┘                        │
│                            │ SoftProgram params             │
│                            ▼                                │
│                     ┌──────────────┐                        │
│                     │  nCPU        │                        │
│                     │  Differentiable                       │
│                     │  Engine      │                        │
│                     └──────┬───────┘                        │
│                            │ ExecutionLoss gradients         │
│                            ▼                                │
│                     ┌──────────────┐                        │
│                     │  Classifier  │                        │
│                     │  Guidance    │                        │
│                     │  (score mod) │                        │
│                     └──────────────┘                        │
│                            │ guided score                   │
│                            ▼                                │
│                       next x_{t-1}                          │
└──────────────────────────────────────────────────────────────┘
```

### 3.1 Component Overview

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| Masked Diffusion Denoiser | Predict masked tokens | x_t + condition | token logits |
| Token-to-SoftProgram Bridge | Map tokens to executable form | token embeddings | SoftProgram params |
| nCPU DifferentiableEngine | Execute and provide gradients | SoftProgram + spec | ExecutionLoss + grads |
| Classifier Guidance | Modify denoising scores | model score + exec grad | guided score |
| Specification Encoder | Encode task description | NL + test cases | conditioning vector |

---

## 4. Detailed Component Specifications

### 4.1 Masked Diffusion Model (Base Generator)

**Architecture:** Encoder-only transformer

**Forward process (masking):**
```
q(x_t | x_0) = Cat(x_t; p = (1 - beta_t) * onehot(x_0) + beta_t * uniform)
```
where beta_t follows a cosine schedule from 0 to 1. At t=T, all tokens
are [MASK]. At t=0, all tokens are the original program.

**Reverse process (denoising):**
```
p_theta(x_{t-1} | x_t) = Cat(x_{t-1}; p = f_theta(x_t, t))
```
The model predicts the clean token at each masked position.

**Training objective (MDLM-style Rao-Blackwellized ELBO):**
```
L = E_{t, x_0, x_t} [ -sum_{i: x_t[i]=MASK} log p_theta(x_0[i] | x_t, t) ]
```
This is a weighted mixture of masked LM losses at different masking rates.

**Model sizes for phased development:**
- Phase 1: 125M params (12 layers, 768 hidden, 12 heads)
- Phase 2: 1B params (24 layers, 2048 hidden, 16 heads)
- Phase 3: 7B params (init from DiffuCoder/Qwen2.5-Coder weights)

**Vocabulary:**
- Phase 1: nCPU 14-opcode ISA tokens (~50 tokens: opcodes + registers + immediates)
- Phase 2+: BPE tokenizer on Python/Rust code (~32K-50K vocab)

**Conditioning:**
- Specification embedding prepended as prefix tokens
- Test case encoding: serialize (input, expected_output) pairs as special tokens
- Classifier-free guidance: 10% conditioning dropout during training

### 4.2 Token-to-SoftProgram Bridge

**Purpose:** Differentiably map token sequence embeddings into nCPU
SoftProgram parameters so the execution engine can process them.

**For Phase 1 (nCPU ISA programs):**

The bridge is straightforward because nCPU ISA tokens directly correspond
to SoftProgram fields:

```python
class TokenToSoftProgramBridge(nn.Module):
    """
    Maps diffusion model token logits -> SoftProgram parameters.
    
    Each instruction in the nCPU ISA is 4 tokens:
      [opcode] [dst_reg] [src_reg/imm] [aux/branch_target]
    
    SoftProgram expects per-instruction:
      opcode_logits: (num_opcodes,)    -- 14 opcodes
      dst_logits:    (num_registers,)  -- 8 registers
      src_logits:    (num_registers,)  -- 8 registers
      immediate:     scalar            -- float value
      branch_logits: (num_instructions,) -- branch targets
    """
    
    def __init__(self, hidden_dim, num_opcodes=14, num_registers=8):
        super().__init__()
        # Project token hidden states to SoftProgram parameter spaces
        self.opcode_proj = nn.Linear(hidden_dim, num_opcodes)
        self.dst_proj = nn.Linear(hidden_dim, num_registers)
        self.src_proj = nn.Linear(hidden_dim, num_registers)
        self.imm_proj = nn.Linear(hidden_dim, 1)
        self.branch_proj = nn.Linear(hidden_dim, max_instructions)
        
    def forward(self, token_hidden_states, mask_positions):
        """
        Args:
            token_hidden_states: (batch, seq_len, hidden_dim)
            mask_positions: which tokens are still masked
        Returns:
            SoftProgram parameters with gradient connectivity
        """
        # Group tokens into instruction quads
        instructions = self.group_into_instructions(token_hidden_states)
        
        # Project each instruction's representation to SoftProgram params
        opcode_logits = self.opcode_proj(instructions)  # (batch, num_instr, 14)
        dst_logits = self.dst_proj(instructions)        # (batch, num_instr, 8)
        src_logits = self.src_proj(instructions)        # (batch, num_instr, 8)
        immediates = self.imm_proj(instructions)        # (batch, num_instr, 1)
        branch_logits = self.branch_proj(instructions)  # (batch, num_instr, max_instr)
        
        return SoftProgram(
            opcode_logits=opcode_logits,
            dst_logits=dst_logits,
            src_logits=src_logits,
            immediates=immediates,
            branch_logits=branch_logits
        )
```

**For Phase 2+ (Python/Rust code):**

The bridge becomes a neural execution proxy -- a small transformer trained
to predict execution outcomes from code token embeddings:

```python
class NeuralExecutionProxy(nn.Module):
    """
    Predicts execution correctness from code embeddings.
    Trained on (code, test_cases, pass/fail) triples.
    Provides differentiable gradients w.r.t. code embeddings.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.code_encoder = TransformerEncoder(hidden_dim, layers=4)
        self.test_encoder = TransformerEncoder(hidden_dim, layers=2)
        self.predictor = MLP(hidden_dim * 2, hidden_dim, 1)
    
    def forward(self, code_embeddings, test_embeddings):
        code_repr = self.code_encoder(code_embeddings).mean(dim=1)
        test_repr = self.test_encoder(test_embeddings).mean(dim=1)
        pass_prob = torch.sigmoid(self.predictor(
            torch.cat([code_repr, test_repr], dim=-1)
        ))
        return pass_prob  # differentiable w.r.t. code_embeddings
```

### 4.3 nCPU Differentiable Execution Engine

**Existing components used (no modification needed):**
- `DifferentiableEngine` (ncpu/differentiable/execution.py)
- `DifferentiableALU` (ncpu/differentiable/alu.py)
- `SoftProgram` (ncpu/differentiable/programs.py)
- `ExecutionLoss` (ncpu/execution_training/execution_loss.py)

**Required modifications:**

1. **Batched execution** -- current engine processes one program at a time:
```python
class BatchedDifferentiableEngine(nn.Module):
    """
    Executes B programs in parallel via vmap or manual batching.
    Essential for diffusion training (many samples per step).
    """
    def __init__(self, num_registers=8, memory_size=256, max_steps=64):
        super().__init__()
        self.engines = None  # lazily initialized
        
    def forward(self, soft_programs_batch, target_states_batch):
        """
        Args:
            soft_programs_batch: SoftProgram with batch dimension
            target_states_batch: expected register states (batch, num_regs)
        Returns:
            execution_loss: (batch,) per-program execution loss
            gradients: gradient of loss w.r.t. SoftProgram params
        """
        # Option A: torch.vmap over single-program engine
        # Option B: manual loop (simpler, sufficient for Phase 1)
        losses = []
        for i in range(batch_size):
            final_state = self.engine.execute(soft_programs_batch[i])
            loss = F.mse_loss(final_state.registers, target_states_batch[i])
            losses.append(loss)
        return torch.stack(losses)
```

2. **Spec-conditioned execution** -- pipe test cases as target states:
```python
class SpecConditionedExecutor(nn.Module):
    """
    Executes a SoftProgram against a specification:
    - Input: initial register values (the "input")
    - Expected: final register values (the "output")
    - Returns: differentiable loss measuring correctness
    """
    def __init__(self):
        super().__init__()
        self.engine = DifferentiableEngine()
    
    def forward(self, soft_program, input_state, expected_output):
        self.engine.set_registers(input_state)
        final_state = self.engine.execute(soft_program)
        
        # Multiple loss signals:
        register_loss = F.mse_loss(
            final_state.registers, expected_output.registers
        )
        halted_loss = F.binary_cross_entropy(
            final_state.halted_prob, torch.ones(1)  # should halt cleanly
        )
        
        return register_loss + 0.1 * halted_loss
```

### 4.4 Classifier Guidance via Execution

**Core mechanism:**

At each denoising step t, we modify the model's predicted token
distribution using gradients from execution:

```
p_guided(x_{t-1} | x_t) ∝ p_model(x_{t-1} | x_t) * p(correct_exec | x_t)^gamma
```

In log space (working with logits):
```
logits_guided = logits_model + gamma * grad_{x_t} log p(correct_exec | x_t)
```

**Implementation:**

```python
class ExecutionGuidedSampler:
    """
    Denoising sampler with execution guidance.
    """
    def __init__(self, model, bridge, executor, gamma=1.0):
        self.model = model           # masked diffusion denoiser
        self.bridge = bridge         # token-to-SoftProgram
        self.executor = executor     # nCPU differentiable engine
        self.gamma = gamma           # guidance strength
    
    def denoise_step(self, x_t, t, spec, mask):
        """
        One step of guided denoising.
        
        Args:
            x_t: current token sequence (batch, seq_len) with MASKs
            t: current timestep
            spec: (input_state, expected_output)
            mask: boolean mask of which positions are still [MASK]
        Returns:
            x_{t-1}: less-masked token sequence
        """
        # 1. Get model's predicted logits for masked positions
        with torch.no_grad():
            model_logits = self.model(x_t, t)  # (batch, seq_len, vocab)
        
        # 2. Get execution guidance gradient
        # Enable grad for the token embeddings
        x_t_embed = self.model.get_embeddings(x_t)
        x_t_embed.requires_grad_(True)
        
        # Map to SoftProgram
        soft_program = self.bridge(x_t_embed, mask)
        
        # Execute differentiably
        exec_loss = self.executor(
            soft_program, spec.input_state, spec.expected_output
        )
        
        # Backprop through execution to get token-level gradients
        exec_loss.backward()
        exec_grad = x_t_embed.grad  # (batch, seq_len, embed_dim)
        
        # 3. Project execution gradient to logit space
        # grad w.r.t. embeddings -> grad w.r.t. logits via chain rule
        logit_guidance = self.embed_grad_to_logit_grad(exec_grad, x_t)
        
        # 4. Apply classifier guidance
        # Negative because we want to MINIMIZE execution loss
        guided_logits = model_logits - self.gamma * logit_guidance
        
        # 5. Sample from guided distribution
        # Low-confidence remasking: unmask tokens where model is most confident
        probs = F.softmax(guided_logits, dim=-1)
        confidence = probs.max(dim=-1).values  # (batch, seq_len)
        
        # Unmask the K most confident positions
        K = self.num_to_unmask(t)  # schedule: more tokens unmasked later
        top_k_positions = confidence[mask].topk(K).indices
        
        # Sample tokens at those positions
        x_next = x_t.clone()
        for pos in top_k_positions:
            x_next[:, pos] = torch.multinomial(probs[:, pos], 1)
        
        return x_next
    
    def generate(self, spec, seq_len, num_steps=256):
        """
        Full generation: start from all [MASK], iteratively unmask.
        """
        x = torch.full((1, seq_len), MASK_TOKEN)
        mask = torch.ones(seq_len, dtype=torch.bool)
        
        for t in reversed(range(num_steps)):
            x = self.denoise_step(x, t, spec, mask)
            mask = (x == MASK_TOKEN)
            
            if not mask.any():
                break  # fully unmasked
        
        return x
```

### 4.5 Guidance Schedule

Execution guidance should be adaptive across the denoising trajectory:

```
gamma(t) = gamma_max * schedule(t)
```

**Proposed schedules (to be ablated):**

1. **Constant:** gamma(t) = gamma_max
2. **Linear ramp:** gamma(t) = gamma_max * (1 - t/T)  
   (more guidance as code becomes clearer)
3. **Cosine:** gamma(t) = gamma_max * (1 + cos(pi * t/T)) / 2
4. **Threshold:** gamma(t) = gamma_max if t < T/2 else 0
   (only guide once code is partially formed)

Rationale for schedule: Early in denoising, code is mostly [MASK] tokens.
Execution of near-random code gives noisy/misleading gradients. Better to
let the language model do most of the work early, then sharpen with
execution guidance once the code skeleton is visible.

### 4.6 Specification Format

**Phase 1 (nCPU ISA):**
```python
@dataclass
class NCPUSpec:
    """A specification for an nCPU program."""
    description: str                    # "compute fibonacci(n)"
    input_registers: Dict[int, int]     # {0: 10}  (r0 = 10)
    expected_registers: Dict[int, int]  # {1: 55}  (r1 = fib(10))
    test_cases: List[Tuple[Dict, Dict]] # multiple (input, output) pairs
    max_steps: int = 64                 # execution budget
```

**Phase 2+ (Python/Rust):**
```python
@dataclass
class CodeSpec:
    """A specification for a code generation task."""
    description: str                     # natural language
    function_signature: str              # "def fibonacci(n: int) -> int:"
    test_cases: List[Tuple[str, str]]    # (input_str, expected_output_str)
    hidden_tests: List[Tuple[str, str]]  # held-out tests for evaluation
```

---

## 5. Training Pipeline

### 5.1 Phase 1 Training Data

**Source:** Procedurally generated nCPU ISA programs with known I/O behavior.

**Program families:**
1. Arithmetic: compute f(a,b) for various functions (add, multiply, power, etc.)
2. Conditionals: if-then-else on register values
3. Loops: iterative computation (factorial, fibonacci, sum, GCD)
4. Sorting: bubble sort, insertion sort on small arrays in registers
5. Bitwise: AND/OR/XOR manipulations with known outputs

**Generation strategy:**
```python
def generate_training_pair():
    """Generate a (spec, program) pair for training."""
    # 1. Pick a program template
    template = random.choice(PROGRAM_TEMPLATES)
    
    # 2. Randomize parameters (loop bounds, constants, etc.)
    program = template.instantiate(random_params())
    
    # 3. Execute on nCPU to get ground-truth I/O
    input_state = random_register_state()
    output_state = execute(program, input_state)
    
    # 4. Generate multiple test cases
    test_cases = []
    for _ in range(NUM_TEST_CASES):
        inp = random_register_state()
        out = execute(program, inp)
        test_cases.append((inp, out))
    
    # 5. Create spec
    spec = NCPUSpec(
        description=template.describe(program),
        input_registers=input_state,
        expected_registers=output_state,
        test_cases=test_cases
    )
    
    return spec, program.tokenize()
```

**Target dataset size:** 100K-1M (spec, program) pairs

**Tokenization for Phase 1:**
Each nCPU instruction becomes 4 tokens: [opcode] [dst] [src] [imm/target]
- Opcode tokens: NOP=0, MOV_IMM=1, ..., HALT=13
- Register tokens: R0=14, R1=15, ..., R7=21
- Immediate tokens: quantized to 256 levels (22-277)
- Branch target tokens: instruction indices (278-341)
- Special: [MASK]=342, [PAD]=343, [BOS]=344, [EOS]=345
- Total vocabulary: ~346 tokens

### 5.2 Training Procedure

**Stage 1: Base diffusion model (no execution guidance)**
```
for epoch in range(NUM_EPOCHS):
    for (spec, program) in dataloader:
        # Random masking rate t ~ Uniform(0, 1)
        t = torch.rand(1)
        mask = torch.rand(program.shape) < t
        x_t = program.clone()
        x_t[mask] = MASK_TOKEN
        
        # Predict original tokens at masked positions
        logits = model(x_t, t, spec_embedding)
        loss = F.cross_entropy(logits[mask], program[mask])
        
        loss.backward()
        optimizer.step()
```

**Stage 2: Fine-tune with execution guidance (after Stage 1 converges)**
```
for epoch in range(NUM_EPOCHS):
    for (spec, program) in dataloader:
        # Standard diffusion loss
        diffusion_loss = compute_diffusion_loss(model, spec, program)
        
        # Execution-aware loss: sample a denoised program, execute it
        with torch.enable_grad():
            x_denoised = model.sample_intermediate(program, t=0.3)
            soft_prog = bridge(model.get_embeddings(x_denoised))
            exec_loss = executor(soft_prog, spec.input, spec.expected)
        
        # Combined loss
        total_loss = diffusion_loss + lambda_exec * exec_loss
        total_loss.backward()
        optimizer.step()
```

**Stage 3: Coupled-GRPO reinforcement learning (optional, following DiffuCoder)**
```
# Generate programs via diffusion sampling
# Score by actual (non-differentiable) execution against test cases
# Use GRPO with complementary mask pairs for variance reduction
# Reward = 2.0 * pass_rate + 0.5 * syntactic_correctness
```

### 5.3 Hyperparameters

| Parameter | Phase 1 Value | Notes |
|-----------|---------------|-------|
| Model size | 125M | 12L, 768H, 12 heads |
| Sequence length | 128 tokens | 32 instructions × 4 tokens |
| Batch size | 256 | |
| Learning rate | 3e-4 | cosine decay to 3e-5 |
| Masking schedule | cosine | beta_t = 0.5*(1 - cos(pi*t)) |
| Denoising steps (inference) | 256 | ablate: 64, 128, 256, 512 |
| Guidance strength gamma | 1.0 | ablate: 0.0, 0.5, 1.0, 2.0, 5.0 |
| Execution loss weight lambda | 0.1 | for Stage 2 training |
| Guidance schedule | cosine ramp | more guidance as code clarifies |
| Low-confidence remasking | yes | unmask most-confident first |
| Temperature | 0.4 | for pass@1; 0.8 for pass@k |
| Training steps | 100K | ~25M program examples seen |
| Optimizer | AdamW | weight_decay=0.01 |
| Warmup | 2000 steps | linear warmup |

---

## 6. Evaluation

### 6.1 Metrics

**Primary:**
- **pass@1**: % of generated programs that pass all test cases (single attempt)
- **pass@k**: % where at least 1 of k attempts passes (k=5, 10, 100)
- **Syntactic validity**: % of programs that parse as valid nCPU ISA

**Secondary:**
- **Execution rate**: % of programs that halt without error (even if wrong output)
- **Partial correctness**: % of test cases passed (even if not all)
- **Generation speed**: tokens/second, programs/second
- **Diversity**: unique programs among k samples (for pass@k analysis)

### 6.2 Ablations

| Experiment | Purpose |
|------------|---------|
| No guidance (gamma=0) vs guided | Does execution guidance help? |
| Gamma sweep (0.5, 1, 2, 5) | How much guidance is optimal? |
| Guidance schedule comparison | When should guidance apply? |
| Denoising steps (64, 128, 256, 512) | Speed-quality tradeoff |
| Low-confidence vs random remasking | Which unmasking strategy wins? |
| With/without CFG on specification | Does spec conditioning help? |
| Stage 2 training vs inference-only guidance | Where should execution feedback enter? |

### 6.3 Baselines

1. **Unguided masked diffusion** (our model, gamma=0)
2. **Autoregressive transformer** (same size, same data)
3. **AR + best-of-N with execution filter** (generate N, keep first passing)
4. **Random program search** (uniform random programs, filter by execution)

### 6.4 Benchmark Tasks

**Tier 1 (Phase 1):**
- Arithmetic functions: add, multiply, power, modulo
- Fibonacci(n), factorial(n)
- GCD(a, b), LCM(a, b)
- Max/min of registers
- Simple conditionals (abs, clamp, sign)
- Iterative sum (1+2+...+n)

**Tier 2 (Phase 2):**
- HumanEval (164 problems)
- MBPP (500 problems)
- LiveCodeBench

**Tier 3 (Phase 3):**
- CodeContests
- SWE-bench

---

## 7. Implementation Plan

### Phase 1: Foundation (Weeks 1-2)

**Week 1:**
- [ ] Implement nCPU ISA tokenizer (encode/decode programs to token sequences)
- [ ] Build program generator for training data (100K+ programs)
- [ ] Set up MDLM training loop (PyTorch, from scratch or fork MDLM repo)
- [ ] Define model architecture (125M encoder-only transformer)
- [ ] Create dataset class and data loading pipeline

**Week 2:**
- [ ] Train base diffusion model (no guidance) on nCPU programs
- [ ] Implement basic sampling loop (iterative unmasking)
- [ ] Validate: model can reconstruct nCPU programs from partial masks
- [ ] Measure baseline pass@1 and pass@k without guidance
- [ ] Set up evaluation harness (execute generated programs, check outputs)

### Phase 2: Execution Bridge (Weeks 3-4)

**Week 3:**
- [ ] Implement TokenToSoftProgramBridge
- [ ] Batch the DifferentiableEngine
- [ ] Wire ExecutionLoss as differentiable signal from generated programs
- [ ] Test gradient flow: execution loss -> bridge -> token embeddings

**Week 4:**
- [ ] Implement ExecutionGuidedSampler
- [ ] Implement guidance schedules (constant, cosine, threshold)
- [ ] Run first guided generation experiments
- [ ] Compare guided vs unguided pass@1

### Phase 3: Optimization (Weeks 5-6)

- [ ] Hyperparameter sweep on gamma, schedule, denoising steps
- [ ] Implement Stage 2 training (execution loss in training loop)
- [ ] Implement coupled-GRPO (optional, if Stage 2 training helps)
- [ ] Run full ablation suite
- [ ] Optimize generation speed (batch execution, caching)

### Phase 4: Scaling (Weeks 7-12)

- [ ] Scale model to 1B params
- [ ] Extend to Python code generation (new tokenizer, NeuralExecutionProxy)
- [ ] Evaluate on HumanEval, MBPP
- [ ] Optional: initialize from DiffuCoder/Qwen2.5-Coder weights

### Phase 5: Paper (Weeks 12-16)

- [ ] Write paper: "Execution-Guided Diffusion for Code"
- [ ] Target venue: ICML, NeurIPS, or ICLR
- [ ] Key contributions:
  1. First differentiable execution guidance for code diffusion
  2. Token-to-SoftProgram bridge enabling gradient flow
  3. Ablations showing when/how execution guidance helps
  4. Analysis of guided vs unguided generation behavior

---

## 8. Risk Analysis

| Risk | Severity | Mitigation |
|------|----------|------------|
| Execution gradients are noisy for partially-masked code | High | Guidance schedule: only guide when code is mostly unmasked. Gradient clipping. |
| Batched differentiable execution is slow | Medium | Start with small programs (32 instructions). Use GPU tier if needed. |
| Guidance doesn't improve over unguided baseline | High | Multiple guidance mechanisms (classifier, CFG, reward-based). Architecture D (diffuse in SoftProgram space) as fallback. |
| 14-opcode ISA is too limited to be interesting | Medium | Phase 1 is proof-of-concept. Real value comes in Phase 2+ with Python/Rust. |
| Training data too synthetic/easy | Medium | Include adversarial programs, composition of primitives, edge cases. |
| Diffusion model can't learn nCPU ISA patterns | Low | ISA is regular and structured; masked LMs handle this easily. |

---

## 9. Hardware Requirements

**Phase 1 (125M model, nCPU ISA):**
- 1x GPU with 24GB+ VRAM (RTX 4090, A100)
- Or Apple Silicon Mac with 32GB+ unified memory (Metal + MPS)
- Training time: ~4-8 hours for 100K steps

**Phase 2 (1B model):**
- 1x A100 80GB or 2x RTX 4090
- Training time: ~24-48 hours

**Phase 3 (7B model):**
- 4-8x A100 80GB
- Training time: ~1-2 weeks

---

## 10. Success Criteria

**Minimum viable result (Phase 1):**
- Execution-guided diffusion achieves >50% pass@1 on Tier 1 benchmarks
- Guidance improves pass@1 by >10% absolute over unguided baseline
- Generated programs are syntactically valid >95% of the time

**Strong result (Phase 2):**
- Matches or exceeds AR baseline of same size on pass@1
- Demonstrates unique strengths: better pass@k, better on complex tasks,
  fill-in-the-middle capability
- Execution guidance provides consistent improvement across task types

**Home run (Phase 3):**
- Competitive with frontier models on HumanEval/MBPP
- Fastest correct code generation (diffusion speed + execution correctness)
- Clear ablation showing execution guidance is the key differentiator

---

## 11. Key References

1. Sahoo et al. "Simple and Effective Masked Diffusion Language Models" (MDLM), NeurIPS 2024
2. Apple. "DiffuCoder: Understanding and Improving Masked Diffusion Models for Code", 2025
3. Lavon et al. "Execution-Guided Classifier-Free Guidance for Code Generation" (EG-CFG), NeurIPS 2026
4. Austin et al. "Structured Denoising Diffusion Models in Discrete State-Spaces" (D3PM), NeurIPS 2021
5. Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (SEDD), ICML 2024
6. Nie et al. "Large Language Diffusion with Assistant" (LLaDA), 2025
7. Ho & Salimans. "Classifier-Free Diffusion Guidance", NeurIPS Workshop 2021
8. Dhariwal & Nichol. "Diffusion Models Beat GANs on Image Synthesis", NeurIPS 2021
9. Bosnjak et al. "Programming with a Differentiable Forth Interpreter" (∂4), ICML 2017
10. Macfarlane et al. "Neural Language Interpreter" (NLI), ICLR 2026

---

*End of Technical Specification*
