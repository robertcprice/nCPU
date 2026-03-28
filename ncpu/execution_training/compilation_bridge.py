"""Mode 2: Differentiable compilation bridge.

Connects language model hidden states to the DifferentiableCompiler and
DifferentiableEngine, enabling end-to-end gradient flow:

    LM hidden states -> projection -> DifferentiableCompiler encoder space
                                              |
                                     DifferentiableCompiler decoder
                                              |
                                         SoftProgram
                                              |
                                     DifferentiableEngine
                                              |
                                       ExecutionResult
                                              |
                                     execution loss (MSE)
                                              |
                                        loss.backward()
                                              |
    gradients flow back through everything <--+

This is Mode 2 from the architecture doc. The key innovation: no text
parsing step. Hidden states flow directly into program generation,
giving the LM dense gradient signal from actual execution.

The bridge has three learnable components:
  1. hidden_proj: Linear(lm_hidden_dim, compiler_d_model) — projects LM
     representations into the compiler's embedding space
  2. The DifferentiableCompiler's decoder heads — map projected
     representations to SoftProgram parameters
  3. (Optional) sequence_adapter: handles sequence length mismatch
     between LM and compiler

Usage:
    bridge = CompilationBridge(lm_hidden_dim=1536)  # e.g. Qwen3.5-0.8B

    # In training loop:
    lm_outputs = model(input_ids, output_hidden_states=True)
    hidden_states = lm_outputs.hidden_states[-1]  # [batch, seq, hidden]

    result = bridge(
        hidden_states[:, :32, :],  # Take first 32 tokens
        test_cases=[{"inputs": {0: 3.0, 1: 5.0}, "expected": {2: 8.0}}],
    )

    result.total_loss.backward()  # Gradients flow into LM hidden states!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.differentiable.execution import (
    DifferentiableEngine,
    SoftProgram,
    ExecutionResult,
    NUM_OPCODES,
)
from ncpu.differentiable.diff_compiler import (
    DifferentiableCompiler,
    AutoregressiveDecoder,
    CompilationResult,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompilationBridgeResult:
    """Result from the compilation bridge.

    All tensor fields maintain gradient connectivity back through the
    bridge projection, compiler decoder, execution engine, and into
    the original LM hidden states.

    Attributes:
        total_loss: Combined loss (execution + compilation + entropy).
            Differentiable w.r.t. lm_hidden_states.
        execution_loss: MSE between expected and actual register values.
        compilation_loss: Auxiliary loss from the compiler (entropy reg).
        program: The SoftProgram produced by the bridge.
        execution_results: Per-test-case ExecutionResults.
        projected_states: The hidden states after projection into
            compiler space, useful for analysis.
        num_correct: Number of output registers within tolerance.
        num_total: Total expected output registers.
        per_test_losses: Per-test-case loss values for diagnostics.
    """

    total_loss: torch.Tensor
    execution_loss: torch.Tensor
    compilation_loss: torch.Tensor
    program: SoftProgram
    execution_results: list[ExecutionResult] = field(default_factory=list)
    projected_states: Optional[torch.Tensor] = None
    num_correct: int = 0
    num_total: int = 0
    per_test_losses: list[float] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of output registers within tolerance."""
        return self.num_correct / max(self.num_total, 1)


# ---------------------------------------------------------------------------
# Sequence adapter: handles LM seq_len -> compiler seq_len mismatch
# ---------------------------------------------------------------------------


class SequenceAdapter(nn.Module):
    """Adapt variable-length LM sequences to the compiler's fixed length.

    LM hidden states may have arbitrary sequence length, but the
    DifferentiableCompiler expects a fixed max_source_len. This module
    provides several strategies for the adaptation:

    - 'pool': Mean-pool across the sequence dimension, then expand.
    - 'truncate': Take the first max_source_len positions.
    - 'linear': Learned linear projection across the sequence dimension.
    """

    def __init__(
        self,
        strategy: str = "pool",
        max_source_len: int = 32,
        d_model: int = 64,
    ):
        super().__init__()
        self.strategy = strategy
        self.max_source_len = max_source_len
        self.d_model = d_model

        if strategy == "linear":
            # Learnable projection: [any_len, d] -> [max_source_len, d]
            # We use a small attention-based pooling
            self.query = nn.Parameter(
                torch.randn(max_source_len, d_model) * 0.02
            )
            self.attn = nn.MultiheadAttention(
                d_model, num_heads=4, batch_first=True
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Adapt sequence to compiler's expected length.

        Args:
            x: [seq_len, d_model] or [batch, seq_len, d_model]

        Returns:
            [max_source_len, d_model] or [batch, max_source_len, d_model]
        """
        has_batch = x.dim() == 3

        if not has_batch:
            x = x.unsqueeze(0)  # [1, seq, d]

        seq_len = x.shape[1]

        if self.strategy == "truncate":
            if seq_len >= self.max_source_len:
                out = x[:, : self.max_source_len, :]
            else:
                # Pad with zeros
                pad = torch.zeros(
                    x.shape[0],
                    self.max_source_len - seq_len,
                    self.d_model,
                    device=x.device,
                    dtype=x.dtype,
                )
                out = torch.cat([x, pad], dim=1)

        elif self.strategy == "linear":
            # Cross-attention: learned queries attend to input sequence
            query = self.query.unsqueeze(0).expand(x.shape[0], -1, -1)
            out, _ = self.attn(query, x, x)

        else:  # pool
            # Mean pool -> repeat to fill max_source_len positions
            pooled = x.mean(dim=1, keepdim=True)  # [batch, 1, d]
            out = pooled.expand(-1, self.max_source_len, -1)

        if not has_batch:
            out = out.squeeze(0)

        return out


# ---------------------------------------------------------------------------
# Main bridge module
# ---------------------------------------------------------------------------


class CompilationBridge(nn.Module):
    """Bridge from LM hidden states to differentiable compilation + execution.

    This is the core Mode 2 module. It:
    1. Projects LM hidden states into the DifferentiableCompiler's d_model space
    2. Adapts sequence length to the compiler's expected input size
    3. Feeds through the compiler's decoder to produce a SoftProgram
    4. Executes the SoftProgram on the DifferentiableEngine
    5. Computes execution loss (MSE on expected output registers)
    6. Returns a result with full gradient connectivity

    The projection layer (hidden_proj) is the key learnable bridge: it
    maps from the LM's representation space to the compiler's. During
    training, gradients from execution error flow back through:
        loss -> engine -> SoftProgram -> decoder heads -> projected states
             -> hidden_proj -> LM hidden states

    This means the LM's hidden representations are directly shaped by
    whether the programs they induce actually compute correctly.

    Architecture:
        LM hidden [seq, lm_dim]
              |
        hidden_proj: Linear(lm_dim, d_model)
              |
        layer_norm
              |
        SequenceAdapter (pool/truncate/linear)
              |
        [max_source_len, d_model]
              |
        Compiler Encoder (Transformer)  -- uses compiler.encoder
              |
        Compiler Decoder (heads or AR)  -- uses compiler.decode/ar_decoder
              |
        SoftProgram
              |
        DifferentiableEngine.execute_soft
              |
        ExecutionResult
              |
        MSE loss vs expected outputs
    """

    def __init__(
        self,
        lm_hidden_dim: int = 896,
        compiler: Optional[DifferentiableCompiler] = None,
        engine: Optional[DifferentiableEngine] = None,
        compiler_d_model: int = 64,
        max_source_len: int = 32,
        max_program_len: int = 16,
        num_registers: int = 8,
        sequence_strategy: str = "pool",
        entropy_weight: float = 0.01,
        correctness_tolerance: float = 0.5,
        max_exec_steps: int = 16,
    ):
        """Initialize the compilation bridge.

        Args:
            lm_hidden_dim: Hidden dimension of the language model.
                Common values: 1536 (Qwen3.5-0.8B), 1536 (Qwen3.5-2B),
                2048 (LLaMA-7B), 768 (GPT-2).
            compiler: DifferentiableCompiler instance. If None, creates one
                with the specified d_model and dimensions.
            engine: DifferentiableEngine instance. If None, creates one.
            compiler_d_model: Hidden dimension of the compiler (default 64).
                Only used if compiler is None.
            max_source_len: Maximum source sequence length for the compiler.
            max_program_len: Maximum program length (instruction count).
            num_registers: Number of registers in the target ISA.
            sequence_strategy: How to handle LM->compiler sequence length
                mismatch. One of 'pool', 'truncate', 'linear'.
            entropy_weight: Weight for opcode entropy regularization.
            correctness_tolerance: Threshold for counting a register as correct.
            max_exec_steps: Maximum execution steps for the engine.
        """
        super().__init__()

        # Create or use provided compiler
        if compiler is not None:
            self.compiler = compiler
            compiler_d_model = compiler.d_model
            max_source_len = compiler.max_source_len
        else:
            self.compiler = DifferentiableCompiler(
                vocab_size=64,  # Won't use token_embed; we project directly
                d_model=compiler_d_model,
                max_source_len=max_source_len,
                max_program_len=max_program_len,
                num_registers=num_registers,
            )

        # Create or use provided engine
        self.engine = engine or DifferentiableEngine()

        # Store config
        self.lm_hidden_dim = lm_hidden_dim
        self.compiler_d_model = compiler_d_model
        self.max_source_len = max_source_len
        self.entropy_weight = entropy_weight
        self.correctness_tolerance = correctness_tolerance
        self.max_exec_steps = max_exec_steps

        # --- Learnable bridge components ---

        # Project LM hidden states into compiler's d_model space
        self.hidden_proj = nn.Linear(lm_hidden_dim, compiler_d_model)

        # Layer norm for stability after projection
        self.layer_norm = nn.LayerNorm(compiler_d_model)

        # Sequence length adapter
        self.seq_adapter = SequenceAdapter(
            strategy=sequence_strategy,
            max_source_len=max_source_len,
            d_model=compiler_d_model,
        )

        self._init_projection()

    def _init_projection(self) -> None:
        """Initialize projection with small weights for stable start."""
        nn.init.xavier_uniform_(self.hidden_proj.weight, gain=0.1)
        nn.init.zeros_(self.hidden_proj.bias)

    def project_hidden_states(
        self, lm_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Project LM hidden states into compiler embedding space.

        Args:
            lm_hidden_states: [seq_len, lm_hidden_dim] or
                [batch, seq_len, lm_hidden_dim]

        Returns:
            [max_source_len, compiler_d_model] or
            [batch, max_source_len, compiler_d_model]
            Projected and adapted to compiler's expected input shape.
        """
        # Project: [*, lm_dim] -> [*, d_model]
        projected = self.hidden_proj(lm_hidden_states)
        projected = self.layer_norm(projected)

        # Adapt sequence length
        adapted = self.seq_adapter(projected)

        return adapted

    def compile_from_hidden(
        self,
        lm_hidden_states: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[SoftProgram, torch.Tensor]:
        """Compile LM hidden states into a SoftProgram.

        This bypasses the compiler's token embedding and feeds projected
        hidden states directly into the compiler's Transformer encoder,
        then through the decoder to produce a SoftProgram.

        Args:
            lm_hidden_states: [seq_len, lm_hidden_dim] hidden states
                from the language model.
            temperature: Gumbel-softmax temperature for execution.

        Returns:
            (program, projected_states) where program is a SoftProgram
            and projected_states are the projected hidden representations.
        """
        # Project into compiler space
        projected = self.project_hidden_states(lm_hidden_states)

        # Feed through compiler's encoder (bypass token_embed)
        # projected is [max_source_len, d_model], encoder expects [1, seq, d]
        if projected.dim() == 2:
            encoder_input = projected.unsqueeze(0)  # [1, seq, d]
        else:
            encoder_input = projected

        # Add positional embeddings from the compiler
        seq_len = encoder_input.shape[1]
        pos_embed = self.compiler.pos_embed[:seq_len]
        encoder_input = encoder_input + pos_embed.unsqueeze(0)

        # Run through compiler's Transformer encoder
        encoded = self.compiler.encoder(encoder_input)  # [1, seq, d]

        # Decode to SoftProgram using compiler's decoder
        if self.compiler.decoder_mode == "autoregressive":
            program = self.compiler.ar_decoder(encoded, temperature=temperature)
        else:
            # Single-shot: mean-pool then decode
            context = encoded.mean(dim=1).squeeze(0)  # [d_model]
            program = self.compiler.decode(context)

        return program, projected

    def execute_program(
        self,
        program: SoftProgram,
        inputs: dict[int, float],
        temperature: float = 1.0,
    ) -> ExecutionResult:
        """Execute a SoftProgram on the differentiable engine.

        Args:
            program: SoftProgram from compile_from_hidden.
            inputs: Initial register values {reg_index: value}.
            temperature: Gumbel-softmax temperature.

        Returns:
            ExecutionResult with register values and metadata.
        """
        return self.engine.execute_soft(
            program,
            inputs,
            max_steps=self.max_exec_steps,
            temperature=temperature,
            skip_bitwise=True,
        )

    def compute_execution_loss(
        self,
        execution_result: ExecutionResult,
        expected: dict[int, float],
    ) -> tuple[torch.Tensor, int, int]:
        """Compute MSE loss between execution result and expected outputs.

        Args:
            execution_result: Result from execute_program.
            expected: Expected register values {reg_index: value}.

        Returns:
            (loss, num_correct, num_total)
        """
        loss = torch.tensor(0.0)
        num_correct = 0
        num_total = len(expected)

        for reg_idx, target_val in expected.items():
            actual = execution_result.registers[reg_idx]
            reg_loss = (actual - target_val) ** 2
            loss = loss + reg_loss

            if abs(actual.item() - target_val) < self.correctness_tolerance:
                num_correct += 1

        if num_total > 0:
            loss = loss / num_total

        return loss, num_correct, num_total

    def compute_entropy_loss(self, program: SoftProgram) -> torch.Tensor:
        """Compute opcode entropy regularization.

        Low entropy = compiler is confident about opcodes.
        We want to encourage decisive opcode selection.

        Args:
            program: SoftProgram with opcode_logits.

        Returns:
            Mean entropy across instruction positions.
        """
        if self.entropy_weight <= 0.0:
            return torch.tensor(0.0)

        opcode_probs = F.softmax(program.opcode_logits, dim=-1)
        entropy = -(opcode_probs * (opcode_probs + 1e-8).log()).sum(dim=-1)
        return self.entropy_weight * entropy.mean()

    def forward(
        self,
        lm_hidden_states: torch.Tensor,
        test_cases: list[dict],
        temperature: float = 1.0,
    ) -> CompilationBridgeResult:
        """Full forward pass: project -> compile -> execute -> loss.

        This is the main entry point for Mode 2 training. Given LM hidden
        states and test cases, produces a differentiable loss that can be
        backpropagated through the entire pipeline into the hidden states.

        Args:
            lm_hidden_states: [seq_len, lm_hidden_dim] or
                [batch, seq_len, lm_hidden_dim] from the language model.
                Typically the output of the last (or a specific) layer.
            test_cases: List of test case dicts, each with:
                - 'inputs': {reg_index: value} initial register values
                - 'expected': {reg_index: value} expected output values
            temperature: Gumbel-softmax temperature for compilation and
                execution. Lower = more discrete, higher = more exploration.

        Returns:
            CompilationBridgeResult with total_loss, per-test results,
            and the generated SoftProgram. total_loss has gradients
            connected back to lm_hidden_states.
        """
        # Handle batched input: take first item for now
        # (compilation is per-program, not batched)
        if lm_hidden_states.dim() == 3:
            lm_hidden_states = lm_hidden_states[0]  # [seq, hidden]

        # Step 1: Compile from hidden states
        program, projected = self.compile_from_hidden(
            lm_hidden_states, temperature=temperature
        )

        # Step 2: Execute on each test case and accumulate loss
        total_exec_loss = torch.tensor(0.0)
        total_correct = 0
        total_expected = 0
        exec_results = []
        per_test_losses = []

        for tc in test_cases:
            inputs = tc.get("inputs", {})
            expected = tc.get("expected", {})

            # Execute
            exec_result = self.execute_program(
                program, inputs, temperature=temperature
            )
            exec_results.append(exec_result)

            # Compute loss for this test case
            tc_loss, nc, nt = self.compute_execution_loss(
                exec_result, expected
            )
            total_exec_loss = total_exec_loss + tc_loss
            total_correct += nc
            total_expected += nt
            per_test_losses.append(tc_loss.item())

        if len(test_cases) > 0:
            total_exec_loss = total_exec_loss / len(test_cases)

        # Step 3: Compilation auxiliary loss (entropy regularization)
        compilation_loss = self.compute_entropy_loss(program)

        # Step 4: Combined loss
        total_loss = total_exec_loss + compilation_loss

        return CompilationBridgeResult(
            total_loss=total_loss,
            execution_loss=total_exec_loss,
            compilation_loss=compilation_loss,
            program=program,
            execution_results=exec_results,
            projected_states=projected,
            num_correct=total_correct,
            num_total=total_expected,
            per_test_losses=per_test_losses,
        )

    def training_step(
        self,
        lm_hidden_states: torch.Tensor,
        test_cases: list[dict],
        temperature: float = 1.0,
        return_result: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, CompilationBridgeResult]:
        """Convenience method for training loops.

        Calls forward() and returns just the loss (or loss + result).
        Designed to integrate cleanly with existing training code.

        Args:
            lm_hidden_states: LM hidden states.
            test_cases: Test cases with inputs/expected.
            temperature: Gumbel-softmax temperature.
            return_result: If True, return (loss, result) tuple.

        Returns:
            loss tensor, or (loss, CompilationBridgeResult) if return_result.
        """
        result = self.forward(lm_hidden_states, test_cases, temperature)
        if return_result:
            return result.total_loss, result
        return result.total_loss
