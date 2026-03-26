"""Differentiable compilation pipeline.

Connects a neural compiler to the differentiable execution engine, enabling
end-to-end gradient flow: source code -> compilation -> execution -> loss.

This is a fundamentally new capability. Traditional compilers are discrete
black boxes: you cannot ask "how should I change the source code to improve
the program's output?" because there is no gradient through compilation.
With a differentiable compiler, that question has a precise mathematical
answer: the gradient of the execution loss with respect to the source
representation tells you exactly how to modify the source to improve output.

This module enables three novel workflows:

1. **Compiler training via execution feedback**: Train the compiler not by
   supervised instruction-matching, but by whether the compiled programs
   actually compute the right answer. The compiler learns compilation
   strategies that the execution engine validates end-to-end.

2. **Source-level parameter optimization**: Given a parameterized source
   program, optimize the parameters by backpropagating through both
   compilation and execution. For instance, find the loop bound or
   constant that minimizes a cost function evaluated after execution.

3. **Program synthesis from specifications**: Given input-output examples,
   jointly search over source representations and compiler behavior to
   discover source code that compiles to a correct program.

The compiler is implemented as a neural sequence-to-sequence model:
- Encoder: Transformer encoder maps source tokens to a context vector
- Decoder: Linear heads project the context into SoftProgram parameters
  (opcode logits, register logits, immediates, branch logits)

Because both the compiler output (SoftProgram) and the execution engine
(DifferentiableEngine.execute_soft) are fully differentiable, gradients
flow seamlessly from execution loss back through the compiler weights
and into the source embedding.

Architecture diagram:

    source_tokens
         |
    [token_embed + pos_embed]
         |
    [Transformer Encoder]
         |
    [mean pool -> context vector]
         |
    +----+----+----+----+----+
    |    |    |    |    |    |
  opcode dst src1 src2 imm branch
  logits logits logits logits values logits
    |    |    |    |    |    |
    +----+----+----+----+----+
         |
    [SoftProgram]
         |
    [DifferentiableEngine.execute_soft]
         |
    ExecutionResult (registers, flags)
         |
    loss = f(result, target)
         |
    loss.backward()  <-- gradients flow through EVERYTHING
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .execution import (
    DifferentiableEngine,
    FixedProgram,
    SoftProgram,
    Instruction,
    OPCODES,
    NUM_OPCODES,
    ExecutionResult,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CompilationResult:
    """Result of differentiable compilation.

    Holds both the intermediate SoftProgram produced by the compiler and the
    ExecutionResult from running it.  All tensor fields maintain gradient
    connectivity so the entire pipeline can be trained end-to-end.

    Attributes:
        program: The SoftProgram emitted by the compiler.  Inspectable via
            program.format_program() or program.extract_discrete_program().
        source_embedding: The continuous embedding of the source tokens,
            useful for visualizing what the compiler learned about the source.
        compilation_loss: Auxiliary loss from the compiler itself (e.g.
            entropy regularization on opcode distributions).  Zero when no
            auxiliary objective is active.
        execution_result: The result of executing the compiled program, or
            None if execution was not requested.
    """

    program: SoftProgram
    source_embedding: torch.Tensor
    compilation_loss: torch.Tensor
    execution_result: Optional[ExecutionResult]


# ---------------------------------------------------------------------------
# Autoregressive instruction decoder
# ---------------------------------------------------------------------------


class AutoregressiveDecoder(nn.Module):
    """Generate program instructions autoregressively.

    Instead of generating all instruction slots at once from a single
    context vector (which loses structural information), this decoder
    generates one instruction at a time, where each instruction is
    conditioned on the source encoding AND all previously generated
    instructions.

    This is analogous to how seq2seq translation works: the decoder
    attends to the encoder output and generates one token at a time.
    In our case each "token" is a full instruction (opcode + registers +
    immediate + branch target).

    Architecture:
        - Learned position embeddings form the decoder input.
        - A causal Transformer decoder cross-attends to the encoder
          output, so position *i* can see positions 0..i but not i+1..L.
        - Linear heads project each decoded position into instruction
          fields (opcode logits, register logits, immediate, branch logits).

    The causal mask enforces the autoregressive property during a single
    forward pass (teacher-forced style), which is efficient and fully
    differentiable while still capturing inter-instruction dependencies.
    """

    def __init__(
        self,
        d_model: int = 64,
        max_program_len: int = 16,
        num_registers: int = 8,
        num_opcodes: int = NUM_OPCODES,
        nhead: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_program_len = max_program_len
        self.num_registers = num_registers
        self.num_opcodes = num_opcodes

        # Instruction embedding (for conditioning on previous instructions)
        self.inst_embed = nn.Linear(
            num_opcodes + num_registers * 3 + 1, d_model
        )
        self.pos_embed = nn.Parameter(
            torch.randn(max_program_len, d_model) * 0.1
        )

        # Cross-attention decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Output heads (per-position)
        self.opcode_head = nn.Linear(d_model, num_opcodes)
        self.dst_head = nn.Linear(d_model, num_registers)
        self.src1_head = nn.Linear(d_model, num_registers)
        self.src2_head = nn.Linear(d_model, num_registers)
        self.imm_head = nn.Linear(d_model, 1)
        self.branch_head = nn.Linear(d_model, max_program_len)

        self._init_weights()

    def _init_weights(self) -> None:
        """Small initialization for stable training start."""
        for head in (
            self.opcode_head,
            self.dst_head,
            self.src1_head,
            self.src2_head,
            self.imm_head,
            self.branch_head,
        ):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward(
        self, encoder_output: torch.Tensor, temperature: float = 1.0
    ) -> SoftProgram:
        """Generate a program autoregressively from encoder output.

        The decoder uses a causal mask so that each instruction position
        can only attend to itself and earlier positions, enforcing the
        autoregressive property.  Cross-attention to the encoder output
        provides conditioning on the source program.

        Args:
            encoder_output: [1, seq_len, d_model] from the source encoder.
            temperature: Gumbel-softmax temperature (unused directly here
                but kept in the signature for API consistency with the
                compilation pipeline).

        Returns:
            SoftProgram with generated instruction parameters, where each
            instruction's logits are differentiable functions of the
            encoder output and the decoder weights.
        """
        program = SoftProgram(
            self.max_program_len,
            self.num_registers,
            self.num_opcodes,
        )

        # Build decoder input: learned position embeddings
        # Each position attends to encoder output via cross-attention
        tgt = self.pos_embed.unsqueeze(0)  # [1, max_len, d_model]

        # Causal mask so position i can only attend to positions 0..i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            self.max_program_len
        )

        # Decode all positions at once (teacher-forced style)
        decoded = self.decoder(
            tgt, encoder_output, tgt_mask=causal_mask
        )  # [1, max_len, d_model]
        decoded = decoded.squeeze(0)  # [max_len, d_model]

        # Remove SoftProgram's random-init nn.Parameters from the
        # parameter registry so we can replace them with live tensors
        # that maintain gradient flow through the decoder.
        _attrs = (
            "opcode_logits",
            "dst_logits",
            "src1_logits",
            "src2_logits",
            "immediates",
            "branch_logits",
        )
        for attr in _attrs:
            if attr in program._parameters:
                del program._parameters[attr]

        # Generate instruction fields from decoded representations
        program.opcode_logits = self.opcode_head(decoded)
        program.dst_logits = self.dst_head(decoded)
        program.src1_logits = self.src1_head(decoded)
        program.src2_logits = self.src2_head(decoded)
        program.immediates = self.imm_head(decoded).squeeze(-1)
        program.branch_logits = self.branch_head(decoded)

        return program


# ---------------------------------------------------------------------------
# Neural compiler
# ---------------------------------------------------------------------------


class DifferentiableCompiler(nn.Module):
    """A neural compiler that maps source representations to executable programs.

    This is a sequence-to-sequence model where:
    - Input: source code tokens (integer IDs embedded as continuous vectors)
    - Output: a SoftProgram whose every parameter is a differentiable function
      of the compiler weights and the source embedding

    Because the output is a SoftProgram (not a discrete instruction list),
    gradients from execution loss flow backward through the compiler.  This
    means the compiler can be trained purely from execution-level supervision:
    "did the compiled program produce the right answer?"

    The architecture is intentionally simple (2-layer Transformer encoder +
    linear decoder heads) to demonstrate the principle.  A production system
    could use cross-attention decoding, autoregressive instruction generation,
    or even a diffusion-based program generator -- the key insight is that
    the SoftProgram interface makes any such architecture trainable via
    execution feedback.

    Architecture:
        source_tokens -> Embedding + positional -> Transformer Encoder
                                                        |
                                                   mean-pool
                                                        |
                                          +---------+---+---+---------+
                                          |         |       |         |
                                     opcode_head dst_head ... branch_head
                                          |         |       |         |
                                      [L, N_op] [L, N_reg] ...  [L, L]

    where L = max_program_len, N_op = num_opcodes, N_reg = num_registers.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        max_source_len: int = 32,
        max_program_len: int = 16,
        num_registers: int = 8,
        num_opcodes: int = NUM_OPCODES,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        decoder_mode: str = "single_shot",
    ):
        """Initialize the differentiable compiler.

        Args:
            vocab_size: Number of distinct source tokens.
            d_model: Embedding and Transformer hidden dimension.
            max_source_len: Maximum number of source tokens accepted.
            max_program_len: Maximum number of instructions the compiler
                can emit.  This is the length of the generated SoftProgram.
            num_registers: Number of registers in the target architecture.
            num_opcodes: Number of opcodes in the target ISA.
            nhead: Number of attention heads in the Transformer encoder.
            num_encoder_layers: Depth of the Transformer encoder.
            dim_feedforward: FFN width inside the Transformer encoder.
            dropout: Dropout rate (0.0 for small models / deterministic use).
            decoder_mode: ``"single_shot"`` uses the original linear-head
                decoder that projects the mean-pooled encoder context into
                all instruction slots at once.  ``"autoregressive"`` uses
                a Transformer decoder with causal masking, so each
                instruction position cross-attends to the encoder output
                and can only see earlier instruction positions.
        """
        super().__init__()
        if decoder_mode not in ("single_shot", "autoregressive"):
            raise ValueError(
                f"decoder_mode must be 'single_shot' or 'autoregressive', "
                f"got {decoder_mode!r}"
            )
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_source_len = max_source_len
        self.max_program_len = max_program_len
        self.num_registers = num_registers
        self.num_opcodes = num_opcodes
        self.decoder_mode = decoder_mode

        # --- Source encoder ---
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(max_source_len, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        # --- Instruction decoder ---
        if decoder_mode == "autoregressive":
            self.ar_decoder = AutoregressiveDecoder(
                d_model=d_model,
                max_program_len=max_program_len,
                num_registers=num_registers,
                num_opcodes=num_opcodes,
                nhead=nhead,
            )
            # Placeholder attributes so _init_weights does not fail.
            # These are not used in autoregressive mode.
            self.opcode_head = None
            self.dst_head = None
            self.src1_head = None
            self.src2_head = None
            self.imm_head = None
            self.branch_head = None
        else:
            self.ar_decoder = None
            # --- Single-shot instruction decoder heads ---
            # Each head projects the pooled encoder output into the full
            # parameter tensor for one aspect of the program.
            self.opcode_head = nn.Linear(
                d_model, max_program_len * num_opcodes
            )
            self.dst_head = nn.Linear(
                d_model, max_program_len * num_registers
            )
            self.src1_head = nn.Linear(
                d_model, max_program_len * num_registers
            )
            self.src2_head = nn.Linear(
                d_model, max_program_len * num_registers
            )
            self.imm_head = nn.Linear(d_model, max_program_len)
            self.branch_head = nn.Linear(
                d_model, max_program_len * max_program_len
            )

            self._init_weights()

    def _init_weights(self) -> None:
        """Initialize decoder heads with small weights for stable training.

        Small initialization keeps the initial SoftProgram close to uniform
        distributions, which gives gradient descent room to differentiate
        between opcodes and registers without committing too early.
        """
        for head in (
            self.opcode_head,
            self.dst_head,
            self.src1_head,
            self.src2_head,
            self.imm_head,
            self.branch_head,
        ):
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def encode(self, source_tokens: torch.Tensor) -> torch.Tensor:
        """Encode source tokens into a context representation.

        In single-shot mode, returns a mean-pooled context vector of
        shape ``[d_model]``.  In autoregressive mode, returns the full
        encoder sequence of shape ``[1, seq_len, d_model]`` because the
        AR decoder needs per-position cross-attention.

        Args:
            source_tokens: [seq_len] integer token IDs.

        Returns:
            [d_model] context vector (single_shot) or
            [1, seq_len, d_model] encoder sequence (autoregressive).
        """
        seq_len = source_tokens.shape[0]
        embedded = self.token_embed(source_tokens) + self.pos_embed[:seq_len]
        # Transformer expects [batch, seq, d_model]
        encoded = self.encoder(embedded.unsqueeze(0))  # [1, seq, d_model]

        if self.decoder_mode == "autoregressive":
            return encoded  # [1, seq_len, d_model]
        else:
            pooled = encoded.mean(dim=1).squeeze(0)  # [d_model]
            return pooled

    def decode(self, context: torch.Tensor) -> SoftProgram:
        """Decode a context vector into a SoftProgram.

        The SoftProgram's nn.Parameter attributes are replaced with plain
        tensors that are live nodes in the computational graph.  This is
        critical for gradient flow: nn.Parameter() calls .detach() internally,
        which would sever the gradient chain from compiler -> program ->
        execution.  By assigning raw tensors, autograd tracks the full path
        from context -> linear head -> program attribute -> execution result.

        SoftProgram.get_soft_instruction accesses these attributes via
        self.opcode_logits, etc., which works identically whether they are
        nn.Parameter or plain Tensor -- PyTorch dispatches the same ops.

        Args:
            context: [d_model] context vector from the encoder.

        Returns:
            A SoftProgram whose parameters are differentiable functions of
            the compiler weights and the input context.
        """
        program = SoftProgram(
            self.max_program_len,
            self.num_registers,
            self.num_opcodes,
        )

        # Replace SoftProgram's random-init nn.Parameters with live tensors
        # from the compiler's decoder heads.  This is critical for gradient
        # flow: nn.Parameter() calls .detach() internally, which would sever
        # the gradient chain from compiler -> program -> execution.
        #
        # Because SoftProgram is an nn.Module that registered these as
        # nn.Parameter in __init__, we must first delete them from the
        # parameter registry, then assign the live tensors as plain
        # attributes.  get_soft_instruction accesses them via self.X,
        # which dispatches identically for Parameter and Tensor.
        _attrs = (
            "opcode_logits", "dst_logits", "src1_logits",
            "src2_logits", "immediates", "branch_logits",
        )
        for attr in _attrs:
            if attr in program._parameters:
                del program._parameters[attr]

        program.opcode_logits = self.opcode_head(context).reshape(
            self.max_program_len, self.num_opcodes
        )
        program.dst_logits = self.dst_head(context).reshape(
            self.max_program_len, self.num_registers
        )
        program.src1_logits = self.src1_head(context).reshape(
            self.max_program_len, self.num_registers
        )
        program.src2_logits = self.src2_head(context).reshape(
            self.max_program_len, self.num_registers
        )
        program.immediates = self.imm_head(context)
        program.branch_logits = self.branch_head(context).reshape(
            self.max_program_len, self.max_program_len
        )

        return program

    def compile(self, source_tokens: torch.Tensor) -> SoftProgram:
        """Compile source tokens into a SoftProgram.

        This is the main entry point: encode, then decode.  The returned
        SoftProgram is fully differentiable with respect to both the
        compiler weights and (via the embedding) the source tokens.

        In single-shot mode the encoder output is mean-pooled into a
        single context vector and decoded by linear heads.  In
        autoregressive mode the full encoder sequence is passed to a
        Transformer decoder with causal masking.

        Args:
            source_tokens: [seq_len] integer token IDs.

        Returns:
            SoftProgram ready for execution by DifferentiableEngine.
        """
        encoded = self.encode(source_tokens)

        if self.decoder_mode == "autoregressive":
            return self.ar_decoder(encoded)
        else:
            return self.decode(encoded)

    def forward(self, source_tokens: torch.Tensor) -> SoftProgram:
        """nn.Module forward -- delegates to compile()."""
        return self.compile(source_tokens)


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class DifferentiableCompilationPipeline(nn.Module):
    """End-to-end differentiable compilation and execution.

    Connects the three stages with unbroken gradient flow:

        Source tokens  -->  DifferentiableCompiler  -->  SoftProgram
                                                             |
                                                    DifferentiableEngine
                                                             |
                                                      ExecutionResult
                                                             |
                                                      loss function
                                                             |
                                                       loss.backward()
                                                             |
        Compiler weights  <--  gradients flow backward  <----+

    This is the central contribution: a pipeline where the compiler is not
    a fixed preprocessor but a differentiable function trained by how well
    the compiled programs actually execute.

    The pipeline supports three training modes:

    1. **Supervised compiler training** (train_compiler): Given source code
       and expected execution outputs, train the compiler to produce programs
       that compute correctly.

    2. **Compiler + source co-optimization** (compile_and_execute): Optimize
       both source embeddings and compiler weights simultaneously.

    3. **Specification-driven synthesis**: Combine with ProgramSynthesizer's
       loss functions to discover source code from I/O specifications.
    """

    def __init__(
        self,
        compiler: Optional[DifferentiableCompiler] = None,
        engine: Optional[DifferentiableEngine] = None,
    ):
        """Initialize the pipeline.

        Args:
            compiler: The neural compiler.  If None, a default compiler
                with reasonable hyperparameters is created.
            engine: The differentiable execution engine.  If None, a default
                engine is created.
        """
        super().__init__()
        self.compiler = compiler or DifferentiableCompiler()
        self.engine = engine or DifferentiableEngine()

    def compile_and_execute(
        self,
        source_tokens: torch.Tensor,
        inputs: Optional[dict[int, float]] = None,
        max_steps: int = 16,
        temperature: float = 1.0,
        skip_bitwise: bool = True,
        entropy_weight: float = 0.0,
    ) -> CompilationResult:
        """Compile source code and execute the result.

        Full gradient flow from execution result back to source tokens.

        Args:
            source_tokens: [seq_len] integer token IDs.
            inputs: Initial register values {reg_index: value}.
            max_steps: Maximum execution steps.
            temperature: Gumbel-softmax temperature for the SoftProgram.
                Lower values push toward discrete programs.
            skip_bitwise: Skip expensive bitwise operations in the ALU.
                Safe when the target computation is arithmetic-only.
            entropy_weight: Weight for opcode entropy regularization.
                When > 0, adds an auxiliary loss that encourages the compiler
                to produce decisive (low-entropy) opcode distributions,
                preventing the "soft mush" failure mode where every
                instruction is a uniform mixture of all opcodes.

        Returns:
            CompilationResult with program, embeddings, and execution result.
        """
        inputs = inputs or {}

        # Compile: source tokens -> SoftProgram
        program = self.compiler.compile(source_tokens)

        # Execute: SoftProgram -> ExecutionResult
        exec_result = self.engine.execute_soft(
            program,
            inputs,
            max_steps=max_steps,
            temperature=temperature,
            skip_bitwise=skip_bitwise,
        )

        # Auxiliary compilation loss: opcode entropy regularization.
        # Low entropy = the compiler is confident about which opcode each
        # instruction should be.  This prevents degenerate solutions where
        # every instruction is a uniform blend of all opcodes.
        compilation_loss = torch.tensor(0.0)
        if entropy_weight > 0.0:
            opcode_probs = F.softmax(program.opcode_logits, dim=-1)
            # Per-instruction entropy, averaged over all instruction slots
            entropy = -(opcode_probs * (opcode_probs + 1e-8).log()).sum(dim=-1)
            compilation_loss = entropy_weight * entropy.mean()

        return CompilationResult(
            program=program,
            source_embedding=self.compiler.token_embed(source_tokens),
            compilation_loss=compilation_loss,
            execution_result=exec_result,
        )

    def train_compiler(
        self,
        training_data: list[tuple[torch.Tensor, dict[int, float], dict[int, float]]],
        epochs: int = 100,
        lr: float = 0.001,
        temperature_start: float = 2.0,
        temperature_end: float = 0.5,
        entropy_weight: float = 0.01,
        grad_clip: float = 5.0,
        max_steps: int = 16,
        verbose: bool = True,
    ) -> list[float]:
        """Train the compiler to produce programs that compute correctly.

        Each training example is a tuple of:
        - source_tokens: the tokenized source code
        - inputs: initial register values
        - targets: expected register values after execution

        The compiler learns by backpropagating execution error through the
        entire pipeline.  Temperature annealing pushes the SoftProgram from
        soft mixtures toward discrete instructions as training progresses.

        Args:
            training_data: List of (source_tokens, inputs, target_outputs).
            epochs: Number of training epochs.
            lr: Learning rate for Adam optimizer.
            temperature_start: Initial Gumbel-softmax temperature (high =
                more exploration, soft mixtures).
            temperature_end: Final temperature (low = more exploitation,
                near-discrete programs).
            entropy_weight: Weight for opcode entropy regularization loss.
            grad_clip: Maximum gradient norm for stability.
            max_steps: Maximum execution steps per program.
            verbose: Print progress every 10% of epochs.

        Returns:
            List of per-epoch average losses.
        """
        optimizer = torch.optim.Adam(self.compiler.parameters(), lr=lr)
        loss_history: list[float] = []

        for epoch in range(epochs):
            # Temperature annealing: linear decay from start to end
            progress = epoch / max(epochs - 1, 1)
            temperature = temperature_start + (
                temperature_end - temperature_start
            ) * progress

            epoch_loss = torch.tensor(0.0)
            optimizer.zero_grad()

            for source_tokens, inputs, targets in training_data:
                result = self.compile_and_execute(
                    source_tokens,
                    inputs,
                    max_steps=max_steps,
                    temperature=temperature,
                    skip_bitwise=True,
                    entropy_weight=entropy_weight,
                )

                # Execution loss: MSE between target and actual registers
                exec_loss = torch.tensor(0.0)
                for reg_idx, target_val in targets.items():
                    exec_loss = exec_loss + (
                        result.execution_result.registers[reg_idx] - target_val
                    ) ** 2

                # Total loss = execution loss + compilation regularization
                epoch_loss = epoch_loss + exec_loss + result.compilation_loss

            epoch_loss = epoch_loss / len(training_data)
            epoch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.compiler.parameters(), grad_clip)
            optimizer.step()

            loss_val = epoch_loss.item()
            loss_history.append(loss_val)

            if verbose and (
                epoch % max(epochs // 10, 1) == 0 or epoch == epochs - 1
            ):
                print(
                    f"  epoch {epoch:4d}/{epochs}  "
                    f"loss={loss_val:8.4f}  "
                    f"temp={temperature:.3f}"
                )

        return loss_history

    def evaluate(
        self,
        test_data: list[tuple[torch.Tensor, dict[int, float], dict[int, float]]],
        temperature: float = 0.1,
        max_steps: int = 16,
    ) -> dict[str, float]:
        """Evaluate the compiler on test data.

        Uses a low temperature to get near-discrete program behavior.

        Args:
            test_data: Same format as training_data.
            temperature: Low temperature for near-discrete evaluation.
            max_steps: Maximum execution steps.

        Returns:
            Dictionary with 'mse', 'max_error', and 'num_correct' metrics.
        """
        total_se = 0.0
        max_err = 0.0
        num_correct = 0
        num_targets = 0

        with torch.no_grad():
            for source_tokens, inputs, targets in test_data:
                result = self.compile_and_execute(
                    source_tokens,
                    inputs,
                    max_steps=max_steps,
                    temperature=temperature,
                    skip_bitwise=True,
                )
                for reg_idx, target_val in targets.items():
                    actual = result.execution_result.registers[reg_idx].item()
                    error = abs(actual - target_val)
                    total_se += error ** 2
                    max_err = max(max_err, error)
                    if error < 1.0:
                        num_correct += 1
                    num_targets += 1

        return {
            "mse": total_se / max(num_targets, 1),
            "max_error": max_err,
            "num_correct": num_correct,
            "num_targets": num_targets,
            "accuracy": num_correct / max(num_targets, 1),
        }


# ---------------------------------------------------------------------------
# Simple tokenizer for demonstrations
# ---------------------------------------------------------------------------


class SimpleTokenizer:
    """Tokenize simple arithmetic expressions into integer token IDs.

    This is a minimal tokenizer for demonstrations.  A production system
    would use a proper tokenizer (BPE, SentencePiece, etc.) with a larger
    vocabulary.  The key property is that the tokenizer maps source text
    to integer IDs that the compiler's embedding layer can process.

    Vocabulary covers:
    - Arithmetic operators: + - *
    - Assignment: =
    - Register names: R0..R7
    - Digits: 0..9
    - Mnemonics: add, sub, mul, mov, halt, nop, cmp
    - Padding: <pad>
    """

    VOCAB: dict[str, int] = {
        "<pad>": 0,
        "+": 1,
        "-": 2,
        "*": 3,
        "=": 4,
        "r0": 5,
        "r1": 6,
        "r2": 7,
        "r3": 8,
        "r4": 9,
        "r5": 10,
        "r6": 11,
        "r7": 12,
        "0": 13,
        "1": 14,
        "2": 15,
        "3": 16,
        "4": 17,
        "5": 18,
        "6": 19,
        "7": 20,
        "8": 21,
        "9": 22,
        "add": 23,
        "sub": 24,
        "mul": 25,
        "mov": 26,
        "halt": 27,
        "nop": 28,
        "cmp": 29,
        "#": 30,
        ",": 31,
        ";": 32,
    }

    def __init__(self, pad_length: int = 32):
        """Initialize the tokenizer.

        Args:
            pad_length: Pad or truncate all sequences to this length.
        """
        self.pad_length = pad_length
        self._inv_vocab = {v: k for k, v in self.VOCAB.items()}

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return max(self.VOCAB.values()) + 1

    def tokenize(self, text: str) -> torch.Tensor:
        """Convert source text to a padded integer tensor.

        Splits on whitespace, lowercases everything, and maps each token
        to its vocabulary ID.  Unknown tokens map to 0 (<pad>).

        Args:
            text: Source code string.

        Returns:
            [pad_length] integer tensor of token IDs.
        """
        tokens: list[int] = []
        for word in text.lower().split():
            tokens.append(self.VOCAB.get(word, 0))
        # Pad or truncate
        while len(tokens) < self.pad_length:
            tokens.append(0)
        return torch.tensor(tokens[: self.pad_length], dtype=torch.long)

    def detokenize(self, token_ids: torch.Tensor) -> str:
        """Convert token IDs back to text.

        Args:
            token_ids: Integer tensor of token IDs.

        Returns:
            Reconstructed source text (without padding tokens).
        """
        words = []
        for tid in token_ids.tolist():
            if tid == 0:
                continue
            words.append(self._inv_vocab.get(int(tid), "?"))
        return " ".join(words)


# ---------------------------------------------------------------------------
# Demonstrations
# ---------------------------------------------------------------------------


def demo_differentiable_compilation() -> dict[str, object]:
    """Demonstrate training a neural compiler via differentiable execution.

    This demo trains a compiler to translate simple mnemonics ("add R0 R1 R2")
    into SoftPrograms that, when executed by the DifferentiableEngine, produce
    the correct arithmetic result.

    The compiler has never seen an instruction encoding.  It learns, purely
    from execution feedback, that "add" means the output register should
    contain the sum of the input registers, and it must emit the right
    combination of opcode logits, register selections, and immediate values
    to make that happen.

    Returns:
        Dictionary with training losses and test results.
    """
    print("=" * 60)
    print("Differentiable Compilation Pipeline")
    print("=" * 60)
    print(
        "Training a neural compiler by backpropagating through execution.\n"
        "The compiler learns to translate source mnemonics into programs\n"
        "that compute correctly -- trained end-to-end, no instruction\n"
        "labels needed.\n"
    )

    tokenizer = SimpleTokenizer()
    pipeline = DifferentiableCompilationPipeline(
        compiler=DifferentiableCompiler(vocab_size=tokenizer.vocab_size),
    )

    # --- Build training data ---
    # "add R0 R1 R2" with various input values -> R2 = R0 + R1
    training_data: list[
        tuple[torch.Tensor, dict[int, float], dict[int, float]]
    ] = []

    for a, b in [(3, 5), (7, 2), (10, 4), (1, 9), (6, 6), (0, 8)]:
        tokens = tokenizer.tokenize("add r0 r1 r2")
        training_data.append(
            (tokens, {0: float(a), 1: float(b)}, {2: float(a + b)})
        )

    # "mul R0 R1 R2" with various input values -> R2 = R0 * R1
    for a, b in [(3, 4), (5, 6), (2, 7), (8, 3), (4, 4), (1, 10)]:
        tokens = tokenizer.tokenize("mul r0 r1 r2")
        training_data.append(
            (tokens, {0: float(a), 1: float(b)}, {2: float(a * b)})
        )

    # "sub R0 R1 R2" with various input values -> R2 = R0 - R1
    for a, b in [(10, 3), (8, 2), (15, 5), (7, 7), (20, 11), (9, 4)]:
        tokens = tokenizer.tokenize("sub r0 r1 r2")
        training_data.append(
            (tokens, {0: float(a), 1: float(b)}, {2: float(a - b)})
        )

    # --- Train ---
    print(f"Training data: {len(training_data)} examples")
    print(f"  add: 6 examples, mul: 6 examples, sub: 6 examples")
    print()

    losses = pipeline.train_compiler(
        training_data,
        epochs=300,
        lr=0.003,
        temperature_start=2.0,
        temperature_end=0.3,
        entropy_weight=0.01,
        verbose=True,
    )

    print(f"\nTraining loss: {losses[0]:.2f} -> {losses[-1]:.4f}")

    # --- Test on held-out values ---
    print("\n--- Test Results (held-out values) ---")
    test_results: dict[str, dict] = {}

    with torch.no_grad():
        # Test addition
        test_tokens = tokenizer.tokenize("add r0 r1 r2")
        result = pipeline.compile_and_execute(
            test_tokens, {0: 15.0, 1: 25.0}, temperature=0.1
        )
        r2 = result.execution_result.registers[2].item()
        print(f"  add r0 r1 r2 | R0=15, R1=25 -> R2={r2:.2f} (expected 40)")
        test_results["add"] = {"actual": r2, "expected": 40.0}

        # Test multiplication
        test_tokens = tokenizer.tokenize("mul r0 r1 r2")
        result = pipeline.compile_and_execute(
            test_tokens, {0: 6.0, 1: 7.0}, temperature=0.1
        )
        r2 = result.execution_result.registers[2].item()
        print(f"  mul r0 r1 r2 | R0=6,  R1=7  -> R2={r2:.2f} (expected 42)")
        test_results["mul"] = {"actual": r2, "expected": 42.0}

        # Test subtraction
        test_tokens = tokenizer.tokenize("sub r0 r1 r2")
        result = pipeline.compile_and_execute(
            test_tokens, {0: 30.0, 1: 13.0}, temperature=0.1
        )
        r2 = result.execution_result.registers[2].item()
        print(f"  sub r0 r1 r2 | R0=30, R1=13 -> R2={r2:.2f} (expected 17)")
        test_results["sub"] = {"actual": r2, "expected": 17.0}

    # Show what the compiler actually emitted for "add"
    print("\n--- Compiled program for 'add r0 r1 r2' ---")
    with torch.no_grad():
        program = pipeline.compiler.compile(
            tokenizer.tokenize("add r0 r1 r2")
        )
        print(program.format_program())

    return {
        "loss_history": losses,
        "test_results": test_results,
    }


def demo_source_optimization() -> dict[str, object]:
    """Demonstrate optimizing source-level parameters through compilation.

    This demo shows that gradients flow all the way back to the source
    embedding.  We create a "parameterized source" by making the token
    embeddings differentiable inputs, then optimize them to produce a
    desired execution result.

    This is conceptually equivalent to: "what source code should I write
    to get output X?" answered by gradient descent.

    Returns:
        Dictionary with optimization trajectory.
    """
    print("\n" + "=" * 60)
    print("Source-Level Optimization Through Compilation")
    print("=" * 60)
    print(
        "Optimizing a continuous source embedding so that the compiled\n"
        "program produces a target output. Gradients flow from execution\n"
        "loss through the compiler back to the source representation.\n"
    )

    compiler = DifferentiableCompiler(vocab_size=64, d_model=64)
    engine = DifferentiableEngine()

    # Create a learnable "source embedding" -- a continuous vector that
    # we will optimize.  This bypasses the discrete tokenizer and feeds
    # directly into the encoder.
    source_embedding = nn.Parameter(torch.randn(8, 64) * 0.1)

    optimizer = torch.optim.Adam(
        list(compiler.parameters()) + [source_embedding], lr=0.01
    )

    target_r0 = 42.0
    loss_history: list[float] = []

    print("Target: R0 = 42.0 after execution")
    print("Optimizing source embedding + compiler jointly...\n")

    for step in range(200):
        optimizer.zero_grad()

        # Encode from continuous embedding (bypass tokenizer)
        encoded = compiler.encoder(source_embedding.unsqueeze(0))
        pooled = encoded.mean(dim=1).squeeze(0)
        program = compiler.decode(pooled)

        result = engine.execute_soft(
            program, {}, max_steps=16, temperature=1.0, skip_bitwise=True
        )

        loss = (result.registers[0] - target_r0) ** 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(compiler.parameters()) + [source_embedding], 5.0
        )
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % 50 == 0 or step == 199:
            r0_val = result.registers[0].item()
            print(f"  step {step:3d}  loss={loss_val:8.4f}  R0={r0_val:.2f}")

    final_r0 = result.registers[0].item()
    print(f"\nFinal R0 = {final_r0:.4f} (target = {target_r0})")
    print(f"Loss: {loss_history[0]:.2f} -> {loss_history[-1]:.6f}")

    return {
        "loss_history": loss_history,
        "final_r0": final_r0,
        "target_r0": target_r0,
    }


if __name__ == "__main__":
    demo_differentiable_compilation()
    demo_source_optimization()
