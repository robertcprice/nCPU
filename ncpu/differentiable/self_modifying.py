"""Self-modifying differentiable programs.

Programs that can write to their own instruction memory during execution,
with gradients flowing through the modification. This enables a form of
meta-programming where gradient descent optimizes not just what a program
computes, but how it modifies itself during execution.

Key innovation: instruction memory is a differentiable tensor. When a program
writes to its own instruction slots (via a STORE_INST opcode), the write
is a soft attention operation over instruction positions, and the new
instruction content is a differentiable projection from register values.
Gradients flow through the self-modification back to the values that
triggered it.

This is the intersection of genetic programming and differentiable computation,
but with real instruction semantics rather than tree-based GP.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .execution import (
    DifferentiableALU,
    DifferentiableEngine,
    OPCODES,
    NUM_OPCODES,
    SoftProgram,
    _OP,
)


# ---------------------------------------------------------------------------
# Extended opcode table for self-modification
# ---------------------------------------------------------------------------

# STORE_INST and LOAD_INST are virtual opcodes that live in the *same*
# opcode probability space as the base ISA.  We extend the opcode table
# at module level so that SelfModifyingProgram logits have the right width.

SELF_MOD_OPCODES: dict[str, int] = {
    **OPCODES,
    "STORE_INST": NUM_OPCODES,      # write register values as new instruction
    "LOAD_INST": NUM_OPCODES + 1,   # read instruction at position into registers
}
NUM_SELF_MOD_OPCODES = len(SELF_MOD_OPCODES)
_SELF_MOD_OP = {v: k for k, v in SELF_MOD_OPCODES.items()}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SelfModifyingResult:
    """Result of self-modifying program execution.

    Attributes:
        final_registers: Differentiable register state after execution.
        final_program: Snapshot of the (potentially modified) opcode logits
            at program termination.  Detached from the graph because it is
            informational only.
        modification_log: Per-step records of self-modification events.
            Each entry contains the step index, the soft modification
            probability, and summary statistics of the write.
        steps_executed: Number of execution steps that ran.
        halted: Whether the program reached a cumulative halt probability
            exceeding 0.5.
        register_trace: Per-step snapshots of the register file (detached).
    """

    final_registers: torch.Tensor
    final_program: torch.Tensor          # [max_len, num_opcodes] detached
    modification_log: list[dict] = field(default_factory=list)
    steps_executed: int = 0
    halted: bool = False
    register_trace: list[torch.Tensor] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Self-modifying program representation
# ---------------------------------------------------------------------------

class SelfModifyingProgram(nn.Module):
    """A program that can modify its own instructions during execution.

    The instruction memory is a continuous tensor ``[max_length, features]``
    where features encode opcode probabilities, register selections, and
    immediates.  Self-modification writes to this tensor using soft attention,
    so the operation is fully differentiable.

    Extended opcodes (beyond the standard 14):

    - **STORE_INST**: Write register values as a new instruction at a target
      position.  ``dst`` selects the target instruction slot (via soft
      attention), ``src1`` provides the value that is projected into new
      opcode logits, and ``src2`` provides the new immediate value.

    - **LOAD_INST**: Read the instruction at the position indicated by
      ``src1`` into the destination register.  The "instruction value"
      is a scalar summary of the opcode logits at that position (their
      argmax index, made differentiable via soft-argmax).

    Both operations maintain full gradient flow.
    """

    def __init__(
        self,
        max_length: int = 16,
        num_registers: int = 8,
        init_program: SoftProgram | None = None,
        init_scale: float = 0.1,
    ):
        super().__init__()
        self.max_length = max_length
        self.num_registers = num_registers

        if init_program is not None:
            # Copy from an existing SoftProgram, padding the opcode dimension
            # from NUM_OPCODES to NUM_SELF_MOD_OPCODES.
            pad_len = max_length - init_program.max_length
            if pad_len < 0:
                raise ValueError(
                    f"init_program has length {init_program.max_length} "
                    f"but max_length is {max_length}"
                )

            def _pad_rows(t: torch.Tensor, rows: int, cols: int | None = None) -> torch.Tensor:
                """Pad a 2-D parameter tensor with small random values."""
                if rows > 0:
                    extra = torch.randn(rows, t.shape[1]) * init_scale
                    t = torch.cat([t, extra], dim=0)
                if cols is not None and cols > t.shape[1]:
                    extra = torch.randn(t.shape[0], cols - t.shape[1]) * init_scale
                    t = torch.cat([t, extra], dim=1)
                return t

            self.opcode_logits = nn.Parameter(
                _pad_rows(
                    init_program.opcode_logits.data.clone(),
                    pad_len,
                    NUM_SELF_MOD_OPCODES,
                )
            )
            self.dst_logits = nn.Parameter(
                _pad_rows(init_program.dst_logits.data.clone(), pad_len)
            )
            self.src1_logits = nn.Parameter(
                _pad_rows(init_program.src1_logits.data.clone(), pad_len)
            )
            self.src2_logits = nn.Parameter(
                _pad_rows(init_program.src2_logits.data.clone(), pad_len)
            )
            imm = init_program.immediates.data.clone()
            if pad_len > 0:
                imm = torch.cat([imm, torch.zeros(pad_len)])
            self.immediates = nn.Parameter(imm)

            bl = init_program.branch_logits.data.clone()
            # Resize branch logits to [max_length, max_length]
            bl_new = torch.randn(max_length, max_length) * init_scale
            bl_new[: bl.shape[0], : bl.shape[1]] = bl
            self.branch_logits = nn.Parameter(bl_new)
        else:
            self.opcode_logits = nn.Parameter(
                torch.randn(max_length, NUM_SELF_MOD_OPCODES) * init_scale
            )
            self.dst_logits = nn.Parameter(
                torch.randn(max_length, num_registers) * init_scale
            )
            self.src1_logits = nn.Parameter(
                torch.randn(max_length, num_registers) * init_scale
            )
            self.src2_logits = nn.Parameter(
                torch.randn(max_length, num_registers) * init_scale
            )
            self.immediates = nn.Parameter(torch.zeros(max_length))
            self.branch_logits = nn.Parameter(
                torch.randn(max_length, max_length) * init_scale
            )

        # --- Learned projections for self-modification ---
        # These map register scalar values into instruction-space features.
        # They are the core learned component that controls *how* the program
        # rewrites itself.
        self.reg_to_opcode = nn.Linear(1, NUM_SELF_MOD_OPCODES)
        self.reg_to_dst = nn.Linear(1, num_registers)
        self.reg_to_src = nn.Linear(1, num_registers)

    def get_soft_instruction(
        self,
        pc_weights: torch.Tensor,
        inst_opcodes: torch.Tensor,
        inst_imm: torch.Tensor,
        temperature: float = 1.0,
        inst_dst: torch.Tensor | None = None,
        inst_src1: torch.Tensor | None = None,
        inst_src2: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fetch a soft instruction from (potentially modified) instruction memory.

        Unlike :class:`SoftProgram`, this method reads from *mutable* opcode,
        immediate, dst, and src tensors that may have been rewritten by
        earlier STORE_INST operations.

        Args:
            pc_weights: ``[max_length]`` soft PC distribution.
            inst_opcodes: ``[max_length, num_opcodes]`` current opcode logits
                (may differ from ``self.opcode_logits`` after self-modification).
            inst_imm: ``[max_length]`` current immediate values.
            temperature: Gumbel-softmax temperature.
            inst_dst: ``[max_length, num_registers]`` current dst logits.
                If ``None``, uses ``self.dst_logits``.
            inst_src1: ``[max_length, num_registers]`` current src1 logits.
                If ``None``, uses ``self.src1_logits``.
            inst_src2: ``[max_length, num_registers]`` current src2 logits.
                If ``None``, uses ``self.src2_logits``.

        Returns:
            ``(opcode_weights, dst_weights, src1_weights, src2_weights,
            immediate, branch_weights)`` -- all differentiable.
        """
        pc = pc_weights.unsqueeze(0)  # [1, max_length]

        opcode_probs = F.gumbel_softmax(inst_opcodes, tau=temperature, hard=False)
        opcode_weights = (pc @ opcode_probs).squeeze(0)

        dst_logits = inst_dst if inst_dst is not None else self.dst_logits
        dst_probs = F.softmax(dst_logits, dim=-1)
        dst_weights = (pc @ dst_probs).squeeze(0)

        src1_logits = inst_src1 if inst_src1 is not None else self.src1_logits
        src1_probs = F.softmax(src1_logits, dim=-1)
        src1_weights = (pc @ src1_probs).squeeze(0)

        src2_logits = inst_src2 if inst_src2 is not None else self.src2_logits
        src2_probs = F.softmax(src2_logits, dim=-1)
        src2_weights = (pc @ src2_probs).squeeze(0)

        immediate = (pc_weights * inst_imm).sum()

        branch_probs = F.softmax(self.branch_logits, dim=-1)
        branch_weights = (pc @ branch_probs).squeeze(0)

        return (
            opcode_weights,
            dst_weights,
            src1_weights,
            src2_weights,
            immediate,
            branch_weights,
        )

    def project_instruction(
        self, opcode_source: torch.Tensor, operand_source: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project scalar register values into instruction-space features.

        This is the differentiable "instruction constructor": it takes two
        register scalars and produces soft opcode logits, soft dst logits, and
        soft src logits that can be written into instruction memory.

        Args:
            opcode_source: Scalar tensor -- register value controlling the
                opcode of the new instruction.
            operand_source: Scalar tensor -- register value controlling the
                operand selections of the new instruction.

        Returns:
            ``(new_opcode_logits, new_dst_logits, new_src_logits)`` as 1-D
            tensors ready to blend into instruction memory.
        """
        new_opcode = self.reg_to_opcode(opcode_source.unsqueeze(0))  # [1, num_opcodes]
        new_dst = self.reg_to_dst(operand_source.unsqueeze(0))        # [1, num_regs]
        new_src = self.reg_to_src(operand_source.unsqueeze(0))        # [1, num_regs]
        return new_opcode.squeeze(0), new_dst.squeeze(0), new_src.squeeze(0)

    def extract_discrete_program(self, inst_opcodes: torch.Tensor | None = None) -> str:
        """Pretty-print the most likely discrete program.

        Args:
            inst_opcodes: Optionally pass the (modified) opcode logits.
                If ``None``, uses the original ``self.opcode_logits``.
        """
        if inst_opcodes is None:
            inst_opcodes = self.opcode_logits.data
        lines: list[str] = []
        for i in range(self.max_length):
            op_idx = int(inst_opcodes[i].argmax().item())
            name = _SELF_MOD_OP.get(op_idx, f"OP{op_idx}")
            dst = int(self.dst_logits[i].argmax().item())
            src1 = int(self.src1_logits[i].argmax().item())
            src2 = int(self.src2_logits[i].argmax().item())
            imm = float(self.immediates[i].item())
            branch = int(self.branch_logits[i].argmax().item())

            if name == "HALT":
                lines.append(f"  {i:2d}: HALT")
            elif name == "NOP":
                lines.append(f"  {i:2d}: NOP")
            elif name == "MOV_IMM":
                lines.append(f"  {i:2d}: MOV R{dst}, #{imm:.1f}")
            elif name == "MOV_REG":
                lines.append(f"  {i:2d}: MOV R{dst}, R{src1}")
            elif name in ("ADD", "SUB", "MUL", "AND", "OR", "XOR"):
                lines.append(f"  {i:2d}: {name} R{dst}, R{src1}, R{src2}")
            elif name == "CMP":
                lines.append(f"  {i:2d}: CMP R{src1}, R{src2}")
            elif name in ("BEQ", "BNE", "BGT"):
                lines.append(f"  {i:2d}: {name} @{branch}")
            elif name == "STORE_INST":
                lines.append(f"  {i:2d}: STORE_INST [@R{dst}], R{src1}, R{src2}")
            elif name == "LOAD_INST":
                lines.append(f"  {i:2d}: LOAD_INST R{dst}, [@R{src1}]")
            else:
                lines.append(f"  {i:2d}: {name} ???")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Self-modifying execution engine
# ---------------------------------------------------------------------------

class SelfModifyingEngine(nn.Module):
    """Execute self-modifying programs with full gradient flow.

    Extends the approach of :class:`DifferentiableEngine` with support for
    programs that modify their own instruction memory.  The instruction
    memory is treated as a mutable differentiable tensor, and
    self-modification operations (STORE_INST) perform soft writes that
    maintain gradient flow.

    **Self-modification mechanism**:

    At each execution step the engine checks the soft probability of the
    STORE_INST opcode.  When non-negligible, it performs a differentiable
    write to instruction memory:

    1. **Target position**: The ``dst`` register distribution is used to
       compute a soft attention vector over instruction positions via a
       Gaussian kernel centered on the weighted register index.

    2. **New opcode logits**: The value in the ``src1`` register is
       projected through a learned linear layer
       (:meth:`SelfModifyingProgram.project_instruction`) to produce
       new opcode logits for the target slot.

    3. **New immediate**: The value in the ``src2`` register is written
       directly as the new immediate.

    4. **Blending**: The write is interpolated with the existing
       instruction content using ``modification_strength * store_prob``
       as the interpolation weight, keeping the operation smooth and
       differentiable.

    **LOAD_INST mechanism**:

    When the soft LOAD_INST probability is non-negligible, the engine
    reads the instruction at the position indicated by ``src1`` and
    writes a scalar summary (soft-argmax of the opcode logits) into the
    destination register.  This allows the program to *inspect* its own
    code and branch/compute based on what instructions exist.

    Both mechanisms maintain full gradient flow: backpropagation through
    the execution trace will produce gradients on the program parameters,
    the projection layers, and the initial register values.
    """

    def __init__(
        self,
        num_registers: int = 8,
        n_bits: int = 16,
        modification_strength: float = 0.8,
    ):
        super().__init__()
        self.num_registers = num_registers
        self.alu = DifferentiableALU(n_bits)
        self.modification_strength = modification_strength

    # -- Soft attention helpers ------------------------------------------------

    @staticmethod
    def _position_attention(
        register_weights: torch.Tensor,
        num_positions: int,
        num_registers: int,
        sharpness: float = 2.0,
    ) -> torch.Tensor:
        """Compute a soft attention distribution over instruction positions.

        Uses the register selection weights to compute a "target index"
        (weighted sum of register indices), then produces a Gaussian
        attention vector peaked at that index.

        Args:
            register_weights: ``[num_registers]`` soft register selection.
            num_positions: Number of instruction slots.
            num_registers: Number of registers.
            sharpness: Controls how peaked the attention is.

        Returns:
            ``[num_positions]`` soft attention distribution (sums to 1).
        """
        # Weighted register index -> target position
        reg_indices = torch.arange(num_registers, dtype=torch.float32)
        target = (register_weights * reg_indices).sum()

        # Scale to instruction position range
        scale = (num_positions - 1) / max(num_registers - 1, 1)
        target_pos = target * scale

        # Gaussian attention over positions
        positions = torch.arange(num_positions, dtype=torch.float32)
        logits = -sharpness * (positions - target_pos) ** 2
        return F.softmax(logits, dim=-1)

    # -- Main execution loop ---------------------------------------------------

    def execute(
        self,
        program: SelfModifyingProgram,
        inputs: dict[int, float] | None = None,
        max_steps: int = 32,
        temperature: float = 1.0,
        skip_bitwise: bool = True,
    ) -> SelfModifyingResult:
        """Execute a self-modifying program.

        At each step the engine:

        1. Fetches a soft instruction from (potentially modified) instruction
           memory.
        2. Executes the instruction through the ALU (all operations computed
           in parallel, weighted by opcode probabilities).
        3. Checks for STORE_INST: if the soft probability is non-trivial,
           performs a differentiable write to instruction memory.
        4. Checks for LOAD_INST: if the soft probability is non-trivial,
           reads instruction content into a register.
        5. Updates flags, PC, and halt accumulator.

        The entire execution loop is differentiable.  Gradients flow
        through self-modification writes, register reads/writes, and PC
        updates.

        Args:
            program: A :class:`SelfModifyingProgram` with learnable
                parameters and projection layers.
            inputs: Initial register values as ``{register_index: value}``.
            max_steps: Maximum execution steps before forced termination.
            temperature: Gumbel-softmax temperature for opcode selection.
                Lower values produce more discrete behaviour.
            skip_bitwise: Skip expensive bitwise operations (AND/OR/XOR)
                for faster training when they are not needed.

        Returns:
            :class:`SelfModifyingResult` with differentiable ``final_registers``.
        """
        if inputs is None:
            inputs = {}

        # Initialize registers
        regs = torch.zeros(self.num_registers)
        for idx, val in inputs.items():
            regs[idx] = float(val) if not isinstance(val, torch.Tensor) else val

        # Mutable instruction memory -- clone from parameters so that
        # in-place modifications build a new computation graph rather than
        # mutating the parameter tensors.
        inst_opcodes = program.opcode_logits.clone()  # [max_length, num_opcodes]
        inst_imm = program.immediates.clone()          # [max_length]
        inst_dst = program.dst_logits.clone()          # [max_length, num_registers]
        inst_src1 = program.src1_logits.clone()        # [max_length, num_registers]
        inst_src2 = program.src2_logits.clone()        # [max_length, num_registers]

        # Soft PC
        pc = torch.zeros(program.max_length)
        pc[0] = 1.0

        flags = torch.zeros(4)  # N, Z, C, V
        modification_log: list[dict] = []
        trace: list[torch.Tensor] = [regs.detach().clone()]
        cumulative_halt = torch.tensor(0.0)
        step = -1

        for step in range(max_steps):
            # --- Fetch soft instruction from mutable memory ---------------
            (
                opcode_w, dst_w, src1_w, src2_w, immediate, branch_w
            ) = program.get_soft_instruction(
                pc, inst_opcodes, inst_imm, temperature,
                inst_dst=inst_dst, inst_src1=inst_src1, inst_src2=inst_src2,
            )

            # --- Read source operands via soft attention -------------------
            src1_val = (src1_w * regs).sum()
            src2_val = (src2_w * regs).sum()

            # --- Compute ALL base ALU operations in parallel ---------------
            all_results = self.alu.compute_all(
                src1_val, src2_val, immediate, skip_bitwise=skip_bitwise
            )

            # Weighted result from base opcodes only
            result = torch.tensor(0.0)
            for op_name, op_idx in OPCODES.items():
                result = result + opcode_w[op_idx] * all_results[op_name]

            # --- Standard register write (base ALU ops) --------------------
            write_ops = {
                "MOV_IMM", "MOV_REG", "ADD", "SUB", "MUL",
                "AND", "OR", "XOR",
            }
            write_enable = sum(opcode_w[OPCODES[op]] for op in write_ops)
            w = dst_w * write_enable
            regs = regs * (1.0 - w) + result * w

            # =============================================================
            # SELF-MODIFICATION: STORE_INST
            # =============================================================
            store_prob = opcode_w[SELF_MOD_OPCODES["STORE_INST"]]

            # Always compute the modification to keep the graph connected
            # for gradient flow.  The modification is scaled by store_prob
            # so when STORE_INST is unlikely the effect is negligible.

            # Target position: soft attention from dst register weights
            target_attention = self._position_attention(
                dst_w, program.max_length, self.num_registers,
            )

            # New instruction content projected from register values
            new_op_logits, new_dst_logits, new_src_logits = (
                program.project_instruction(src1_val, src2_val)
            )

            # New immediate from src2 register value
            new_imm = src2_val

            # Differentiable soft write to opcode logits:
            #   inst_opcodes[pos] += alpha * target_attention[pos] * (new - old)
            alpha = store_prob * self.modification_strength
            target_col = target_attention.unsqueeze(1)  # [max_length, 1]
            opcode_delta = new_op_logits.unsqueeze(0) - inst_opcodes  # [max_length, num_opcodes]
            inst_opcodes = inst_opcodes + alpha * target_col * opcode_delta

            # Differentiable soft write to dst logits:
            dst_delta = new_dst_logits.unsqueeze(0) - inst_dst  # [max_length, num_registers]
            inst_dst = inst_dst + alpha * target_col * dst_delta

            # Differentiable soft write to src logits (src1 and src2 share
            # the same projected values -- the projection captures "operand
            # selection" broadly):
            src_delta_1 = new_src_logits.unsqueeze(0) - inst_src1
            inst_src1 = inst_src1 + alpha * target_col * src_delta_1

            src_delta_2 = new_src_logits.unsqueeze(0) - inst_src2
            inst_src2 = inst_src2 + alpha * target_col * src_delta_2

            # Differentiable soft write to immediates:
            imm_delta = new_imm - inst_imm  # [max_length] broadcast scalar
            inst_imm = inst_imm + alpha * target_attention * imm_delta

            # Log the modification (detached scalars for diagnostics only)
            modification_log.append({
                "step": step,
                "store_prob": store_prob.detach().item(),
                "target_entropy": -(target_attention * (target_attention + 1e-8).log()).sum().detach().item(),
                "alpha": alpha.detach().item(),
            })

            # =============================================================
            # SELF-MODIFICATION: LOAD_INST
            # =============================================================
            load_prob = opcode_w[SELF_MOD_OPCODES["LOAD_INST"]]

            # Read instruction at position indicated by src1.
            # Position attention from src1 register weights.
            read_attention = self._position_attention(
                src1_w, program.max_length, self.num_registers,
            )

            # Soft-argmax of opcode logits at the attended position:
            # weighted sum of opcode indices, producing a scalar "instruction
            # identity" that the program can branch on.
            opcode_probs_at_pos = F.softmax(inst_opcodes, dim=-1)       # [L, O]
            attended_probs = (read_attention.unsqueeze(1) * opcode_probs_at_pos).sum(0)  # [O]
            opcode_indices = torch.arange(NUM_SELF_MOD_OPCODES, dtype=torch.float32)
            inst_value = (attended_probs * opcode_indices).sum()

            # Write the instruction value into the destination register,
            # scaled by the LOAD_INST probability.
            load_w = dst_w * load_prob
            regs = regs * (1.0 - load_w) + inst_value * load_w

            # --- Update flags (CMP) ----------------------------------------
            cmp_weight = opcode_w[OPCODES["CMP"]]
            new_flags = self.alu.compute_flags(src1_val, src2_val, 0.1)
            flags = flags * (1.0 - cmp_weight) + new_flags * cmp_weight

            # --- Update PC --------------------------------------------------
            next_pc = torch.roll(pc, 1, dims=0)

            beq_prob = opcode_w[OPCODES["BEQ"]] * flags[1]
            bne_prob = opcode_w[OPCODES["BNE"]] * (1.0 - flags[1])
            bgt_prob = opcode_w[OPCODES["BGT"]] * (1.0 - flags[0]) * (1.0 - flags[1])
            branch_prob = (beq_prob + bne_prob + bgt_prob).clamp(0.0, 1.0)

            pc = next_pc * (1.0 - branch_prob) + branch_w * branch_prob

            # --- Halt accumulation -------------------------------------------
            halt_prob = opcode_w[OPCODES["HALT"]]
            cumulative_halt = cumulative_halt + halt_prob * (1.0 - cumulative_halt)

            trace.append(regs.detach().clone())

            if cumulative_halt.item() > 0.99:
                break

        return SelfModifyingResult(
            final_registers=regs,
            final_program=inst_opcodes.detach(),
            modification_log=modification_log,
            steps_executed=step + 1,
            halted=cumulative_halt.item() > 0.5,
            register_trace=trace,
        )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_self_modifying(
    program: SelfModifyingProgram,
    engine: SelfModifyingEngine,
    target_fn: callable,
    input_specs: list[dict[int, float]],
    output_register: int = 0,
    lr: float = 0.01,
    steps: int = 500,
    max_exec_steps: int = 16,
    temperature_schedule: callable | None = None,
    verbose: bool = True,
) -> list[float]:
    """Train a self-modifying program to match a target function.

    This is a convenience wrapper that handles the optimisation loop,
    temperature annealing, and gradient clipping.

    Args:
        program: The :class:`SelfModifyingProgram` to optimise.
        engine: The :class:`SelfModifyingEngine` to execute with.
        target_fn: Callable that takes an input dict and returns the
            desired scalar output value.
        input_specs: List of input dicts to train on (each maps register
            index to float value).
        output_register: Which register should match the target.
        lr: Learning rate for Adam.
        steps: Number of training steps.
        max_exec_steps: Maximum execution steps per program run.
        temperature_schedule: Optional callable ``step -> temperature``.
            Defaults to linear annealing from 1.0 to 0.1.
        verbose: Print progress every 50 steps.

    Returns:
        List of per-step loss values.
    """
    optimizer = torch.optim.Adam(
        list(program.parameters()) + list(engine.parameters()), lr=lr
    )

    if temperature_schedule is None:
        def temperature_schedule(s: int) -> float:
            return max(0.1, 1.0 - 0.9 * s / max(steps - 1, 1))

    losses: list[float] = []

    for s in range(steps):
        optimizer.zero_grad()
        temp = temperature_schedule(s)
        total_loss = torch.tensor(0.0)

        for inp in input_specs:
            result = engine.execute(program, inp, max_steps=max_exec_steps, temperature=temp)
            target = target_fn(inp)
            total_loss = total_loss + (result.final_registers[output_register] - target) ** 2

        total_loss = total_loss / len(input_specs)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(program.parameters(), 5.0)
        optimizer.step()

        loss_val = total_loss.detach().item()
        losses.append(loss_val)

        if verbose and s % 50 == 0:
            result = engine.execute(program, input_specs[0], max_steps=max_exec_steps, temperature=temp)
            n_mods = sum(
                1 for m in result.modification_log if m["store_prob"] > 0.1
            )
            print(
                f"  step {s:4d}  loss={loss_val:.4f}  "
                f"temp={temp:.3f}  "
                f"R{output_register}={result.final_registers[output_register].item():.3f}  "
                f"active_mods={n_mods}"
            )

    return losses


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_self_modifying() -> None:
    """Demo: a program that modifies its own constants during execution.

    The program starts with random instructions.  Through gradient descent,
    it learns to use STORE_INST to rewrite its own immediate values and
    produce a target output.  The optimiser discovers both:

    - What the initial instructions should be.
    - How the program should modify itself mid-execution to reach the goal.

    This demonstrates that gradient descent can optimise self-modification
    behaviour -- a capability that has no analogue in conventional program
    synthesis.
    """
    torch.manual_seed(42)

    print("=" * 60)
    print("Self-Modifying Differentiable Program")
    print("=" * 60)
    print()
    print("A program that rewrites its own instructions during execution,")
    print("with gradients flowing through the self-modification.")
    print()

    prog = SelfModifyingProgram(max_length=6)
    engine = SelfModifyingEngine(modification_strength=0.8)

    # Target: make the program produce R0 = 42 with no inputs.
    print("Task: discover a self-modifying program where R0 = 42")
    print()

    losses = train_self_modifying(
        program=prog,
        engine=engine,
        target_fn=lambda _: 42.0,
        input_specs=[{}],
        output_register=0,
        lr=0.02,
        steps=150,
        max_exec_steps=12,
        verbose=True,
    )

    # Final evaluation at low temperature (near-discrete)
    result = engine.execute(prog, {}, max_steps=12, temperature=0.1)
    print()
    print(f"Final R0 = {result.final_registers[0].item():.4f}  (target: 42)")
    print(f"Steps executed: {result.steps_executed}")
    print(f"Halted: {result.halted}")
    print()

    active_mods = [m for m in result.modification_log if m["store_prob"] > 0.1]
    print(f"Self-modifications during execution: {len(active_mods)}")
    for mod in active_mods[:5]:
        print(
            f"  Step {mod['step']}: "
            f"store_prob={mod['store_prob']:.3f}, "
            f"alpha={mod['alpha']:.3f}, "
            f"target_entropy={mod['target_entropy']:.3f}"
        )

    print()
    print("Learned program (most likely discrete form):")
    print(prog.extract_discrete_program(result.final_program))
    print()

    # ---------------------------------------------------------------
    # Second demo: learn f(x) = 2x + 1 with self-modification
    # ---------------------------------------------------------------
    print("=" * 60)
    print("Task 2: f(x) = 2x + 1 via self-modifying program")
    print("=" * 60)
    print()

    prog2 = SelfModifyingProgram(max_length=6)
    engine2 = SelfModifyingEngine(modification_strength=0.8)

    train_inputs = [{0: float(x)} for x in range(1, 4)]

    losses2 = train_self_modifying(
        program=prog2,
        engine=engine2,
        target_fn=lambda inp: 2.0 * inp.get(0, 0.0) + 1.0,
        input_specs=train_inputs,
        output_register=1,
        lr=0.02,
        steps=200,
        max_exec_steps=12,
        verbose=True,
    )

    print()
    print("Generalisation test:")
    for x in [0, 3, 7, 10]:
        result = engine2.execute(prog2, {0: float(x)}, max_steps=12, temperature=0.1)
        expected = 2.0 * x + 1.0
        actual = result.final_registers[1].item()
        n_mods = sum(1 for m in result.modification_log if m["store_prob"] > 0.1)
        print(
            f"  f({x}) = {actual:.2f}  "
            f"(expected {expected:.1f}, "
            f"err={abs(actual - expected):.3f}, "
            f"mods={n_mods})"
        )


if __name__ == "__main__":
    demo_self_modifying()
